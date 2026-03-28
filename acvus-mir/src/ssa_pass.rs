//! SSA Pass (mem2reg for context and local variables)
//!
//! Promotes ContextProject/ContextLoad/ContextStore and VarStore/VarLoad/ParamLoad to SSA form.
//! Self-contained: inserts initial loads, computes PHIs, patches instructions.
//!
//! Write-back model (context only): branch-internal ContextStores are removed;
//! a single write-back ContextStore is inserted after each merge block.
//! Local variables do NOT need write-back — they exist only in SSA form.

use acvus_utils::Astr;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, BTreeSet};

use crate::analysis::cfg::{BlockIdx, Cfg, Terminator};
use crate::graph::QualifiedRef;
use crate::ir::{Callee, Inst, InstKind, Label, MirBody, ValueId};
use crate::ssa::{ENTRY_BLOCK, SSABuilder, SsaVar};
use crate::ty::{Effect, Ty};

/// Run the SSA context pass on a MirBody.
///
/// `fn_types` maps FunctionId → Ty for resolving callee effects
/// (which contexts a function reads/writes). Used to populate
/// `context_uses`/`context_defs` on FunctionCall/Spawn instructions.
pub fn run(body: &mut MirBody, fn_types: &FxHashMap<QualifiedRef, Ty>) {
    // Step 1: Build CFG + collect context + local variable ops.
    let cfg = Cfg::build(&body.insts);
    if cfg.blocks.is_empty() {
        return;
    }
    let ssa_info = collect_ssa_info(&cfg, &body.insts, &body.val_types);

    // Step 2: Run SSABuilder + patch PHIs (only if there are writes + merge points).
    let var_subst = if !ssa_info.written_contexts.is_empty() || !ssa_info.written_vars.is_empty() {
        let preds = cfg.predecessors();
        let (phi_insertions, var_subst, undef_defs) = run_ssa_builder(
            &cfg,
            &preds,
            &ssa_info,
            &mut body.val_factory,
            &mut body.val_types,
        );
        if !phi_insertions.is_empty() {
            patch_instructions(body, &cfg, &phi_insertions, &ssa_info);
        }
        // Emit Undef instructions at the start of the entry block for SSA initial values.
        if !undef_defs.is_empty() {
            let undef_insts: Vec<Inst> = undef_defs
                .into_iter()
                .map(|dst| Inst {
                    span: acvus_ast::Span { start: 0, end: 0 },
                    kind: InstKind::Undef { dst },
                })
                .collect();
            body.insts.splice(0..0, undef_insts);
        }
        var_subst
    } else {
        FxHashMap::default()
    };

    // Step 3: Forward context values — eliminate redundant loads + populate call context bindings.
    let fwd_subst = forward_context_values(body, fn_types);

    // Step 4: Apply VarLoad/ParamLoad substitutions and remove var instructions.
    // Chain var_subst through forward's subst: if var_subst maps r10→r5 and
    // forward maps r5→r3, the effective mapping is r10→r3.
    if !var_subst.is_empty() {
        let chained: FxHashMap<ValueId, ValueId> = var_subst
            .into_iter()
            .map(|(from, to)| {
                let final_target = fwd_subst.get(&to).copied().unwrap_or(to);
                (from, final_target)
            })
            .collect();
        apply_var_subst(body, &chained);
    }
}

/// Store-load forwarding for context variables.
///
/// Tracks the "current SSA value" of each context. When a ContextLoad
/// follows a ContextStore to the same context (with no intervening branch),
/// the load is replaced by the stored value.
///
/// Also eliminates the initial entry load if the context is never read
/// before it's written (dead initial load).
fn forward_context_values(
    body: &mut MirBody,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
) -> FxHashMap<ValueId, ValueId> {
    // Map: projection ValueId → QualifiedRef.
    let mut val_to_ctx: FxHashMap<ValueId, QualifiedRef> = FxHashMap::default();
    // Current known value per context (from entry load or store).
    let mut current_val: FxHashMap<QualifiedRef, ValueId> = FxHashMap::default();
    // ValueId substitutions: old → new.
    let mut subst: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    // Instructions to remove (dead loads + their preceding projects).
    let mut remove: FxHashSet<usize> = FxHashSet::default();
    // Deferred patches: (inst_index, new context_uses, new context_defs).
    let mut call_patches: Vec<(
        usize,
        Vec<(QualifiedRef, ValueId)>,
        Vec<(QualifiedRef, ValueId)>,
    )> = Vec::new();

    for (i, inst) in body.insts.iter().enumerate() {
        match &inst.kind {
            InstKind::ContextProject { dst, ctx, .. } => {
                val_to_ctx.insert(*dst, *ctx);
            }
            InstKind::ContextLoad { dst, src } => {
                let src_resolved = subst.get(src).copied().unwrap_or(*src);
                if let Some(&ctx_id) = val_to_ctx.get(&src_resolved) {
                    if let Some(&known_val) = current_val.get(&ctx_id) {
                        // We already know this context's value — substitute.
                        subst.insert(*dst, known_val);
                        remove.insert(i);
                        // Also remove the preceding ContextProject if it was just for this load.
                        if i > 0
                            && matches!(body.insts[i - 1].kind, InstKind::ContextProject { dst: proj_dst, .. } if proj_dst == src_resolved)
                        {
                            remove.insert(i - 1);
                        }
                    } else {
                        // First load of this context — record it.
                        current_val.insert(ctx_id, *dst);
                    }
                }
            }
            InstKind::ContextStore { dst, value } => {
                let dst_resolved = subst.get(dst).copied().unwrap_or(*dst);
                let value_resolved = subst.get(value).copied().unwrap_or(*value);
                if let Some(&ctx_id) = val_to_ctx.get(&dst_resolved) {
                    current_val.insert(ctx_id, value_resolved);
                }
            }

            // FunctionCall with Direct callee: populate context_uses/context_defs from effect.
            InstKind::FunctionCall {
                callee: Callee::Direct(fn_id),
                context_uses,
                context_defs,
                ..
            } if context_uses.is_empty() && context_defs.is_empty() => {
                if let Some(reads_writes) = extract_effect_refs(fn_types, fn_id) {
                    let (reads, writes) = reads_writes;
                    // Populate context_uses: bind current SSA value for each read.
                    let mut uses = Vec::new();
                    for qref in &reads {
                        if let Some(&val) = current_val.get(qref) {
                            uses.push((*qref, val));
                        }
                    }
                    // Populate context_defs: allocate new ValueId for each write.
                    let mut defs = Vec::new();
                    for qref in &writes {
                        let new_val = body.val_factory.next();
                        // Copy type from current SSA value (if known).
                        if let Some(&current) = current_val.get(qref) {
                            if let Some(ty) = body.val_types.get(&current) {
                                body.val_types.insert(new_val, ty.clone());
                            }
                        }
                        defs.push((*qref, new_val));
                        // Update current_val — subsequent loads see the new value.
                        current_val.insert(*qref, new_val);
                    }
                    if !uses.is_empty() || !defs.is_empty() {
                        call_patches.push((i, uses, defs));
                    }
                }
            }

            // Branch/merge: invalidate tracked values (conservative).
            InstKind::BlockLabel { .. } | InstKind::Jump { .. } | InstKind::JumpIf { .. } => {
                current_val.clear();
            }
            _ => {}
        }
    }

    if remove.is_empty() && subst.is_empty() && call_patches.is_empty() {
        return subst;
    }

    // Apply call patches (context_uses/context_defs population).
    let patch_map: FxHashMap<usize, _> = call_patches
        .into_iter()
        .map(|(i, u, d)| (i, (u, d)))
        .collect();

    // Apply substitutions, patches, and remove dead instructions.
    let old_insts = std::mem::take(&mut body.insts);
    body.insts = old_insts
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !remove.contains(i))
        .map(|(i, mut inst)| {
            apply_subst(&mut inst.kind, &subst);
            // Apply context_uses/context_defs patches.
            if let Some((uses, defs)) = patch_map.get(&i) {
                if let InstKind::FunctionCall {
                    context_uses,
                    context_defs,
                    ..
                } = &mut inst.kind
                {
                    *context_uses = uses.clone();
                    *context_defs = defs.clone();
                }
            }
            inst
        })
        .collect();

    subst
}

/// Extract reads/writes QualifiedRefs from a function's type effect.
///
/// Only `EffectTarget::Context` refs are returned — `Token` targets are NOT
/// SSA-compatible and must never be converted to context_uses/context_defs.
fn extract_effect_refs(
    fn_types: &FxHashMap<QualifiedRef, Ty>,
    fn_id: &QualifiedRef,
) -> Option<(Vec<QualifiedRef>, Vec<QualifiedRef>)> {
    use crate::ty::EffectTarget;

    let ty = fn_types.get(fn_id)?;
    let Ty::Fn {
        effect: Effect::Resolved(eff),
        ..
    } = ty
    else {
        return None;
    };
    let reads: Vec<QualifiedRef> = eff
        .reads
        .iter()
        .filter_map(|t| match t {
            EffectTarget::Context(qref) => Some(*qref),
            EffectTarget::Token(_) => None,
        })
        .collect();
    let writes: Vec<QualifiedRef> = eff
        .writes
        .iter()
        .filter_map(|t| match t {
            EffectTarget::Context(qref) => Some(*qref),
            EffectTarget::Token(_) => None,
        })
        .collect();
    if reads.is_empty() && writes.is_empty() {
        return None;
    }
    Some((reads, writes))
}

/// Apply value substitutions to an instruction's operands.
fn apply_subst(kind: &mut InstKind, subst: &FxHashMap<ValueId, ValueId>) {
    let s = |v: &mut ValueId| {
        if let Some(&new) = subst.get(v) {
            *v = new;
        }
    };
    match kind {
        InstKind::Const { .. } | InstKind::Nop | InstKind::Poison { .. } | InstKind::Undef { .. } => {}
        InstKind::ContextProject { .. } => {}
        InstKind::ContextLoad { src, .. } => s(src),
        InstKind::ContextStore { dst, value } => {
            s(dst);
            s(value);
        }
        InstKind::VarLoad { .. } => {}
        InstKind::ParamLoad { .. } => {}
        InstKind::VarStore { src, .. } => s(src),
        InstKind::BinOp { left, right, .. } => {
            s(left);
            s(right);
        }
        InstKind::UnaryOp { operand, .. } => s(operand),
        InstKind::FieldGet { object, .. } => s(object),
        InstKind::LoadFunction { .. } => {}
        InstKind::FunctionCall {
            callee,
            args,
            context_uses,
            context_defs,
            ..
        } => {
            if let Callee::Indirect(v) = callee {
                s(v);
            }
            args.iter_mut().for_each(&s);
            context_uses.iter_mut().for_each(|(_, v)| s(v));
            context_defs.iter_mut().for_each(|(_, v)| s(v));
        }
        InstKind::Spawn {
            callee,
            args,
            context_uses,
            ..
        } => {
            if let Callee::Indirect(v) = callee {
                s(v);
            }
            args.iter_mut().for_each(&s);
            context_uses.iter_mut().for_each(|(_, v)| s(v));
        }
        InstKind::Eval {
            src, context_defs, ..
        } => {
            s(src);
            context_defs.iter_mut().for_each(|(_, v)| s(v));
        }
        InstKind::MakeDeque { elements, .. } => elements.iter_mut().for_each(&s),
        InstKind::MakeObject { fields, .. } => fields.iter_mut().for_each(|(_, v)| s(v)),
        InstKind::MakeRange { start, end, .. } => {
            s(start);
            s(end);
        }
        InstKind::MakeTuple { elements, .. } => elements.iter_mut().for_each(&s),
        InstKind::TupleIndex { tuple, .. } => s(tuple),
        InstKind::TestLiteral { src, .. } => s(src),
        InstKind::TestListLen { src, .. } => s(src),
        InstKind::TestObjectKey { src, .. } => s(src),
        InstKind::TestRange { src, .. } => s(src),
        InstKind::ListIndex { list, .. } => s(list),
        InstKind::ListGet { list, index, .. } => {
            s(list);
            s(index);
        }
        InstKind::ListSlice { list, .. } => s(list),
        InstKind::ObjectGet { object, .. } => s(object),
        InstKind::MakeClosure { captures, .. } => captures.iter_mut().for_each(&s),
        InstKind::ListStep {
            list,
            index_src,
            done_args,
            ..
        } => {
            s(list);
            s(index_src);
            done_args.iter_mut().for_each(&s);
        }
        InstKind::MakeVariant { payload, .. } => {
            if let Some(p) = payload {
                s(p);
            }
        }
        InstKind::TestVariant { src, .. } => s(src),
        InstKind::UnwrapVariant { src, .. } => s(src),
        InstKind::BlockLabel { params, .. } => params.iter_mut().for_each(&s),
        InstKind::Jump { args, .. } => args.iter_mut().for_each(&s),
        InstKind::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            s(cond);
            then_args.iter_mut().for_each(&s);
            else_args.iter_mut().for_each(&s);
        }
        InstKind::Return(v) => s(v),
        InstKind::Cast { src, .. } => s(src),
    }
}

// ── Step 1: SSA info collection ──────────────────────────────────────

/// A single SSA-relevant operation, recorded in instruction order.
#[derive(Debug, Clone)]
enum SsaOp {
    CtxStore {
        inst_idx: usize,
        ctx: QualifiedRef,
        value: ValueId,
    },
    VarStore {
        inst_idx: usize,
        name: Astr,
        value: ValueId,
    },
    VarLoad {
        dst: ValueId,
        name: Astr,
    },
    ParamLoad {
        dst: ValueId,
        name: Astr,
    },
}

/// Per-block operations in instruction order.
#[derive(Debug, Default)]
struct BlockOps {
    ops: Vec<SsaOp>,
}

/// Aggregated SSA info for both context and local variables.
struct SsaInfo {
    // ── Context fields ──
    written_contexts: BTreeSet<QualifiedRef>,
    /// ctx → root type (from ContextProject instructions).
    ctx_types: FxHashMap<QualifiedRef, Ty>,
    /// ctx → initial loaded value (from entry region ContextLoad).
    entry_ctx_defs: BTreeMap<QualifiedRef, ValueId>,

    // ── Local variable fields ──
    written_vars: BTreeSet<Astr>,
    /// var name → type (from VarStore src types).
    var_types: FxHashMap<Astr, Ty>,
    /// param name → initial loaded value (from entry region ParamLoad).
    entry_param_defs: BTreeMap<Astr, ValueId>,

    // ── Block ops (instruction-ordered) ──
    block_ops: FxHashMap<BlockIdx, BlockOps>,
}

fn collect_ssa_info(cfg: &Cfg, insts: &[Inst], val_types: &FxHashMap<ValueId, Ty>) -> SsaInfo {
    let mut val_to_ctx: FxHashMap<ValueId, QualifiedRef> = FxHashMap::default();
    let mut written_contexts = BTreeSet::default();
    let mut block_ops: FxHashMap<BlockIdx, BlockOps> = FxHashMap::default();
    let mut entry_ctx_defs: BTreeMap<QualifiedRef, ValueId> = BTreeMap::default();
    let mut ctx_types: FxHashMap<QualifiedRef, Ty> = FxHashMap::default();

    let mut written_vars: BTreeSet<Astr> = BTreeSet::default();
    let mut var_types: FxHashMap<Astr, Ty> = FxHashMap::default();
    let mut entry_param_defs: BTreeMap<Astr, ValueId> = BTreeMap::default();

    for (bi, block) in cfg.blocks.iter().enumerate() {
        let ops = block_ops.entry(BlockIdx(bi)).or_default();

        for &inst_i in &block.inst_indices {
            let inst = &insts[inst_i];
            match &inst.kind {
                InstKind::ContextProject { dst, ctx } => {
                    val_to_ctx.insert(*dst, *ctx);
                    if let Some(ty) = val_types.get(dst) {
                        ctx_types.entry(*ctx).or_insert_with(|| ty.clone());
                    }
                }
                InstKind::ContextLoad { dst, src } => {
                    if let Some(&ctx_id) = val_to_ctx.get(src) {
                        // Entry block loads = initial definitions.
                        if bi == 0 {
                            entry_ctx_defs.entry(ctx_id).or_insert(*dst);
                        }
                    }
                }
                InstKind::ContextStore { dst, value } => {
                    if let Some(&ctx_id) = val_to_ctx.get(dst) {
                        ops.ops.push(SsaOp::CtxStore {
                            inst_idx: inst_i,
                            ctx: ctx_id,
                            value: *value,
                        });
                        written_contexts.insert(ctx_id);
                    }
                }
                InstKind::FieldGet { dst, object, .. } => {
                    if let Some(&ctx_id) = val_to_ctx.get(object) {
                        val_to_ctx.insert(*dst, ctx_id);
                    }
                }
                InstKind::VarStore { name, src } => {
                    ops.ops.push(SsaOp::VarStore {
                        inst_idx: inst_i,
                        name: *name,
                        value: *src,
                    });
                    written_vars.insert(*name);
                    if let Some(ty) = val_types.get(src) {
                        var_types.entry(*name).or_insert_with(|| ty.clone());
                    }
                }
                InstKind::VarLoad { dst, name } => {
                    ops.ops.push(SsaOp::VarLoad {
                        dst: *dst,
                        name: *name,
                    });
                }
                InstKind::ParamLoad { dst, name } => {
                    if bi == 0 {
                        entry_param_defs.entry(*name).or_insert(*dst);
                    }
                    ops.ops.push(SsaOp::ParamLoad {
                        dst: *dst,
                        name: *name,
                    });
                }
                _ => {}
            }
        }
    }

    SsaInfo {
        written_contexts,
        ctx_types,
        entry_ctx_defs,
        written_vars,
        var_types,
        entry_param_defs,
        block_ops,
    }
}

// ── Step 2: SSABuilder execution ────────────────────────────────────

fn run_ssa_builder(
    cfg: &Cfg,
    preds: &FxHashMap<BlockIdx, smallvec::SmallVec<[BlockIdx; 2]>>,
    ssa_info: &SsaInfo,
    val_factory: &mut acvus_utils::LocalFactory<ValueId>,
    val_types: &mut FxHashMap<ValueId, Ty>,
) -> (Vec<crate::ssa::PhiInsertion>, FxHashMap<ValueId, ValueId>, Vec<ValueId>) {
    let mut ssa = SSABuilder::new();

    let block_label = |bi: BlockIdx| -> Label {
        if bi.0 == 0 {
            ENTRY_BLOCK
        } else {
            cfg.blocks[bi.0].label.unwrap_or(Label(bi.0 as u32))
        }
    };

    // Detect loop headers (backedge target: succ index <= current index).
    let mut loop_headers: BTreeSet<BlockIdx> = BTreeSet::default();
    for (bi, _) in cfg.blocks.iter().enumerate() {
        for succ in cfg.successors(BlockIdx(bi)) {
            if succ.0 <= bi {
                loop_headers.insert(succ);
            }
        }
    }

    // Register predecessors.
    for (block_idx, block_preds) in preds {
        let label = block_label(*block_idx);
        for pred in block_preds {
            ssa.add_predecessor(label, block_label(*pred));
        }
    }

    // Seal entry block.
    ssa.seal_block(ENTRY_BLOCK, &mut || val_factory.next());

    // Define initial values in entry block.
    let mut undef_defs: Vec<ValueId> = Vec::new();

    // Context entry defs (from ContextLoad in entry region).
    for (&ctx_id, &val) in &ssa_info.entry_ctx_defs {
        ssa.define(ENTRY_BLOCK, SsaVar::Context(ctx_id), val);
    }
    // Written contexts without entry defs need undef initial value.
    // These are write-only contexts (first operation is ContextStore, no prior ContextLoad).
    for &ctx_id in &ssa_info.written_contexts {
        if !ssa_info.entry_ctx_defs.contains_key(&ctx_id) {
            let undef_val = val_factory.next();
            if let Some(ty) = ssa_info.ctx_types.get(&ctx_id) {
                val_types.insert(undef_val, ty.clone());
            }
            ssa.define(ENTRY_BLOCK, SsaVar::Context(ctx_id), undef_val);
            undef_defs.push(undef_val);
        }
    }

    // Param entry defs (from ParamLoad in entry region).
    for (name, &val) in &ssa_info.entry_param_defs {
        ssa.define(ENTRY_BLOCK, SsaVar::Local(*name), val);
    }
    // Written vars without entry defs need undef initial value.
    // These are variables first defined inside a block (e.g., loop iterator bindings).
    for &name in &ssa_info.written_vars {
        if !ssa_info.entry_param_defs.contains_key(&name) {
            let undef_val = val_factory.next();
            if let Some(ty) = ssa_info.var_types.get(&name) {
                val_types.insert(undef_val, ty.clone());
            }
            ssa.define(ENTRY_BLOCK, SsaVar::Local(name), undef_val);
            undef_defs.push(undef_val);
        }
    }

    // Process blocks: define stores in instruction order.
    // VarLoad/ParamLoad substitutions are deferred until all blocks are sealed,
    // because use_var on an unsealed block returns a pending phi placeholder
    // that may be resolved to a different value after sealing.
    for (bi, _) in cfg.blocks.iter().enumerate() {
        let block_idx = BlockIdx(bi);
        let label = block_label(block_idx);

        if let Some(ops) = ssa_info.block_ops.get(&block_idx) {
            for op in &ops.ops {
                match op {
                    SsaOp::CtxStore { ctx, value, .. } => {
                        ssa.define(label, SsaVar::Context(*ctx), *value);
                    }
                    SsaOp::VarStore { name, value, .. } => {
                        ssa.define(label, SsaVar::Local(*name), *value);
                    }
                    // VarLoad/ParamLoad — deferred, see below.
                    SsaOp::VarLoad { .. } | SsaOp::ParamLoad { .. } => {}
                }
            }
        }

        if bi > 0 && !loop_headers.contains(&block_idx) {
            ssa.seal_block(label, &mut || val_factory.next());
        }
    }

    // Seal loop headers.
    for &header in &loop_headers {
        ssa.seal_block(block_label(header), &mut || val_factory.next());
    }

    // Trigger PHIs at merge points (sorted for deterministic ValueId allocation).
    let mut merge_blocks: Vec<_> = preds.iter().filter(|(_, p)| p.len() > 1).collect();
    merge_blocks.sort_by_key(|(idx, _)| *idx);
    for (block_idx, _) in merge_blocks {
        {
            let label = block_label(*block_idx);
            for &ctx_id in &ssa_info.written_contexts {
                let _ = ssa.use_var(label, SsaVar::Context(ctx_id), &mut || val_factory.next());
            }
            for &name in &ssa_info.written_vars {
                let _ = ssa.use_var(label, SsaVar::Local(name), &mut || val_factory.next());
            }
        }
    }

    // Build var_subst: resolve VarLoad/ParamLoad to SSA values.
    // Two mechanisms:
    //   1. Intra-block: VarStore in same block before VarLoad → direct forwarding.
    //   2. Inter-block: use_var on sealed SSA builder → predecessor lookup.
    let mut var_subst: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    for (bi, _) in cfg.blocks.iter().enumerate() {
        let block_idx = BlockIdx(bi);
        let label = block_label(block_idx);
        // Track last VarStore value per name within this block.
        let mut intra_block: FxHashMap<Astr, ValueId> = FxHashMap::default();

        if let Some(ops) = ssa_info.block_ops.get(&block_idx) {
            for op in &ops.ops {
                match op {
                    SsaOp::VarStore { name, value, .. } => {
                        intra_block.insert(*name, *value);
                    }
                    SsaOp::VarLoad { dst, name } | SsaOp::ParamLoad { dst, name } => {
                        let ssa_val = if let Some(&local_val) = intra_block.get(name) {
                            // Same block — forward from preceding VarStore.
                            local_val
                        } else {
                            // Cross-block — ask SSA builder (all blocks sealed).
                            ssa.use_var(label, SsaVar::Local(*name), &mut || val_factory.next())
                        };
                        if ssa_val != *dst {
                            var_subst.insert(*dst, ssa_val);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    (ssa.finish(), var_subst, undef_defs)
}

// ── Step 3: Patch instructions ──────────────────────────────────────

fn patch_instructions(
    body: &mut MirBody,
    cfg: &Cfg,
    phi_insertions: &[crate::ssa::PhiInsertion],
    ssa_info: &SsaInfo,
) {
    // PHI lookup tables.
    let mut block_phis: BTreeMap<Label, Vec<&crate::ssa::PhiInsertion>> = BTreeMap::default();
    for phi in phi_insertions {
        block_phis.entry(phi.block).or_default().push(phi);
    }
    // Sort PHIs by SsaVar for deterministic block param ordering.
    for phis in block_phis.values_mut() {
        phis.sort_by_key(|p| p.var);
    }

    // Build jump args in the same SsaVar-sorted order as block_phis.
    let mut jump_extra_args: FxHashMap<(Label, Label), Vec<ValueId>> = FxHashMap::default();
    for (&label, phis) in &block_phis {
        for phi in phis {
            for &(pred, val) in &phi.incoming {
                jump_extra_args.entry((pred, label)).or_default().push(val);
            }
        }
    }

    // Identify ContextStores to remove (in branches superseded by PHI write-back).
    let phi_contexts: FxHashSet<QualifiedRef> = phi_insertions
        .iter()
        .filter_map(|p| match p.var {
            SsaVar::Context(ctx) => Some(ctx),
            SsaVar::Local(_) => None,
        })
        .collect();
    let merge_labels: FxHashSet<Label> = block_phis.keys().copied().collect();
    let mut remove_indices: FxHashSet<usize> = FxHashSet::default();

    for (bi, block) in cfg.blocks.iter().enumerate() {
        let jumps_to_merge = match &block.terminator {
            Terminator::Jump { target, .. } => merge_labels.contains(target),
            Terminator::JumpIf {
                then_label,
                else_label,
                ..
            } => merge_labels.contains(then_label) || merge_labels.contains(else_label),
            Terminator::ListStep { done, .. } => merge_labels.contains(done),
            _ => false,
        };
        if jumps_to_merge {
            if let Some(ops) = ssa_info.block_ops.get(&BlockIdx(bi)) {
                for op in &ops.ops {
                    if let SsaOp::CtxStore { inst_idx, ctx, .. } = op {
                        if phi_contexts.contains(ctx) {
                            remove_indices.insert(*inst_idx);
                            // Remove preceding ContextProject if it exists.
                            if *inst_idx > 0
                                && matches!(
                                    body.insts[*inst_idx - 1].kind,
                                    InstKind::ContextProject { .. }
                                )
                            {
                                remove_indices.insert(*inst_idx - 1);
                            }
                        }
                    }
                }
            }
        }
    }

    // Rebuild instruction list.
    let old_insts = std::mem::take(&mut body.insts);
    let mut new_insts = Vec::with_capacity(old_insts.len());
    let mut current_label = ENTRY_BLOCK;

    for (i, mut inst) in old_insts.into_iter().enumerate() {
        if remove_indices.contains(&i) {
            continue;
        }

        match &mut inst.kind {
            InstKind::BlockLabel { label, params, .. } => {
                current_label = *label;
                let phis_here = block_phis.get(label);
                if let Some(phis) = phis_here {
                    for phi in phis {
                        params.push(phi.result);
                    }
                }
                let span = inst.span;
                new_insts.push(inst);

                // Write-back ContextStores for context PHI values.
                // Local variable PHIs do NOT need write-back.
                if let Some(phis) = phis_here {
                    for phi in phis {
                        let ctx = match phi.var {
                            SsaVar::Context(ctx) => ctx,
                            SsaVar::Local(name) => {
                                // For local PHIs, just set the type on the result value.
                                if let Some(ty) = ssa_info.var_types.get(&name) {
                                    body.val_types.insert(phi.result, ty.clone());
                                }
                                continue;
                            }
                        };
                        let ty = ssa_info
                            .ctx_types
                            .get(&ctx)
                            .expect("missing ty for context in PHI write-back")
                            .clone();
                        // Set type for PHI result value.
                        body.val_types.insert(phi.result, ty.clone());
                        let proj = body.val_factory.next();
                        body.val_types.insert(proj, ty.clone());
                        new_insts.push(Inst {
                            span,
                            kind: InstKind::ContextProject { dst: proj, ctx },
                        });
                        new_insts.push(Inst {
                            span,
                            kind: InstKind::ContextStore {
                                dst: proj,
                                value: phi.result,
                            },
                        });
                    }
                }
                continue;
            }
            InstKind::Jump { label, args } => {
                if let Some(extra) = jump_extra_args.get(&(current_label, *label)) {
                    args.extend_from_slice(extra);
                }
            }
            InstKind::JumpIf {
                then_label,
                then_args,
                else_label,
                else_args,
                ..
            } => {
                if let Some(extra) = jump_extra_args.get(&(current_label, *then_label)) {
                    then_args.extend_from_slice(extra);
                }
                if let Some(extra) = jump_extra_args.get(&(current_label, *else_label)) {
                    else_args.extend_from_slice(extra);
                }
            }
            _ => {}
        }
        new_insts.push(inst);
    }

    body.insts = new_insts;
}

// ── Step 4: Apply var substitutions + remove VarLoad/VarStore/ParamLoad ──

fn apply_var_subst(body: &mut MirBody, var_subst: &FxHashMap<ValueId, ValueId>) {
    // Apply substitutions to all instruction operands.
    for inst in &mut body.insts {
        apply_subst(&mut inst.kind, var_subst);
    }

    // Remove VarLoad/VarStore/ParamLoad instructions (now dead).
    body.insts.retain(|inst| {
        !matches!(
            inst.kind,
            InstKind::VarLoad { .. } | InstKind::VarStore { .. } | InstKind::ParamLoad { .. }
        )
    });
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::{compile_script, compile_template};
    use crate::ty::{EffectSet, EffectTarget, Ty};
    use acvus_utils::Interner;
    use std::collections::BTreeSet;

    /// Empty fn_types — no ExternFn effect info.
    fn no_fn_types() -> FxHashMap<QualifiedRef, Ty> {
        FxHashMap::default()
    }

    fn count_phi_blocks(body: &MirBody) -> usize {
        body.insts
            .iter()
            .filter(
                |i| matches!(&i.kind, InstKind::BlockLabel { params, .. } if !params.is_empty()),
            )
            .count()
    }

    fn count_context_stores(body: &MirBody) -> usize {
        body.insts
            .iter()
            .filter(|i| matches!(&i.kind, InstKind::ContextStore { .. }))
            .count()
    }

    // ── Completeness: PHI inserted when needed ──

    #[test]
    fn match_one_arm_write_phi() {
        let i = Interner::new();
        let (mut module, _) = compile_template(
            &i,
            r#"{{ true = @name == "test" }}{{ @x = 42 }}{{ _ }}noop{{/}}"#,
            &[("x", Ty::Int), ("name", Ty::String)],
        )
        .unwrap();
        run(&mut module.main, &no_fn_types());
        assert!(
            count_phi_blocks(&module.main) >= 1,
            "merge should have PHI for @x"
        );
        assert!(
            count_context_stores(&module.main) >= 1,
            "should have write-back ContextStore"
        );
    }

    #[test]
    fn match_both_arms_write_phi() {
        let i = Interner::new();
        let (mut module, _) = compile_template(
            &i,
            r#"{{ true = @name == "test" }}{{ @x = 1 }}{{ _ }}{{ @x = 2 }}{{/}}"#,
            &[("x", Ty::Int), ("name", Ty::String)],
        )
        .unwrap();
        run(&mut module.main, &no_fn_types());
        assert!(count_phi_blocks(&module.main) >= 1);
    }

    #[test]
    fn iter_context_write_phi() {
        let i = Interner::new();
        let (mut module, _) = compile_template(
            &i,
            r#"{{ x in @items }}{{ @sum = @sum + x }}{{/}}"#,
            &[("items", Ty::List(Box::new(Ty::Int))), ("sum", Ty::Int)],
        )
        .unwrap();
        run(&mut module.main, &no_fn_types());
        assert!(
            count_phi_blocks(&module.main) >= 1,
            "loop header should have PHI for @sum"
        );
    }

    // ── Soundness: PHI NOT inserted when not needed ──

    #[test]
    fn match_no_write_no_phi() {
        let i = Interner::new();
        let (mut module, _) = compile_template(
            &i,
            r#"{{ true = @name == "test" }}yes{{ _ }}no{{/}}"#,
            &[("name", Ty::String)],
        )
        .unwrap();
        let stores_before = count_context_stores(&module.main);
        run(&mut module.main, &no_fn_types());
        assert_eq!(count_context_stores(&module.main), stores_before);
    }

    #[test]
    fn straight_line_write_preserved() {
        let i = Interner::new();
        let (mut module, _) = compile_script(&i, "@x = 42; @x", &[("x", Ty::Int)]).unwrap();
        let stores_before = count_context_stores(&module.main);
        run(&mut module.main, &no_fn_types());
        assert_eq!(count_context_stores(&module.main), stores_before);
    }

    // ── Script regression: entry loads + nested loops ──

    #[test]
    fn script_entry_loads_all_contexts() {
        // All contexts must have ContextProject + ContextLoad in entry (before first BlockLabel).
        let i = Interner::new();
        let (module, _) = compile_script(
            &i,
            "x in @items { @sum = @sum + x; }; @sum",
            &[("items", Ty::List(Box::new(Ty::Int))), ("sum", Ty::Int)],
        )
        .unwrap();

        let first_label = module
            .main
            .insts
            .iter()
            .position(|i| matches!(i.kind, InstKind::BlockLabel { .. }))
            .unwrap_or(module.main.insts.len());
        let entry_projects: Vec<_> = module.main.insts[..first_label]
            .iter()
            .filter(|i| matches!(i.kind, InstKind::ContextProject { .. }))
            .collect();
        let entry_loads: Vec<_> = module.main.insts[..first_label]
            .iter()
            .filter(|i| matches!(i.kind, InstKind::ContextLoad { .. }))
            .collect();
        // Both @items and @sum should have entry loads.
        assert!(
            entry_projects.len() >= 2,
            "expected entry ContextProject for all contexts"
        );
        assert!(
            entry_loads.len() >= 2,
            "expected entry ContextLoad for all contexts"
        );
    }

    #[test]
    fn script_nested_loop_phi() {
        // Nested loop must not panic in SSA pass (regression: ListStep CFG terminator).
        let i = Interner::new();
        let (module, _) = compile_script(
            &i,
            "row in @matrix { x in row { @sum = @sum + x; }; }; @sum",
            &[
                ("matrix", Ty::List(Box::new(Ty::List(Box::new(Ty::Int))))),
                ("sum", Ty::Int),
            ],
        )
        .unwrap();
        // Must have PHI for @sum (written in inner loop).
        assert!(
            count_phi_blocks(&module.main) >= 1,
            "nested loop should produce PHI for @sum"
        );
    }

    #[test]
    fn script_loop_with_branch_phi() {
        // Loop body with conditional context write — needs PHI.
        let i = Interner::new();
        let (module, _) = compile_script(
            &i,
            "x in @items { 0 = x { @count = @count + 1; }; }; @count",
            &[("items", Ty::List(Box::new(Ty::Int))), ("count", Ty::Int)],
        )
        .unwrap();
        assert!(
            count_phi_blocks(&module.main) >= 1,
            "loop + branch should produce PHI"
        );
    }

    // ── FunctionCall context_uses/context_defs population ──

    /// Build a minimal MirBody with a FunctionCall to a callee that reads and writes @ctx.
    /// Before SSA pass: context_uses/context_defs are empty.
    /// After SSA pass: they should be populated.
    #[test]
    fn function_call_populates_context_uses_defs() {
        let interner = Interner::new();
        let ctx_name = interner.intern("ctx");
        let qref = QualifiedRef::root(ctx_name);
        let callee_id = QualifiedRef::root(interner.intern("callee"));

        // Build fn_types: callee reads + writes @ctx.
        let mut fn_types = FxHashMap::default();
        fn_types.insert(
            callee_id,
            Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::Int),
                captures: vec![],
                effect: Effect::Resolved(EffectSet {
                    reads: BTreeSet::from([EffectTarget::Context(qref)]),
                    writes: BTreeSet::from([EffectTarget::Context(qref)]),
                    io: false,
                    self_modifying: false,
                }),
            },
        );

        // Build MIR manually:
        // v0 = ContextProject @ctx
        // v1 = ContextLoad v0        (entry load)
        // v2 = FunctionCall callee() uses[] defs[]   ← SSA pass should fill
        // Return v2
        let mut body = MirBody::new();
        let v0 = body.val_factory.next();
        let v1 = body.val_factory.next();
        let v2 = body.val_factory.next();
        body.val_types.insert(v0, Ty::Int);
        body.val_types.insert(v1, Ty::Int);
        body.val_types.insert(v2, Ty::Int);

        let span = acvus_ast::Span::ZERO;
        body.insts.push(Inst {
            span,
            kind: InstKind::ContextProject { dst: v0, ctx: qref },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::ContextLoad { dst: v1, src: v0 },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::FunctionCall {
                dst: v2,
                callee: Callee::Direct(callee_id),
                args: vec![],
                context_uses: vec![],
                context_defs: vec![],
            },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::Return(v2),
        });

        run(&mut body, &fn_types);

        // Find the FunctionCall and verify context_uses/context_defs are populated.
        let call_inst = body
            .insts
            .iter()
            .find(|i| matches!(i.kind, InstKind::FunctionCall { .. }));
        assert!(call_inst.is_some(), "FunctionCall should exist");

        if let InstKind::FunctionCall {
            context_uses,
            context_defs,
            ..
        } = &call_inst.unwrap().kind
        {
            assert_eq!(context_uses.len(), 1, "should have 1 context use (@ctx)");
            assert_eq!(context_uses[0].0, qref, "use should be @ctx");
            assert_eq!(context_uses[0].1, v1, "use should bind to entry load v1");

            assert_eq!(context_defs.len(), 1, "should have 1 context def (@ctx)");
            assert_eq!(context_defs[0].0, qref, "def should be @ctx");
            // def ValueId should be a fresh value (not v0, v1, or v2).
            let def_val = context_defs[0].1;
            assert!(
                def_val != v0 && def_val != v1 && def_val != v2,
                "def should be a fresh SSA value, got {def_val:?}"
            );
        } else {
            panic!("expected FunctionCall");
        }
    }

    /// FunctionCall with pure callee (no reads/writes) should NOT get context bindings.
    #[test]
    fn function_call_pure_no_context() {
        let interner = Interner::new();
        let callee_id = QualifiedRef::root(interner.intern("callee"));

        // Pure function — no reads, no writes.
        let mut fn_types = FxHashMap::default();
        fn_types.insert(
            callee_id,
            Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::Int),
                captures: vec![],
                effect: Effect::pure(),
            },
        );

        let mut body = MirBody::new();
        let v0 = body.val_factory.next();
        body.val_types.insert(v0, Ty::Int);

        let span = acvus_ast::Span::ZERO;
        body.insts.push(Inst {
            span,
            kind: InstKind::FunctionCall {
                dst: v0,
                callee: Callee::Direct(callee_id),
                args: vec![],
                context_uses: vec![],
                context_defs: vec![],
            },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::Return(v0),
        });

        run(&mut body, &fn_types);

        let call_inst = body
            .insts
            .iter()
            .find(|i| matches!(i.kind, InstKind::FunctionCall { .. }))
            .unwrap();
        if let InstKind::FunctionCall {
            context_uses,
            context_defs,
            ..
        } = &call_inst.kind
        {
            assert!(
                context_uses.is_empty(),
                "pure function should have no context_uses"
            );
            assert!(
                context_defs.is_empty(),
                "pure function should have no context_defs"
            );
        }
    }

    /// After a FunctionCall that writes @ctx, subsequent ContextLoad should see the new value.
    #[test]
    fn function_call_def_forwards_to_subsequent_load() {
        let interner = Interner::new();
        let ctx_name = interner.intern("ctx");
        let qref = QualifiedRef::root(ctx_name);
        let callee_id = QualifiedRef::root(interner.intern("callee"));

        let mut fn_types = FxHashMap::default();
        fn_types.insert(
            callee_id,
            Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::Unit),
                captures: vec![],
                effect: Effect::Resolved(EffectSet {
                    reads: BTreeSet::new(),
                    writes: BTreeSet::from([EffectTarget::Context(qref)]),
                    io: false,
                    self_modifying: false,
                }),
            },
        );

        // v0 = ContextProject @ctx
        // v1 = ContextLoad v0
        // v2 = FunctionCall callee()    ← writes @ctx, def = v_new
        // v3 = ContextProject @ctx
        // v4 = ContextLoad v3           ← should be replaced by v_new
        // Return v4
        let mut body = MirBody::new();
        let v0 = body.val_factory.next();
        let v1 = body.val_factory.next();
        let v2 = body.val_factory.next();
        let v3 = body.val_factory.next();
        let v4 = body.val_factory.next();
        for v in [v0, v1, v2, v3, v4] {
            body.val_types.insert(v, Ty::Int);
        }

        let span = acvus_ast::Span::ZERO;
        body.insts.push(Inst {
            span,
            kind: InstKind::ContextProject { dst: v0, ctx: qref },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::ContextLoad { dst: v1, src: v0 },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::FunctionCall {
                dst: v2,
                callee: Callee::Direct(callee_id),
                args: vec![],
                context_uses: vec![],
                context_defs: vec![],
            },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::ContextProject { dst: v3, ctx: qref },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::ContextLoad { dst: v4, src: v3 },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::Return(v4),
        });

        run(&mut body, &fn_types);

        // The second ContextLoad (v4) should be eliminated — replaced by the def from the call.
        let remaining_loads: Vec<_> = body
            .insts
            .iter()
            .filter(|i| matches!(i.kind, InstKind::ContextLoad { .. }))
            .collect();
        // Only the entry load should remain; the second should be forwarded.
        assert_eq!(
            remaining_loads.len(),
            1,
            "second ContextLoad should be eliminated by forwarding from FunctionCall def"
        );
    }

    #[test]
    fn script_sequential_loops_no_panic() {
        // Two sequential loops writing same context must not panic.
        let i = Interner::new();
        let (module, _) = compile_script(
            &i,
            "x in @a { @sum = @sum + x; }; y in @b { @sum = @sum + y; }; @sum",
            &[
                ("a", Ty::List(Box::new(Ty::Int))),
                ("b", Ty::List(Box::new(Ty::Int))),
                ("sum", Ty::Int),
            ],
        )
        .unwrap();
        // Both loops write @sum — SSA pass must handle this.
        assert!(count_context_stores(&module.main) >= 1);
    }

    // ── Token/Context soundness tests ──────────────────────────────

    use crate::ty::TokenId;

    /// Token read must NOT appear in context_uses after SSA pass.
    #[test]
    fn token_read_not_in_context_uses() {
        let interner = Interner::new();
        let ctx_name = interner.intern("ctx");
        let qref = QualifiedRef::root(ctx_name);
        let token = TokenId::alloc();
        let callee_id = QualifiedRef::root(interner.intern("callee"));

        let mut fn_types = FxHashMap::default();
        fn_types.insert(
            callee_id,
            Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::Int),
                captures: vec![],
                effect: Effect::Resolved(EffectSet {
                    reads: BTreeSet::from([
                        EffectTarget::Context(qref),
                        EffectTarget::Token(token),
                    ]),
                    writes: BTreeSet::new(),
                    io: false,
                    self_modifying: false,
                }),
            },
        );

        let mut body = MirBody::new();
        let v0 = body.val_factory.next();
        let v1 = body.val_factory.next();
        let v2 = body.val_factory.next();
        body.val_types.insert(v0, Ty::Int);
        body.val_types.insert(v1, Ty::Int);
        body.val_types.insert(v2, Ty::Int);

        let span = acvus_ast::Span::ZERO;
        body.insts.push(Inst {
            span,
            kind: InstKind::ContextProject { dst: v0, ctx: qref },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::ContextLoad { dst: v1, src: v0 },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::FunctionCall {
                dst: v2,
                callee: Callee::Direct(callee_id),
                args: vec![],
                context_uses: vec![],
                context_defs: vec![],
            },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::Return(v2),
        });

        run(&mut body, &fn_types);

        // context_uses should have exactly 1 entry (the Context), NOT 2.
        for inst in &body.insts {
            if let InstKind::FunctionCall {
                callee: Callee::Direct(ref fid),
                ref context_uses,
                ..
            } = inst.kind
            {
                if fid == &callee_id {
                    assert_eq!(
                        context_uses.len(),
                        1,
                        "Token must NOT appear in context_uses; only Context should"
                    );
                }
            }
        }
    }

    /// Token write must NOT appear in context_defs after SSA pass.
    #[test]
    fn token_write_not_in_context_defs() {
        let interner = Interner::new();
        let ctx_name = interner.intern("ctx");
        let qref = QualifiedRef::root(ctx_name);
        let token = TokenId::alloc();
        let callee_id = QualifiedRef::root(interner.intern("callee"));

        let mut fn_types = FxHashMap::default();
        fn_types.insert(
            callee_id,
            Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::Unit),
                captures: vec![],
                effect: Effect::Resolved(EffectSet {
                    reads: BTreeSet::new(),
                    writes: BTreeSet::from([
                        EffectTarget::Context(qref),
                        EffectTarget::Token(token),
                    ]),
                    io: false,
                    self_modifying: false,
                }),
            },
        );

        let mut body = MirBody::new();
        let v0 = body.val_factory.next();
        let v1 = body.val_factory.next();
        let v2 = body.val_factory.next();
        body.val_types.insert(v0, Ty::Int);
        body.val_types.insert(v1, Ty::Int);
        body.val_types.insert(v2, Ty::Unit);

        let span = acvus_ast::Span::ZERO;
        body.insts.push(Inst {
            span,
            kind: InstKind::ContextProject { dst: v0, ctx: qref },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::ContextLoad { dst: v1, src: v0 },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::FunctionCall {
                dst: v2,
                callee: Callee::Direct(callee_id),
                args: vec![],
                context_uses: vec![],
                context_defs: vec![],
            },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::Return(v2),
        });

        run(&mut body, &fn_types);

        for inst in &body.insts {
            if let InstKind::FunctionCall {
                callee: Callee::Direct(ref fid),
                ref context_defs,
                ..
            } = inst.kind
            {
                if fid == &callee_id {
                    assert_eq!(
                        context_defs.len(),
                        1,
                        "Token must NOT appear in context_defs; only Context should"
                    );
                }
            }
        }
    }

    /// Mixed Context + Token: only Context appears in SSA uses/defs.
    #[test]
    fn mixed_context_and_token_only_context_in_ssa() {
        let interner = Interner::new();
        let ctx_name = interner.intern("ctx");
        let qref = QualifiedRef::root(ctx_name);
        let token = TokenId::alloc();
        let callee_id = QualifiedRef::root(interner.intern("callee"));

        let mut fn_types = FxHashMap::default();
        fn_types.insert(
            callee_id,
            Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::Int),
                captures: vec![],
                effect: Effect::Resolved(EffectSet {
                    reads: BTreeSet::from([
                        EffectTarget::Context(qref),
                        EffectTarget::Token(token),
                    ]),
                    writes: BTreeSet::from([
                        EffectTarget::Context(qref),
                        EffectTarget::Token(token),
                    ]),
                    io: false,
                    self_modifying: false,
                }),
            },
        );

        let mut body = MirBody::new();
        let v0 = body.val_factory.next();
        let v1 = body.val_factory.next();
        let v2 = body.val_factory.next();
        body.val_types.insert(v0, Ty::Int);
        body.val_types.insert(v1, Ty::Int);
        body.val_types.insert(v2, Ty::Int);

        let span = acvus_ast::Span::ZERO;
        body.insts.push(Inst {
            span,
            kind: InstKind::ContextProject { dst: v0, ctx: qref },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::ContextLoad { dst: v1, src: v0 },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::FunctionCall {
                dst: v2,
                callee: Callee::Direct(callee_id),
                args: vec![],
                context_uses: vec![],
                context_defs: vec![],
            },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::Return(v2),
        });

        run(&mut body, &fn_types);

        for inst in &body.insts {
            if let InstKind::FunctionCall {
                callee: Callee::Direct(ref fid),
                ref context_uses,
                ref context_defs,
                ..
            } = inst.kind
            {
                if fid == &callee_id {
                    assert_eq!(
                        context_uses.len(),
                        1,
                        "only Context(@ctx) should be in context_uses, not Token"
                    );
                    assert_eq!(
                        context_defs.len(),
                        1,
                        "only Context(@ctx) should be in context_defs, not Token"
                    );
                    // Verify the actual QualifiedRef is correct.
                    assert_eq!(context_uses[0].0, qref);
                    assert_eq!(context_defs[0].0, qref);
                }
            }
        }
    }

    /// EffectSet union preserves both Context and Token targets.
    #[test]
    fn effect_set_union_preserves_context_and_token() {
        let interner = Interner::new();
        let qref = QualifiedRef::root(interner.intern("ctx"));
        let token = TokenId::alloc();

        let a = EffectSet {
            reads: BTreeSet::from([EffectTarget::Context(qref)]),
            ..Default::default()
        };
        let b = EffectSet {
            reads: BTreeSet::from([EffectTarget::Token(token)]),
            ..Default::default()
        };
        let u = a.union(&b);
        assert_eq!(u.reads.len(), 2, "union should contain both Context and Token");
        assert!(u.reads.contains(&EffectTarget::Context(qref)));
        assert!(u.reads.contains(&EffectTarget::Token(token)));
    }

    /// extract_effect_refs filters Token targets out.
    #[test]
    fn extract_effect_refs_filters_tokens() {
        let interner = Interner::new();
        let qref = QualifiedRef::root(interner.intern("ctx"));
        let token = TokenId::alloc();
        let fid = QualifiedRef::root(interner.intern("callee"));

        let mut fn_types = FxHashMap::default();
        fn_types.insert(
            fid,
            Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::Int),
                captures: vec![],
                effect: Effect::Resolved(EffectSet {
                    reads: BTreeSet::from([
                        EffectTarget::Context(qref),
                        EffectTarget::Token(token),
                    ]),
                    writes: BTreeSet::from([EffectTarget::Token(token)]),
                    io: false,
                    self_modifying: false,
                }),
            },
        );

        let result = extract_effect_refs(&fn_types, &fid);
        let (reads, writes) = result.unwrap();
        assert_eq!(reads.len(), 1, "only Context should be in reads");
        assert_eq!(reads[0], qref);
        assert!(writes.is_empty(), "Token-only writes should yield empty");
    }
}
