//! SSA Pass (mem2reg for context and local variables)
//!
//! Promotes ContextProject/ContextLoad/ContextStore and VarStore/VarLoad/ParamLoad to SSA form.
//!
//! ## Pipeline
//!
//! 1. **Collect SSA info** (`collect_ssa_info`): scan all blocks for context ops
//!    (ContextProject/ContextLoad/ContextStore) and local variable ops
//!    (VarStore/VarLoad/ParamLoad). Records entry defs, written sets, and per-block ops.
//!
//! 2. **SSA builder** (`run_ssa_builder`): insert PHI nodes at merge points via
//!    the standard SSA construction algorithm. Produces `var_subst` (VarLoad/ParamLoad
//!    → SSA value) and `phi_insertions` (block params + jump args + write-back stores).
//!
//! 3. **Forward context values** (`forward_context_values`): dominator-tree-scoped
//!    store-load forwarding for context variables. Eliminates redundant ContextLoads
//!    and populates `context_uses`/`context_defs` on FunctionCall instructions.
//!    At merge points (>1 predecessor), written contexts are cleared from the
//!    forwarding state; unwritten (immutable) contexts remain forwarded.
//!
//! 4. **Apply var substitutions** (`apply_var_subst`): rewrite all VarLoad/ParamLoad
//!    uses with their SSA values, chained through the forwarding subst, then remove
//!    dead VarLoad/VarStore/ParamLoad instructions.
//!
//! ## Write-back model (context only)
//!
//! Branch-internal ContextStores are removed; a single write-back ContextStore is
//! inserted after each merge block. Local variables do NOT need write-back — they
//! exist only in SSA form.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};

use super::ssa::{ENTRY_BLOCK, SSABuilder, SsaVar};
use crate::analysis::domtree::DomTree;
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::graph::{FnMetadata, QualifiedRef};
use crate::ir::{Callee, Inst, InstKind, Label, ValueId};
use crate::ty::{Effect, Ty};

/// Run the SSA context pass on a CfgBody.
///
/// `fn_metadata` maps FunctionId → FnMetadata for resolving callee effects
/// (which contexts a function reads/writes). Used to populate
/// `context_uses`/`context_defs` on FunctionCall/Spawn instructions.
pub fn run(cfg: &mut CfgBody, fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>) {
    if cfg.blocks.is_empty() {
        return;
    }
    let ssa_info = collect_ssa_info(cfg);

    // Step 1: SSA construction — phi insertion + variable promotion.
    let has_work = !ssa_info.written_contexts.is_empty()
        || !ssa_info.written_vars.is_empty()
        || !ssa_info.read_vars.is_empty()
        || !ssa_info.entry_ctx_defs.is_empty()
        || !ssa_info.entry_param_defs.is_empty();
    let var_subst = if has_work {
        let preds = cfg.predecessors();
        let all_successors: Vec<SmallVec<[BlockIdx; 2]>> = (0..cfg.blocks.len())
            .map(|i| cfg.successors(BlockIdx(i)))
            .collect();
        let (phi_insertions, var_subst, undef_defs) = run_ssa_builder(
            &cfg.blocks,
            &all_successors,
            &preds,
            &ssa_info,
            &mut cfg.val_factory,
            &mut cfg.val_types,
        );
        if !phi_insertions.is_empty() {
            patch_instructions(cfg, &phi_insertions, &ssa_info);
        }
        if !undef_defs.is_empty() {
            let undef_insts: Vec<Inst> = undef_defs
                .into_iter()
                .map(|dst| Inst {
                    span: acvus_ast::Span::ZERO,
                    kind: InstKind::Undef { dst },
                })
                .collect();
            cfg.blocks[0].insts.splice(0..0, undef_insts);
        }
        var_subst
    } else {
        FxHashMap::default()
    };

    // Step 2: Context value forwarding.
    let fwd_subst = forward_context_values(cfg, fn_metadata, &ssa_info.written_contexts);

    // Step 3: Apply substitutions + remove promoted instructions.
    if !var_subst.is_empty() {
        let chained: FxHashMap<ValueId, ValueId> = var_subst
            .into_iter()
            .map(|(from, to)| (from, fwd_subst.get(&to).copied().unwrap_or(to)))
            .collect();
        apply_var_subst(cfg, &chained, &ssa_info);
    }

}

/// Store-load forwarding for context variables, scoped by the dominator tree.
///
/// Walks blocks in dominator-tree preorder (DFS), inheriting the context
/// forwarding state from the dominator parent. At merge points (>1 predecessor),
/// written contexts are cleared — the SSA builder inserted PHIs for those.
/// Unwritten (immutable) contexts are safe to forward across all blocks.
///
/// Two phases:
///   1. **Collect** — read `cfg.blocks`, allocate new ValueIds via `val_factory`,
///      produce substitution map + removal set + call patches.
///   2. **Apply** — mutate `cfg.blocks` with the collected results.
fn forward_context_values(
    cfg: &mut CfgBody,
    fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>,
    written_contexts: &BTreeSet<QualifiedRef>,
) -> FxHashMap<ValueId, ValueId> {
    let num_blocks = cfg.blocks.len();
    if num_blocks == 0 {
        return FxHashMap::default();
    }

    // Build dominator tree and predecessors for merge-point detection.
    let domtree = DomTree::build(cfg);
    let preds = cfg.predecessors();

    // Split val_factory out of cfg to allow simultaneous read of cfg.blocks
    // and mutable allocation of new ValueIds during the collect phase.
    let mut val_factory = std::mem::replace(&mut cfg.val_factory, acvus_utils::LocalFactory::new());

    // ── Collect phase: walk dominator tree, gather forwarding results ──

    // Global state: projection ValueId → QualifiedRef (accumulated across all blocks).
    let mut val_to_ctx: FxHashMap<ValueId, QualifiedRef> = FxHashMap::default();
    // Collected results.
    let mut subst: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    let mut remove: FxHashSet<(usize, usize)> = FxHashSet::default();
    let mut call_patches: Vec<(
        usize,
        usize,
        Vec<(QualifiedRef, ValueId)>,
        Vec<(QualifiedRef, ValueId)>,
    )> = Vec::new();

    // Precompute dominator tree children for each block.
    let mut dom_children: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); num_blocks];
    for child in 1..num_blocks {
        if let Some(parent) = domtree.idom(BlockIdx(child)) {
            dom_children[parent.0].push(child);
        }
    }

    // Recursive DFS over the dominator tree.
    //
    // Each block inherits its dominator parent's ctx_state. At merge points
    // (>1 predecessor), written contexts are cleared. After processing a
    // block's subtree, ctx_state is restored for sibling processing.
    type CtxState = FxHashMap<QualifiedRef, ValueId>;

    /// Collected results from the dominator-tree walk.
    struct CollectState<'a> {
        blocks: &'a [crate::cfg::Block],
        val_types: &'a mut FxHashMap<ValueId, Ty>,
        fn_metadata: &'a FxHashMap<QualifiedRef, FnMetadata>,
        preds: &'a FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>>,
        written_contexts: &'a BTreeSet<QualifiedRef>,
        dom_children: &'a [SmallVec<[usize; 4]>],
        val_to_ctx: &'a mut FxHashMap<ValueId, QualifiedRef>,
        val_factory: &'a mut acvus_utils::LocalFactory<ValueId>,
        subst: &'a mut FxHashMap<ValueId, ValueId>,
        remove: &'a mut FxHashSet<(usize, usize)>,
        call_patches: &'a mut Vec<(
            usize,
            usize,
            Vec<(QualifiedRef, ValueId)>,
            Vec<(QualifiedRef, ValueId)>,
        )>,
        /// Per-block exit ctx_state, recorded after processing each block.
        /// `None` = not yet visited (back edge target).
        block_exit_states: Vec<Option<CtxState>>,
    }

    fn walk_dom_tree(bi: usize, ctx_state: &mut CtxState, st: &mut CollectState<'_>) {
        let saved = ctx_state.clone();

        // At merge points, refine ctx_state using predecessor exit states.
        // For each written context: keep if ALL predecessors agree on the same value.
        // If any predecessor is unvisited (back edge) or disagrees → clear.
        let is_merge = st.preds.get(&BlockIdx(bi)).is_some_and(|p| p.len() > 1);
        if is_merge {
            if let Some(preds) = st.preds.get(&BlockIdx(bi)) {
                for ctx in st.written_contexts.iter() {
                    let mut agreed_val: Option<ValueId> = None;
                    let mut all_agree = true;
                    for pred in preds {
                        match &st.block_exit_states[pred.0] {
                            Some(pred_state) => match pred_state.get(ctx) {
                                Some(&val) => match agreed_val {
                                    None => agreed_val = Some(val),
                                    Some(prev) if prev == val => {}
                                    Some(_) => {
                                        all_agree = false;
                                        break;
                                    }
                                },
                                None => {
                                    // Predecessor doesn't know this context's value.
                                    all_agree = false;
                                    break;
                                }
                            },
                            None => {
                                // Unvisited predecessor (back edge) — conservative.
                                all_agree = false;
                                break;
                            }
                        }
                    }
                    if all_agree {
                        if let Some(val) = agreed_val {
                            ctx_state.insert(*ctx, val);
                        }
                    } else {
                        ctx_state.remove(ctx);
                    }
                }
            }
        }

        // Process instructions in this block.
        let block = &st.blocks[bi];
        for (ii, inst) in block.insts.iter().enumerate() {
            match &inst.kind {
                // Identity Ref to context → register in val_to_ctx for forwarding.
                InstKind::Ref {
                    dst,
                    target: crate::ir::RefTarget::Context(ctx),
                    path,
                } if path.is_empty() => {
                    st.val_to_ctx.insert(*dst, *ctx);
                }

                // Load from context Ref → context forwarding.
                InstKind::Load { dst, src, volatile } => {
                    // Volatile loads must not be forwarded.
                    if *volatile {
                        continue;
                    }
                    let src_resolved = st.subst.get(src).copied().unwrap_or(*src);
                    if let Some(&ctx_id) = st.val_to_ctx.get(&src_resolved) {
                        if let Some(&known_val) = ctx_state.get(&ctx_id) {
                            // Known value — substitute and mark for removal.
                            st.subst.insert(*dst, known_val);
                            st.remove.insert((bi, ii));
                            // Don't remove the Ref here — it might be used by other
                            // instructions (canonical Ref reuse). DCE will clean up
                            // orphaned Refs after all passes complete.
                        } else {
                            // First load of this context in this dom-tree path — record it.
                            ctx_state.insert(ctx_id, *dst);
                        }
                    }
                }

                // Store to context Ref → update forwarding state.
                InstKind::Store {
                    dst,
                    value,
                    volatile,
                } => {
                    // Volatile stores must not record into forwarding state.
                    if *volatile {
                        continue;
                    }
                    let dst_resolved = st.subst.get(dst).copied().unwrap_or(*dst);
                    let value_resolved = st.subst.get(value).copied().unwrap_or(*value);
                    if let Some(&ctx_id) = st.val_to_ctx.get(&dst_resolved) {
                        ctx_state.insert(ctx_id, value_resolved);
                    }
                }

                // FunctionCall with Direct callee: populate context_uses/context_defs.
                InstKind::FunctionCall {
                    callee: Callee::Direct(fn_id),
                    context_uses,
                    context_defs,
                    ..
                } if context_uses.is_empty() && context_defs.is_empty() => {
                    if let Some((reads, writes)) = extract_effect_refs(st.fn_metadata, fn_id) {
                        let mut uses = Vec::new();
                        for qref in &reads {
                            if let Some(&val) = ctx_state.get(qref) {
                                uses.push((*qref, val));
                            }
                        }
                        let mut defs = Vec::new();
                        for qref in &writes {
                            // Derive type: from current forwarded value, or from Ref inner type.
                            let ty = ctx_state
                                .get(qref)
                                .and_then(|v| st.val_types.get(v))
                                .cloned()
                                .or_else(|| {
                                    st.val_to_ctx.iter().find_map(|(vid, ctx)| {
                                        if ctx == qref {
                                            if let Some(Ty::Ref(inner, _)) = st.val_types.get(vid) {
                                                Some(*inner.clone())
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    })
                                })
                                .expect("context_def type must be derivable");
                            let new_val = alloc_val(st.val_factory, st.val_types, ty);
                            defs.push((*qref, new_val));
                            ctx_state.insert(*qref, new_val);
                        }
                        if !uses.is_empty() || !defs.is_empty() {
                            st.call_patches.push((bi, ii, uses, defs));
                        }
                    }
                }

                _ => {}
            }
        }

        // Record exit state for this block (before recursing into children).
        st.block_exit_states[bi] = Some(ctx_state.clone());

        // Recurse into dominated children, then restore state for siblings.
        for &child in &st.dom_children[bi] {
            walk_dom_tree(child, ctx_state, st);
        }

        *ctx_state = saved;
    }

    let mut ctx_state: CtxState = FxHashMap::default();
    let mut state = CollectState {
        blocks: &cfg.blocks,
        val_types: &mut cfg.val_types,
        fn_metadata,
        preds: &preds,
        written_contexts,
        dom_children: &dom_children,
        val_to_ctx: &mut val_to_ctx,
        val_factory: &mut val_factory,
        subst: &mut subst,
        remove: &mut remove,
        call_patches: &mut call_patches,
        block_exit_states: vec![None; num_blocks],
    };
    walk_dom_tree(0, &mut ctx_state, &mut state);

    // Restore val_factory back to cfg.
    cfg.val_factory = val_factory;

    if remove.is_empty() && subst.is_empty() && call_patches.is_empty() {
        return subst;
    }

    // ── Apply phase: mutate cfg.blocks with collected results ──

    let patch_map: FxHashMap<(usize, usize), _> = call_patches
        .into_iter()
        .map(|(bi, ii, u, d)| ((bi, ii), (u, d)))
        .collect();

    for (bi, block) in cfg.blocks.iter_mut().enumerate() {
        let old_insts = std::mem::take(&mut block.insts);
        block.insts = old_insts
            .into_iter()
            .enumerate()
            .filter(|(ii, _)| !remove.contains(&(bi, *ii)))
            .map(|(ii, mut inst)| {
                apply_subst(&mut inst.kind, &subst);
                if let Some((uses, defs)) = patch_map.get(&(bi, ii))
                    && let InstKind::FunctionCall {
                        context_uses,
                        context_defs,
                        ..
                    } = &mut inst.kind
                {
                    *context_uses = uses.clone();
                    *context_defs = defs.clone();
                }
                inst
            })
            .collect();

        apply_subst_terminator(&mut block.terminator, &subst);
    }

    subst
}

/// Extract reads/writes QualifiedRefs from a function's type effect.
///
/// Only `EffectTarget::Context` refs are returned — `Token` targets are NOT
/// SSA-compatible and must never be converted to context_uses/context_defs.
pub(crate) fn extract_effect_refs(
    fn_metadata: &FxHashMap<QualifiedRef, FnMetadata>,
    fn_id: &QualifiedRef,
) -> Option<(Vec<QualifiedRef>, Vec<QualifiedRef>)> {
    use crate::ty::EffectTarget;

    let m = fn_metadata.get(fn_id)?;
    let Ty::Fn {
        effect: Effect::Resolved(eff),
        ..
    } = &m.ty
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
        InstKind::Const { .. }
        | InstKind::Ref { .. }
        | InstKind::Nop
        | InstKind::Poison { .. }
        | InstKind::Undef { .. } => {}
        InstKind::Load { src, .. } => s(src),
        InstKind::Store { dst, value, .. } => {
            s(dst);
            s(value);
        }
        InstKind::BinOp { left, right, .. } => {
            s(left);
            s(right);
        }
        InstKind::UnaryOp { operand, .. } => s(operand),
        InstKind::FieldGet { object, .. } => s(object),
        InstKind::FieldSet { object, value, .. } => {
            s(object);
            s(value);
        }
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
        // BlockLabel, Jump, JumpIf, Return are terminators in CfgBody, not instructions.
        // But they may still exist as InstKind variants for demoted code paths.
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

/// Apply value substitutions to a block terminator's operands.
fn apply_subst_terminator(term: &mut Terminator, subst: &FxHashMap<ValueId, ValueId>) {
    let s = |v: &mut ValueId| {
        if let Some(&new) = subst.get(v) {
            *v = new;
        }
    };
    match term {
        Terminator::Jump { args, .. } => args.iter_mut().for_each(&s),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            s(cond);
            then_args.iter_mut().for_each(&s);
            else_args.iter_mut().for_each(&s);
        }
        Terminator::ListStep {
            dst,
            list,
            index_src,
            index_dst,
            done_args,
            ..
        } => {
            s(dst);
            s(list);
            s(index_src);
            s(index_dst);
            done_args.iter_mut().for_each(&s);
        }
        Terminator::Return(v) => s(v),
        Terminator::Fallthrough => {}
    }
}

// ── Step 1: SSA info collection ──────────────────────────────────────

/// A single SSA-relevant operation, recorded in instruction order.
#[derive(Debug, Clone)]
enum SsaOp {
    CtxStore {
        /// Position within the CfgBody, used by `patch_instructions` to locate
        /// and remove ContextStore instructions superseded by PHI write-back.
        block_idx: usize,
        local_idx: usize,
        ctx: QualifiedRef,
        value: ValueId,
    },
    VarStore {
        slot: ValueId,
        value: ValueId,
    },
    VarLoad {
        dst: ValueId,
        slot: ValueId,
    },
    ParamLoad {
        dst: ValueId,
        slot: ValueId,
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
    written_vars: BTreeSet<ValueId>,
    /// All vars that are read (VarLoad) — for ensuring entry defines.
    read_vars: BTreeSet<ValueId>,
    /// var slot → type (from VarStore/VarLoad).
    var_types: FxHashMap<ValueId, Ty>,
    /// param/capture slot → initial loaded value (from entry region ParamLoad).
    entry_param_defs: BTreeMap<ValueId, ValueId>,

    // ── Block ops (instruction-ordered) ──
    block_ops: FxHashMap<BlockIdx, BlockOps>,
}

fn collect_ssa_info(cfg: &CfgBody) -> SsaInfo {
    use crate::ir::RefTarget;

    // Maps Ref dst → its RefTarget (for Load/Store to look up).
    let mut ref_target: FxHashMap<ValueId, RefTarget> = FxHashMap::default();
    // Variables that have field Refs — non-promotable (SROA not run or incomplete).
    let mut non_promotable_vars: std::collections::BTreeSet<ValueId> =
        std::collections::BTreeSet::new();
    let mut non_promotable_contexts: std::collections::BTreeSet<QualifiedRef> =
        std::collections::BTreeSet::new();

    let mut written_contexts = BTreeSet::default();
    let mut block_ops: FxHashMap<BlockIdx, BlockOps> = FxHashMap::default();
    let mut entry_ctx_defs: BTreeMap<QualifiedRef, ValueId> = BTreeMap::default();
    let mut ctx_types: FxHashMap<QualifiedRef, Ty> = FxHashMap::default();

    let mut written_vars: BTreeSet<ValueId> = BTreeSet::default();
    let mut read_vars: BTreeSet<ValueId> = BTreeSet::default();
    let mut var_types: FxHashMap<ValueId, Ty> = FxHashMap::default();

    // LLVM-style: param_regs ARE the SSA definitions for params.
    // Build entry_param_defs from param slots (not from first ParamLoad).
    // For params/captures: the param_reg IS both the storage slot and the initial value.
    let mut entry_param_defs: BTreeMap<ValueId, ValueId> = BTreeMap::default();
    for (_name, reg) in cfg.params.iter() {
        entry_param_defs.insert(*reg, *reg);
        if let Some(ty) = cfg.val_types.get(reg) {
            var_types.insert(*reg, ty.clone());
        }
    }
    // Same for captures.
    for (_name, reg) in cfg.captures.iter() {
        entry_param_defs.insert(*reg, *reg);
        if let Some(ty) = cfg.val_types.get(reg) {
            var_types.insert(*reg, ty.clone());
        }
    }

    // First pass: collect Ref targets and identify non-promotable storage.
    for block in &cfg.blocks {
        for inst in &block.insts {
            if let InstKind::Ref { dst, target, path } = &inst.kind {
                ref_target.insert(*dst, target.clone());
                if !path.is_empty() {
                    // Field Ref present → this storage is non-promotable.
                    match target {
                        RefTarget::Var(slot) | RefTarget::Param(slot) => {
                            non_promotable_vars.insert(*slot);
                        }
                        RefTarget::Context(qref) => {
                            non_promotable_contexts.insert(*qref);
                        }
                    }
                }
            }
        }
    }


    // Second pass: collect SSA ops (only for promotable identity Refs).
    for (bi, block) in cfg.blocks.iter().enumerate() {
        let ops = block_ops.entry(BlockIdx(bi)).or_default();

        for (ii, inst) in block.insts.iter().enumerate() {
            match &inst.kind {
                // Identity Ref → register for ctx_types.
                InstKind::Ref {
                    dst,
                    target: RefTarget::Context(ctx),
                    path,
                } if path.is_empty() => {
                    if !non_promotable_contexts.contains(ctx)
                        && let Some(Ty::Ref(inner, _)) = cfg.val_types.get(dst)
                    {
                        ctx_types.entry(*ctx).or_insert_with(|| *inner.clone());
                    }
                }

                // Load from identity Ref → entry defs or SsaOp.
                InstKind::Load {
                    dst,
                    src,
                    volatile: false,
                } => {
                    match ref_target.get(src) {
                        Some(RefTarget::Context(ctx)) if !non_promotable_contexts.contains(ctx) => {
                            if bi == 0 {
                                entry_ctx_defs.entry(*ctx).or_insert(*dst);
                            }
                        }
                        Some(RefTarget::Var(slot)) if !non_promotable_vars.contains(slot) => {
                            ops.ops.push(SsaOp::VarLoad {
                                dst: *dst,
                                slot: *slot,
                            });
                            read_vars.insert(*slot);
                            if let Some(ty) = cfg.val_types.get(dst) {
                                var_types.entry(*slot).or_insert_with(|| ty.clone());
                            }
                        }
                        Some(RefTarget::Param(slot)) if !non_promotable_vars.contains(slot) => {
                            // entry_param_defs already set from param_regs (LLVM-style).
                            read_vars.insert(*slot);
                            if let Some(ty) = cfg.val_types.get(dst) {
                                var_types.entry(*slot).or_insert_with(|| ty.clone());
                            }
                            ops.ops.push(SsaOp::ParamLoad {
                                dst: *dst,
                                slot: *slot,
                            });
                        }
                        _ => {} // volatile, field Ref, or unknown → skip
                    }
                }

                // Store to identity Ref → SsaOp.
                InstKind::Store {
                    dst,
                    value,
                    volatile: false,
                } => {
                    match ref_target.get(dst) {
                        Some(RefTarget::Context(ctx)) if !non_promotable_contexts.contains(ctx) => {
                            ops.ops.push(SsaOp::CtxStore {
                                block_idx: bi,
                                local_idx: ii,
                                ctx: *ctx,
                                value: *value,
                            });
                            written_contexts.insert(*ctx);
                        }
                        Some(RefTarget::Var(slot)) if !non_promotable_vars.contains(slot) => {
                            ops.ops.push(SsaOp::VarStore {
                                slot: *slot,
                                value: *value,
                            });
                            written_vars.insert(*slot);
                            if let Some(ty) = cfg.val_types.get(value) {
                                var_types.entry(*slot).or_insert_with(|| ty.clone());
                            }
                        }
                        _ => {} // volatile, field Ref, param store (shouldn't happen), or unknown
                    }
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
        read_vars,
        var_types,
        entry_param_defs,
        block_ops,
    }
}

// ── Step 2: SSABuilder execution ────────────────────────────────────

/// Allocate a typed ValueId. This is the ONLY way to create new ValueIds
/// in the SSA pass. val_factory.next() must never be called directly.
fn alloc_val(
    val_factory: &mut acvus_utils::LocalFactory<ValueId>,
    val_types: &mut FxHashMap<ValueId, Ty>,
    ty: Ty,
) -> ValueId {
    let val = val_factory.next();
    val_types.insert(val, ty);
    val
}

/// Convenience: allocate a typed ValueId for an SSA variable, looking up the type from ssa_info.
fn alloc_var_val(
    val_factory: &mut acvus_utils::LocalFactory<ValueId>,
    val_types: &mut FxHashMap<ValueId, Ty>,
    var: SsaVar,
    ssa_info: &SsaInfo,
) -> ValueId {
    let ty = match var {
        SsaVar::Context(ctx) => ssa_info.ctx_types.get(&ctx).cloned(),
        SsaVar::Local(slot) => ssa_info.var_types.get(&slot).cloned(),
    };
    // Every SSA variable MUST have a known type. If not, it's a collect_ssa_info bug.
    let ty = ty.unwrap_or_else(|| panic!("SSA variable {:?} has no type in ssa_info", var));
    alloc_val(val_factory, val_types, ty)
}

fn run_ssa_builder(
    blocks: &[crate::cfg::Block],
    all_successors: &[SmallVec<[BlockIdx; 2]>],
    preds: &FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>>,
    ssa_info: &SsaInfo,
    val_factory: &mut acvus_utils::LocalFactory<ValueId>,
    val_types: &mut FxHashMap<ValueId, Ty>,
) -> (
    Vec<super::ssa::PhiInsertion>,
    FxHashMap<ValueId, ValueId>,
    Vec<ValueId>,
) {
    let mut ssa = SSABuilder::new();

    let block_label = |bi: BlockIdx| -> Label {
        blocks[bi.0].label
    };

    // ── Detect loop headers (backedge target: succ index <= current index) ──
    let mut loop_headers: BTreeSet<BlockIdx> = BTreeSet::default();
    for (bi, _) in blocks.iter().enumerate() {
        for &succ in &all_successors[bi] {
            if succ.0 <= bi {
                loop_headers.insert(succ);
            }
        }
    }

    // ── Register predecessors ──
    for (block_idx, block_preds) in preds {
        let label = block_label(*block_idx);
        for pred in block_preds {
            ssa.add_predecessor(label, block_label(*pred));
        }
    }

    // ── Define initial values in entry block ──
    let mut undef_defs: Vec<ValueId> = Vec::new();

    // Context entry defs (from ContextLoad in entry region).
    for (&ctx_id, &val) in &ssa_info.entry_ctx_defs {
        ssa.define(ENTRY_BLOCK, SsaVar::Context(ctx_id), val);
    }
    // Written contexts without entry defs need undef initial value.
    for &ctx_id in &ssa_info.written_contexts {
        if !ssa_info.entry_ctx_defs.contains_key(&ctx_id) {
            let undef_val = alloc_var_val(val_factory, val_types, SsaVar::Context(ctx_id), ssa_info);
            ssa.define(ENTRY_BLOCK, SsaVar::Context(ctx_id), undef_val);
            undef_defs.push(undef_val);
        }
    }

    // Param entry defs (from ParamLoad in entry region).
    for (&slot, &val) in &ssa_info.entry_param_defs {
        ssa.define(ENTRY_BLOCK, SsaVar::Local(slot), val);
    }
    // ALL variables that appear in any SsaOp (read or write) need entry defines.
    // This is the LLVM alloca pattern: every variable has a definition at entry.
    // Collect all variable slots from written_vars + read_vars.
    let all_local_vars: BTreeSet<ValueId> = ssa_info
        .written_vars
        .iter()
        .chain(ssa_info.read_vars.iter())
        .copied()
        .collect();
    for &slot in &all_local_vars {
        if !ssa_info.entry_param_defs.contains_key(&slot) {
            let undef_val = alloc_var_val(val_factory, val_types, SsaVar::Local(slot), ssa_info);
            ssa.define(ENTRY_BLOCK, SsaVar::Local(slot), undef_val);
            undef_defs.push(undef_val);
        }
    }

    // Typed alloc closure for SSA builder — every ValueId gets a type at birth.
    let mut typed_alloc = |var: SsaVar| -> ValueId {
        alloc_var_val(val_factory, val_types, var, ssa_info)
    };

    // Seal ENTRY_BLOCK (virtual predecessor of block 0) unless block 0 is a loop header.
    if !loop_headers.contains(&BlockIdx(0)) {
        ssa.seal_block(ENTRY_BLOCK, &mut typed_alloc);
    }

    // ── Single-pass: define and use in program order (Braun algorithm) ──
    //
    // For each block, process ops in instruction order:
    //   - VarStore/CtxStore → ssa.define() (updates current_defs)
    //   - VarLoad/ParamLoad → ssa.use_var() (reads current_defs or predecessor)
    //
    // This ensures a VarLoad BEFORE a VarStore for the same slot gets the
    // predecessor's value (not the Store's value), because define() hasn't
    // been called yet at that point. A VarLoad AFTER a VarStore correctly
    // gets the stored value via current_defs (intra-block forwarding).
    //
    // Non-loop-header blocks are sealed BEFORE processing their ops,
    // since all predecessors (earlier in block order) are already processed.
    // Loop headers are sealed after all blocks (backedge sources come later).
    let mut var_subst: FxHashMap<ValueId, ValueId> = FxHashMap::default();

    for (bi, _) in blocks.iter().enumerate() {
        let block_idx = BlockIdx(bi);
        let label = block_label(block_idx);

        // Seal non-loop-header blocks before processing (all predecessors done).
        if bi > 0 && !loop_headers.contains(&block_idx) {
            ssa.seal_block(label, &mut typed_alloc);
        }

        if let Some(ops) = ssa_info.block_ops.get(&block_idx) {
            for op in &ops.ops {
                match op {
                    SsaOp::CtxStore { ctx, value, .. } => {
                        ssa.define(label, SsaVar::Context(*ctx), *value);
                    }
                    SsaOp::VarStore { slot, value, .. } => {
                        ssa.define(label, SsaVar::Local(*slot), *value);
                    }
                    SsaOp::VarLoad { dst, slot } | SsaOp::ParamLoad { dst, slot } => {
                        let ssa_val =
                            ssa.use_var(label, SsaVar::Local(*slot), &mut typed_alloc);
                        if ssa_val != *dst {
                            var_subst.insert(*dst, ssa_val);
                        }
                    }
                }
            }
        }
    }

    // ── Seal loop headers (deferred: backedge predecessors processed above) ──
    for &header in &loop_headers {
        ssa.seal_block(block_label(header), &mut typed_alloc);
    }

    // ── Trigger PHIs at merge points ──
    //
    // Explicitly request the merged value for every written variable at each
    // merge block. This ensures PHI nodes exist even when the variable is not
    // loaded at the merge point (needed for write-back and loop backedge args).
    let mut merge_blocks: Vec<_> = preds.iter().filter(|(_, p)| p.len() > 1).collect();
    merge_blocks.sort_by_key(|(idx, _)| *idx);
    for (block_idx, _) in merge_blocks {
        let label = block_label(*block_idx);
        for &ctx_id in &ssa_info.written_contexts {
            let _ = ssa.use_var(label, SsaVar::Context(ctx_id), &mut typed_alloc);
        }
        for &name in &ssa_info.written_vars {
            let _ = ssa.use_var(label, SsaVar::Local(name), &mut typed_alloc);
        }
    }

    let (phi_insertions, trivial_subst) = ssa.finish();

    // Resolve trivial-phi references in var_subst.
    // A VarLoad may have been mapped to a pending phi ValueId that was later
    // eliminated as trivial. Walk the trivial_subst chain to reach the real value.
    if !trivial_subst.is_empty() {
        for val in var_subst.values_mut() {
            let mut resolved = *val;
            while let Some(&next) = trivial_subst.get(&resolved) {
                resolved = next;
            }
            *val = resolved;
        }
    }

    (phi_insertions, var_subst, undef_defs)
}

// ── Step 3: Patch instructions ──────────────────────────────────────

fn patch_instructions(
    cfg: &mut CfgBody,
    phi_insertions: &[super::ssa::PhiInsertion],
    ssa_info: &SsaInfo,
) {
    // PHI lookup tables.
    let mut block_phis: BTreeMap<Label, Vec<&super::ssa::PhiInsertion>> = BTreeMap::default();
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
    // Set of (block_idx, inst_idx_within_block) to remove.
    let mut remove_positions: FxHashSet<(usize, usize)> = FxHashSet::default();

    for (bi, block) in cfg.blocks.iter().enumerate() {
        let jumps_to_merge = match &block.terminator {
            Terminator::Jump { label, .. } => merge_labels.contains(label),
            Terminator::JumpIf {
                then_label,
                else_label,
                ..
            } => merge_labels.contains(then_label) || merge_labels.contains(else_label),
            Terminator::ListStep { done, .. } => merge_labels.contains(done),
            _ => false,
        };
        if jumps_to_merge && let Some(ops) = ssa_info.block_ops.get(&BlockIdx(bi)) {
            for op in &ops.ops {
                if let SsaOp::CtxStore {
                    block_idx,
                    local_idx,
                    ctx,
                    ..
                } = op
                    && phi_contexts.contains(ctx)
                {
                    remove_positions.insert((*block_idx, *local_idx));
                    // Remove preceding Ref if it exists.
                    if *local_idx > 0
                        && matches!(
                            cfg.blocks[*block_idx].insts[*local_idx - 1].kind,
                            InstKind::Ref { .. }
                        )
                    {
                        remove_positions.insert((*block_idx, *local_idx - 1));
                    }
                }
            }
        }
    }

    // Remove marked instructions per block.
    for (bi, block) in cfg.blocks.iter_mut().enumerate() {
        let has_removals = remove_positions.iter().any(|(b, _)| *b == bi);
        if has_removals {
            let old_insts = std::mem::take(&mut block.insts);
            block.insts = old_insts
                .into_iter()
                .enumerate()
                .filter(|(ii, _)| !remove_positions.contains(&(bi, *ii)))
                .map(|(_, inst)| inst)
                .collect();
        }
    }

    // Create ONE canonical Ref per written context in the entry block.
    // These are never removed by forwarding (no associated Load).
    // Write-back Stores reference these canonical Refs.
    let mut ctx_ref_cache: FxHashMap<QualifiedRef, ValueId> = FxHashMap::default();
    let mut canonical_ref_insts: Vec<Inst> = Vec::new();
    for &ctx in &ssa_info.written_contexts {
        let inner_ty = ssa_info
            .ctx_types
            .get(&ctx)
            .expect("written context must have a type")
            .clone();
        let ref_dst = alloc_val(
            &mut cfg.val_factory,
            &mut cfg.val_types,
            Ty::Ref(Box::new(inner_ty), false),
        );
        canonical_ref_insts.push(Inst {
            span: acvus_ast::Span::ZERO,
            kind: InstKind::Ref {
                dst: ref_dst,
                target: crate::ir::RefTarget::Context(ctx),
                path: vec![],
            },
        });
        ctx_ref_cache.insert(ctx, ref_dst);
    }
    if !canonical_ref_insts.is_empty() {
        cfg.blocks[0].insts.splice(0..0, canonical_ref_insts);
    }

    // Add PHI params to block params + insert write-back ContextStores.
    for (&label, phis) in &block_phis {
        if let Some(&block_idx) = cfg.label_to_block.get(&label) {
            let block = &mut cfg.blocks[block_idx.0];

            for phi in phis {
                block.params.push(phi.result);
            }

            // Write-back ContextStores for context PHI values.
            // Local variable PHIs do NOT need write-back.
            // NOTE: phi result/operand types are set in run() before this call.
            let mut write_back_insts: Vec<Inst> = Vec::new();
            for phi in phis {
                let ctx = match phi.var {
                    SsaVar::Context(ctx) => ctx,
                    SsaVar::Local(_) => continue,
                };

                // Use canonical Ref from entry block (always exists for written contexts).
                let ref_dst = ctx_ref_cache[&ctx];

                write_back_insts.push(Inst {
                    span: acvus_ast::Span::ZERO,
                    kind: InstKind::Store {
                        dst: ref_dst,
                        value: phi.result,
                        volatile: false,
                    },
                });
            }

            if !write_back_insts.is_empty() {
                let block = &mut cfg.blocks[block_idx.0];
                block.insts.splice(0..0, write_back_insts);
            }
        }
    }

    // Add jump args to terminators of predecessor blocks.
    for block in cfg.blocks.iter_mut() {
        let pred_label = block.label;

        match &mut block.terminator {
            Terminator::Jump { label, args } => {
                if let Some(extra) = jump_extra_args.get(&(pred_label, *label)) {
                    args.extend_from_slice(extra);
                }
            }
            Terminator::JumpIf {
                then_label,
                then_args,
                else_label,
                else_args,
                ..
            } => {
                if let Some(extra) = jump_extra_args.get(&(pred_label, *then_label)) {
                    then_args.extend_from_slice(extra);
                }
                if let Some(extra) = jump_extra_args.get(&(pred_label, *else_label)) {
                    else_args.extend_from_slice(extra);
                }
            }
            Terminator::ListStep {
                done, done_args, ..
            } => {
                if let Some(extra) = jump_extra_args.get(&(pred_label, *done)) {
                    done_args.extend_from_slice(extra);
                }
            }
            _ => {}
        }
    }
}

// ── Step 4: Apply var substitutions + remove VarLoad/VarStore/ParamLoad ──

fn apply_var_subst(cfg: &mut CfgBody, var_subst: &FxHashMap<ValueId, ValueId>, ssa_info: &SsaInfo) {
    use crate::ir::RefTarget;

    // Collect the set of substituted Load dsts — these Loads are dead.
    let substituted_loads: FxHashSet<ValueId> = var_subst.keys().copied().collect();

    // Collect all users of each Ref dst: which Load/Store instructions reference it.
    let mut ref_users: FxHashMap<ValueId, Vec<ValueId>> = FxHashMap::default();
    for block in &cfg.blocks {
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Load { dst, src, .. } => {
                    ref_users.entry(*src).or_default().push(*dst);
                }
                InstKind::Store { dst, .. } => {
                    // Store's dst is the Ref ValueId. The Store itself is a "user".
                    // Use a sentinel to distinguish from Load users.
                    ref_users.entry(*dst).or_default();
                }
                _ => {}
            }
        }
    }

    // Collect Ref ValueIds that can be safely removed:
    // - Points to a promoted Var/Param
    // - ALL Loads through this Ref have been substituted (in substituted_loads)
    // - ALL Stores through this Ref target a promoted variable
    let mut promoted_refs: FxHashSet<ValueId> = FxHashSet::default();
    for block in &cfg.blocks {
        for inst in &block.insts {
            if let InstKind::Ref {
                dst,
                target,
                path,
            } = &inst.kind
                && path.is_empty()
            {
                let is_promoted = match target {
                    RefTarget::Context(_) => false,
                    RefTarget::Var(name) | RefTarget::Param(name) => {
                        ssa_info.written_vars.contains(name)
                            || ssa_info.entry_param_defs.contains_key(name)
                            || ssa_info.read_vars.contains(name)
                    }
                };
                if !is_promoted {
                    continue;
                }

                // Check: all Load users of this Ref are substituted.
                let all_loads_dead = ref_users
                    .get(dst)
                    .map(|users| users.iter().all(|u| substituted_loads.contains(u)))
                    .unwrap_or(true);

                if all_loads_dead {
                    promoted_refs.insert(*dst);
                }
            }
        }
    }

    // Apply substitutions to all instruction operands and terminators.
    for block in &mut cfg.blocks {
        for inst in &mut block.insts {
            apply_subst(&mut inst.kind, var_subst);
        }
        apply_subst_terminator(&mut block.terminator, var_subst);

        // Remove dead instructions for promoted storage:
        // - Refs whose ALL users are dead (substituted Loads, promoted Stores)
        // - Loads whose dst was substituted
        // - Stores whose dst Ref is promoted
        block.insts.retain(|inst| match &inst.kind {
            InstKind::Ref { dst, .. } if promoted_refs.contains(dst) => false,
            InstKind::Load { dst, .. } if substituted_loads.contains(dst) => false,
            InstKind::Store { dst, .. } if promoted_refs.contains(dst) => false,
            _ => true,
        });
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{self, CfgBody};
    use crate::ir::MirBody;
    use crate::test::{compile_script, compile_template};
    use crate::ty::{EffectSet, EffectTarget, Ty};
    use acvus_utils::Interner;
    use std::collections::BTreeSet;

    /// Empty fn_metadata — no ExternFn effect info.
    fn no_fn_metadata() -> FxHashMap<QualifiedRef, FnMetadata> {
        FxHashMap::default()
    }

    fn count_phi_blocks(cfg_body: &CfgBody) -> usize {
        cfg_body
            .blocks
            .iter()
            .filter(|b| !b.params.is_empty())
            .count()
    }

    fn count_context_stores(cfg_body: &CfgBody) -> usize {
        cfg_body
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(&i.kind, InstKind::Store { .. }))
            .count()
    }

    // ── Completeness: PHI inserted when needed ──

    #[test]
    fn match_one_arm_write_phi() {
        let i = Interner::new();
        let module = compile_template(
            &i,
            r#"{{ true = @name == "test" }}{{ @x = 42 }}{{ _ }}noop{{/}}"#,
            &[("x", Ty::Int), ("name", Ty::String)],
        )
        .unwrap();
        let mut cfg_body = cfg::promote(module.main);
        run(&mut cfg_body, &no_fn_metadata());
        assert!(
            count_phi_blocks(&cfg_body) >= 1,
            "merge should have PHI for @x"
        );
        assert!(
            count_context_stores(&cfg_body) >= 1,
            "should have write-back Store"
        );
    }

    #[test]
    fn match_both_arms_write_phi() {
        let i = Interner::new();
        let module = compile_template(
            &i,
            r#"{{ true = @name == "test" }}{{ @x = 1 }}{{ _ }}{{ @x = 2 }}{{/}}"#,
            &[("x", Ty::Int), ("name", Ty::String)],
        )
        .unwrap();
        let mut cfg_body = cfg::promote(module.main);
        run(&mut cfg_body, &no_fn_metadata());
        assert!(count_phi_blocks(&cfg_body) >= 1);
    }

    #[test]
    fn iter_context_write_phi() {
        let i = Interner::new();
        let module = compile_template(
            &i,
            r#"{{ x in @items }}{{ @sum = @sum + x }}{{/}}"#,
            &[("items", Ty::List(Box::new(Ty::Int))), ("sum", Ty::Int)],
        )
        .unwrap();
        let mut cfg_body = cfg::promote(module.main);
        run(&mut cfg_body, &no_fn_metadata());
        assert!(
            count_phi_blocks(&cfg_body) >= 1,
            "loop header should have PHI for @sum"
        );
    }

    // ── Soundness: PHI NOT inserted when not needed ──

    #[test]
    fn match_no_write_no_phi() {
        let i = Interner::new();
        let module = compile_template(
            &i,
            r#"{{ true = @name == "test" }}yes{{ _ }}no{{/}}"#,
            &[("name", Ty::String)],
        )
        .unwrap();
        let mut cfg_body = cfg::promote(module.main);
        let stores_before = count_context_stores(&cfg_body);
        run(&mut cfg_body, &no_fn_metadata());
        assert_eq!(count_context_stores(&cfg_body), stores_before);
    }

    #[test]
    fn straight_line_write_preserved() {
        let i = Interner::new();
        let module = compile_script(&i, "@x = 42; @x", &[("x", Ty::Int)]).unwrap();
        let mut cfg_body = cfg::promote(module.main);
        let stores_before = count_context_stores(&cfg_body);
        run(&mut cfg_body, &no_fn_metadata());
        assert_eq!(count_context_stores(&cfg_body), stores_before);
    }

    // ── Script regression: entry loads + nested loops ──

    #[test]
    fn script_entry_loads_all_contexts() {
        // All contexts must have Ref + Load in entry block.
        let i = Interner::new();
        let module = compile_script(
            &i,
            "x in @items { @sum = @sum + x; }; @sum",
            &[("items", Ty::List(Box::new(Ty::Int))), ("sum", Ty::Int)],
        )
        .unwrap();
        let cfg_body = cfg::promote(module.main);

        // Entry block is blocks[0].
        let entry = &cfg_body.blocks[0];
        let entry_projects: Vec<_> = entry
            .insts
            .iter()
            .filter(|i| matches!(i.kind, InstKind::Ref { .. }))
            .collect();
        let entry_loads: Vec<_> = entry
            .insts
            .iter()
            .filter(|i| matches!(i.kind, InstKind::Load { .. }))
            .collect();
        // Both @items and @sum should have entry loads.
        assert!(
            entry_projects.len() >= 2,
            "expected entry Ref for all contexts"
        );
        assert!(
            entry_loads.len() >= 2,
            "expected entry Load for all contexts"
        );
    }

    #[test]
    fn script_nested_loop_phi() {
        // Nested loop must not panic in SSA pass (regression: ListStep CFG terminator).
        let i = Interner::new();
        let module = compile_script(
            &i,
            "row in @matrix { x in row { @sum = @sum + x; }; }; @sum",
            &[
                ("matrix", Ty::List(Box::new(Ty::List(Box::new(Ty::Int))))),
                ("sum", Ty::Int),
            ],
        )
        .unwrap();
        let cfg_body = cfg::promote(module.main);
        // Must have PHI for @sum (written in inner loop).
        assert!(
            count_phi_blocks(&cfg_body) >= 1,
            "nested loop should produce PHI for @sum"
        );
    }

    #[test]
    fn script_loop_with_branch_phi() {
        // Loop body with conditional context write — needs PHI.
        let i = Interner::new();
        let module = compile_script(
            &i,
            "x in @items { 0 = x { @count = @count + 1; }; }; @count",
            &[("items", Ty::List(Box::new(Ty::Int))), ("count", Ty::Int)],
        )
        .unwrap();
        let cfg_body = cfg::promote(module.main);
        assert!(
            count_phi_blocks(&cfg_body) >= 1,
            "loop + branch should produce PHI"
        );
    }

    // ── FunctionCall context_uses/context_defs population ──

    /// Build a minimal CfgBody with a FunctionCall to a callee that reads and writes @ctx.
    /// Before SSA pass: context_uses/context_defs are empty.
    /// After SSA pass: they should be populated.
    #[test]
    fn function_call_populates_context_uses_defs() {
        let interner = Interner::new();
        let ctx_name = interner.intern("ctx");
        let qref = QualifiedRef::root(ctx_name);
        let callee_id = QualifiedRef::root(interner.intern("callee"));

        // Build fn_metadata: callee reads + writes @ctx.
        let mut fn_metadata = FxHashMap::default();
        fn_metadata.insert(
            callee_id,
            FnMetadata {
                ty: Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::Int),
                    captures: vec![],
                    effect: Effect::Resolved(EffectSet {
                        reads: BTreeSet::from([EffectTarget::Context(qref)]),
                        writes: BTreeSet::from([EffectTarget::Context(qref)]),
                        self_modifying: false,
                    }),
                },
                hint: None,
            },
        );

        // Build MIR manually then promote:
        // v0 = Ref @ctx
        // v1 = Load v0        (entry load)
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
            kind: InstKind::Ref {
                dst: v0,
                target: crate::ir::RefTarget::Context(qref),
                path: vec![],
            },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::Load {
                dst: v1,
                src: v0,
                volatile: false,
            },
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

        let mut cfg_body = cfg::promote(body);
        run(&mut cfg_body, &fn_metadata);

        // Find the FunctionCall and verify context_uses/context_defs are populated.
        let call_inst = cfg_body
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
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
        let mut fn_metadata = FxHashMap::default();
        fn_metadata.insert(
            callee_id,
            FnMetadata {
                ty: Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::Int),
                    captures: vec![],
                    effect: Effect::pure(),
                },
                hint: None,
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

        let mut cfg_body = cfg::promote(body);
        run(&mut cfg_body, &fn_metadata);

        let call_inst = cfg_body
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
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

    /// After a FunctionCall that writes @ctx, subsequent Load should see the new value.
    #[test]
    fn function_call_def_forwards_to_subsequent_load() {
        let interner = Interner::new();
        let ctx_name = interner.intern("ctx");
        let qref = QualifiedRef::root(ctx_name);
        let callee_id = QualifiedRef::root(interner.intern("callee"));

        let mut fn_metadata = FxHashMap::default();
        fn_metadata.insert(
            callee_id,
            FnMetadata {
                ty: Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::Unit),
                    captures: vec![],
                    effect: Effect::Resolved(EffectSet {
                        reads: BTreeSet::new(),
                        writes: BTreeSet::from([EffectTarget::Context(qref)]),
                        self_modifying: false,
                    }),
                },
                hint: None,
            },
        );

        // v0 = Ref @ctx
        // v1 = Load v0
        // v2 = FunctionCall callee()    ← writes @ctx, def = v_new
        // v3 = Ref @ctx
        // v4 = Load v3           ← should be replaced by v_new
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
            kind: InstKind::Ref {
                dst: v0,
                target: crate::ir::RefTarget::Context(qref),
                path: vec![],
            },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::Load {
                dst: v1,
                src: v0,
                volatile: false,
            },
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
            kind: InstKind::Ref {
                dst: v3,
                target: crate::ir::RefTarget::Context(qref),
                path: vec![],
            },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::Load {
                dst: v4,
                src: v3,
                volatile: false,
            },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::Return(v4),
        });

        let mut cfg_body = cfg::promote(body);
        run(&mut cfg_body, &fn_metadata);

        // The second Load (v4) should be eliminated — replaced by the def from the call.
        let remaining_loads: Vec<_> = cfg_body
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i.kind, InstKind::Load { .. }))
            .collect();
        // Only the entry load should remain; the second should be forwarded.
        assert_eq!(
            remaining_loads.len(),
            1,
            "second Load should be eliminated by forwarding from FunctionCall def"
        );
    }

    #[test]
    fn script_sequential_loops_no_panic() {
        // Two sequential loops writing same context must not panic.
        let i = Interner::new();
        let module = compile_script(
            &i,
            "x in @a { @sum = @sum + x; }; y in @b { @sum = @sum + y; }; @sum",
            &[
                ("a", Ty::List(Box::new(Ty::Int))),
                ("b", Ty::List(Box::new(Ty::Int))),
                ("sum", Ty::Int),
            ],
        )
        .unwrap();
        let cfg_body = cfg::promote(module.main);
        // Both loops write @sum — SSA pass must handle this.
        assert!(count_context_stores(&cfg_body) >= 1);
    }

    // ── Token/Context soundness tests ──────────────────────────────

    /// Token read must NOT appear in context_uses after SSA pass.
    #[test]
    fn token_read_not_in_context_uses() {
        let interner = Interner::new();
        let ctx_name = interner.intern("ctx");
        let qref = QualifiedRef::root(ctx_name);
        let token = QualifiedRef::root(interner.intern("net"));
        let callee_id = QualifiedRef::root(interner.intern("callee"));

        let mut fn_metadata = FxHashMap::default();
        fn_metadata.insert(
            callee_id,
            FnMetadata {
                ty: Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::Int),
                    captures: vec![],
                    effect: Effect::Resolved(EffectSet {
                        reads: BTreeSet::from([
                            EffectTarget::Context(qref),
                            EffectTarget::Token(token),
                        ]),
                        writes: BTreeSet::new(),
                        self_modifying: false,
                    }),
                },
                hint: None,
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
            kind: InstKind::Ref {
                dst: v0,
                target: crate::ir::RefTarget::Context(qref),
                path: vec![],
            },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::Load {
                dst: v1,
                src: v0,
                volatile: false,
            },
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

        let mut cfg_body = cfg::promote(body);
        run(&mut cfg_body, &fn_metadata);

        // context_uses should have exactly 1 entry (the Context), NOT 2.
        for block in &cfg_body.blocks {
            for inst in &block.insts {
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
    }

    /// Token write must NOT appear in context_defs after SSA pass.
    #[test]
    fn token_write_not_in_context_defs() {
        let interner = Interner::new();
        let ctx_name = interner.intern("ctx");
        let qref = QualifiedRef::root(ctx_name);
        let token = QualifiedRef::root(interner.intern("fs"));
        let callee_id = QualifiedRef::root(interner.intern("callee"));

        let mut fn_metadata = FxHashMap::default();
        fn_metadata.insert(
            callee_id,
            FnMetadata {
                ty: Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::Unit),
                    captures: vec![],
                    effect: Effect::Resolved(EffectSet {
                        reads: BTreeSet::new(),
                        writes: BTreeSet::from([
                            EffectTarget::Context(qref),
                            EffectTarget::Token(token),
                        ]),
                        self_modifying: false,
                    }),
                },
                hint: None,
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
            kind: InstKind::Ref {
                dst: v0,
                target: crate::ir::RefTarget::Context(qref),
                path: vec![],
            },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::Load {
                dst: v1,
                src: v0,
                volatile: false,
            },
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

        let mut cfg_body = cfg::promote(body);
        run(&mut cfg_body, &fn_metadata);

        for block in &cfg_body.blocks {
            for inst in &block.insts {
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
    }

    /// Mixed Context + Token: only Context appears in SSA uses/defs.
    #[test]
    fn mixed_context_and_token_only_context_in_ssa() {
        let interner = Interner::new();
        let ctx_name = interner.intern("ctx");
        let qref = QualifiedRef::root(ctx_name);
        let token = QualifiedRef::root(interner.intern("db"));
        let callee_id = QualifiedRef::root(interner.intern("callee"));

        let mut fn_metadata = FxHashMap::default();
        fn_metadata.insert(
            callee_id,
            FnMetadata {
                ty: Ty::Fn {
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
                        self_modifying: false,
                    }),
                },
                hint: None,
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
            kind: InstKind::Ref {
                dst: v0,
                target: crate::ir::RefTarget::Context(qref),
                path: vec![],
            },
        });
        body.insts.push(Inst {
            span,
            kind: InstKind::Load {
                dst: v1,
                src: v0,
                volatile: false,
            },
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

        let mut cfg_body = cfg::promote(body);
        run(&mut cfg_body, &fn_metadata);

        for block in &cfg_body.blocks {
            for inst in &block.insts {
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
    }

    /// EffectSet union preserves both Context and Token targets.
    #[test]
    fn effect_set_union_preserves_context_and_token() {
        let interner = Interner::new();
        let qref = QualifiedRef::root(interner.intern("ctx"));
        let token = QualifiedRef::root(interner.intern("net"));

        let a = EffectSet {
            reads: BTreeSet::from([EffectTarget::Context(qref)]),
            ..Default::default()
        };
        let b = EffectSet {
            reads: BTreeSet::from([EffectTarget::Token(token)]),
            ..Default::default()
        };
        let u = a.union(&b);
        assert_eq!(
            u.reads.len(),
            2,
            "union should contain both Context and Token"
        );
        assert!(u.reads.contains(&EffectTarget::Context(qref)));
        assert!(u.reads.contains(&EffectTarget::Token(token)));
    }

    /// extract_effect_refs filters Token targets out.
    #[test]
    fn extract_effect_refs_filters_tokens() {
        let interner = Interner::new();
        let qref = QualifiedRef::root(interner.intern("ctx"));
        let token = QualifiedRef::root(interner.intern("fs"));
        let fid = QualifiedRef::root(interner.intern("callee"));

        let mut fn_metadata = FxHashMap::default();
        fn_metadata.insert(
            fid,
            FnMetadata {
                ty: Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::Int),
                    captures: vec![],
                    effect: Effect::Resolved(EffectSet {
                        reads: BTreeSet::from([
                            EffectTarget::Context(qref),
                            EffectTarget::Token(token),
                        ]),
                        writes: BTreeSet::from([EffectTarget::Token(token)]),
                        self_modifying: false,
                    }),
                },
                hint: None,
            },
        );

        let result = extract_effect_refs(&fn_metadata, &fid);
        let (reads, writes) = result.unwrap();
        assert_eq!(reads.len(), 1, "only Context should be in reads");
        assert_eq!(reads[0], qref);
        assert!(writes.is_empty(), "Token-only writes should yield empty");
    }

    // ── Volatile context: forwarding must be skipped ──

    fn count_context_loads(cfg_body: &CfgBody) -> usize {
        cfg_body
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(&i.kind, InstKind::Load { .. }))
            .count()
    }

    /// Build a minimal CfgBody with: Ref → Store → Load → Return.
    /// When volatile=false, SSA should forward the store value and eliminate the load.
    /// When volatile=true, SSA must preserve the load.
    fn make_store_then_load(volatile: bool) -> (CfgBody, FxHashMap<QualifiedRef, FnMetadata>) {
        use acvus_utils::LocalFactory;
        let interner = Interner::new();
        let ctx_qref = QualifiedRef::root(interner.intern("history"));
        let mut f = LocalFactory::<ValueId>::new();
        let v: Vec<ValueId> = (0..5).map(|_| f.next()).collect();
        // v0 = const 42
        // v1 = ref @history
        // store v1, v0
        // v2 = ref @history  (for the load)
        // v3 = load v2
        // return v3
        let insts = vec![
            Inst {
                span: acvus_ast::Span::ZERO,
                kind: InstKind::Const {
                    dst: v[0],
                    value: acvus_ast::Literal::Int(42),
                },
            },
            Inst {
                span: acvus_ast::Span::ZERO,
                kind: InstKind::Ref {
                    dst: v[1],
                    target: crate::ir::RefTarget::Context(ctx_qref),
                    path: vec![],
                },
            },
            Inst {
                span: acvus_ast::Span::ZERO,
                kind: InstKind::Store {
                    dst: v[1],
                    value: v[0],
                    volatile,
                },
            },
            Inst {
                span: acvus_ast::Span::ZERO,
                kind: InstKind::Ref {
                    dst: v[2],
                    target: crate::ir::RefTarget::Context(ctx_qref),
                    path: vec![],
                },
            },
            Inst {
                span: acvus_ast::Span::ZERO,
                kind: InstKind::Load {
                    dst: v[3],
                    src: v[2],
                    volatile,
                },
            },
            Inst {
                span: acvus_ast::Span::ZERO,
                kind: InstKind::Return(v[3]),
            },
        ];
        let mut val_types = FxHashMap::default();
        for &vid in &v {
            val_types.insert(vid, Ty::Int);
        }
        let body = MirBody {
            insts,
            val_types,
            params: Vec::new(),
            captures: Vec::new(),
            debug: crate::ir::DebugInfo::new(),
            val_factory: f,
            label_count: 0,
        };
        (cfg::promote(body), no_fn_metadata())
    }

    #[test]
    fn non_volatile_context_is_forwarded() {
        let (mut cfg, fn_metadata) = make_store_then_load(false);
        let loads_before = count_context_loads(&cfg);
        run(&mut cfg, &fn_metadata);
        let loads_after = count_context_loads(&cfg);
        // Non-volatile: SSA should forward the store → load is eliminated.
        assert!(
            loads_after < loads_before,
            "non-volatile context load should be forwarded (before={}, after={})",
            loads_before,
            loads_after
        );
    }

    #[test]
    fn volatile_context_not_forwarded() {
        let (mut cfg, fn_metadata) = make_store_then_load(true);
        let loads_before = count_context_loads(&cfg);
        run(&mut cfg, &fn_metadata);
        let loads_after = count_context_loads(&cfg);
        // Volatile: SSA must NOT forward — load is preserved.
        assert_eq!(
            loads_before, loads_after,
            "volatile context load must not be forwarded (before={}, after={})",
            loads_before, loads_after
        );
    }
}
