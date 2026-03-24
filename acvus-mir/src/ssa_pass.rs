//! SSA Context Pass (mem2reg for context variables)
//!
//! Promotes ContextProject/ContextLoad/ContextStore to SSA form.
//! Self-contained: inserts initial loads, computes PHIs, patches instructions.
//!
//! Write-back model: branch-internal ContextStores are removed;
//! a single write-back ContextStore is inserted after each merge block.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::cfg::{BlockIdx, Cfg, Terminator};
use crate::graph::ContextId;
use crate::ir::{Callee, Inst, InstKind, Label, MirBody, ValueId};
use crate::ssa::{ENTRY_BLOCK, SSABuilder};
use crate::ty::Ty;

/// Run the SSA context pass on a MirBody.
pub fn run(body: &mut MirBody) {
    // Step 1: Build CFG + collect context ops.
    let cfg = Cfg::build(&body.insts);
    if cfg.blocks.is_empty() {
        return;
    }
    let ctx_info = collect_context_info(&cfg, &body.insts);

    // Step 2: Run SSABuilder + patch PHIs (only if there are writes + merge points).
    if !ctx_info.written_contexts.is_empty() {
        let preds = cfg.predecessors();
        let phi_insertions = run_ssa_builder(&cfg, &preds, &ctx_info, &mut body.val_factory);
        if !phi_insertions.is_empty() {
            patch_instructions(body, &cfg, &phi_insertions, &ctx_info);
        }
    }

    // Step 3: Forward context values — eliminate redundant loads.
    forward_context_values(body);
}

/// Store-load forwarding for context variables.
///
/// Tracks the "current SSA value" of each context. When a ContextLoad
/// follows a ContextStore to the same context (with no intervening branch),
/// the load is replaced by the stored value.
///
/// Also eliminates the initial entry load if the context is never read
/// before it's written (dead initial load).
fn forward_context_values(body: &mut MirBody) {
    // Map: projection ValueId → ContextId.
    let mut val_to_ctx: FxHashMap<ValueId, ContextId> = FxHashMap::default();
    // Current known value per context (from entry load or store).
    let mut current_val: FxHashMap<ContextId, ValueId> = FxHashMap::default();
    // ValueId substitutions: old → new.
    let mut subst: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    // Instructions to remove (dead loads + their preceding projects).
    let mut remove: FxHashSet<usize> = FxHashSet::default();

    for (i, inst) in body.insts.iter().enumerate() {
        match &inst.kind {
            InstKind::ContextProject { dst, id, .. } => {
                val_to_ctx.insert(*dst, *id);
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
            // Branch/merge: invalidate tracked values (conservative).
            InstKind::BlockLabel { .. } | InstKind::Jump { .. } | InstKind::JumpIf { .. } => {
                current_val.clear();
            }
            _ => {}
        }
    }

    if remove.is_empty() && subst.is_empty() {
        return;
    }

    // Apply substitutions and remove dead instructions.
    let old_insts = std::mem::take(&mut body.insts);
    body.insts = old_insts
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !remove.contains(i))
        .map(|(_, mut inst)| {
            apply_subst(&mut inst.kind, &subst);
            inst
        })
        .collect();
}

/// Apply value substitutions to an instruction's operands.
fn apply_subst(kind: &mut InstKind, subst: &FxHashMap<ValueId, ValueId>) {
    let s = |v: &mut ValueId| {
        if let Some(&new) = subst.get(v) {
            *v = new;
        }
    };
    match kind {
        InstKind::Const { .. } | InstKind::Nop | InstKind::Poison { .. } => {}
        InstKind::ContextProject { .. } => {}
        InstKind::ContextLoad { src, .. } => s(src),
        InstKind::ContextStore { dst, value } => {
            s(dst);
            s(value);
        }
        InstKind::VarLoad { .. } => {}
        InstKind::VarStore { src, .. } => s(src),
        InstKind::BinOp { left, right, .. } => {
            s(left);
            s(right);
        }
        InstKind::UnaryOp { operand, .. } => s(operand),
        InstKind::FieldGet { object, .. } => s(object),
        InstKind::LoadFunction { .. } => {}
        InstKind::FunctionCall { callee, args, .. } => {
            if let Callee::Indirect(v) = callee {
                s(v);
            }
            args.iter_mut().for_each(&s);
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
        InstKind::IterStep {
            iter_src,
            done_args,
            ..
        } => {
            s(iter_src);
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

// ── Step 0: Ensure initial loads ────────────────────────────────────

/// Scan the MIR for all ContextIds referenced by ContextProject.
/// For any context that doesn't have a ContextLoad in the entry region
/// (before the first BlockLabel), insert ContextProject + ContextLoad.
fn ensure_initial_loads(body: &mut MirBody) {
    // Find the first BlockLabel position — everything before it is the entry region.
    let first_label_pos = body
        .insts
        .iter()
        .position(|i| matches!(&i.kind, InstKind::BlockLabel { .. }))
        .unwrap_or(body.insts.len());

    // Collect all referenced ContextIds and which ones already have loads in entry.
    let mut all_ctx_ids: FxHashSet<ContextId> = FxHashSet::default();
    let mut ctx_types: FxHashMap<ContextId, Ty> = FxHashMap::default();
    let mut entry_loaded: FxHashSet<ContextId> = FxHashSet::default();
    let mut val_to_ctx: FxHashMap<ValueId, ContextId> = FxHashMap::default();

    for (i, inst) in body.insts.iter().enumerate() {
        match &inst.kind {
            InstKind::ContextProject { dst, id, ty } => {
                all_ctx_ids.insert(*id);
                ctx_types.entry(*id).or_insert_with(|| ty.clone());
                val_to_ctx.insert(*dst, *id);
            }
            InstKind::ContextLoad { src, .. } if i < first_label_pos => {
                if let Some(&ctx_id) = val_to_ctx.get(src) {
                    entry_loaded.insert(ctx_id);
                }
            }
            InstKind::FieldGet { dst, object, .. } => {
                if let Some(&ctx_id) = val_to_ctx.get(object) {
                    val_to_ctx.insert(*dst, ctx_id);
                }
            }
            _ => {}
        }
    }

    // Insert initial loads for contexts that don't have one.
    // Collect missing contexts in order of first appearance (deterministic).
    let mut seen = FxHashSet::default();
    let mut missing: Vec<ContextId> = Vec::new();
    for inst in body.insts.iter() {
        if let InstKind::ContextProject { id, .. } = &inst.kind
            && all_ctx_ids.contains(id)
            && !entry_loaded.contains(id)
            && seen.insert(*id)
        {
            missing.push(*id);
        }
    }
    if missing.is_empty() {
        return;
    }

    let span = body
        .insts
        .first()
        .map(|i| i.span)
        .unwrap_or(acvus_ast::Span::new(0, 0));

    // Insert at position 0 (before all other instructions).
    let mut prefix = Vec::with_capacity(missing.len() * 2);
    for ctx_id in missing {
        let ty = ctx_types
            .get(&ctx_id)
            .expect("missing ty for context")
            .clone();
        let proj = body.val_factory.next();
        body.val_types.insert(proj, ty.clone());
        prefix.push(Inst {
            span,
            kind: InstKind::ContextProject {
                dst: proj,
                id: ctx_id,
                ty: ty.clone(),
            },
        });
        let val = body.val_factory.next();
        body.val_types.insert(val, ty);
        prefix.push(Inst {
            span,
            kind: InstKind::ContextLoad {
                dst: val,
                src: proj,
            },
        });
    }

    prefix.append(&mut body.insts);
    body.insts = prefix;
}

// ── Step 1: Context info collection ─────────────────────────────────

/// Per-block context operations.
#[derive(Debug, Default)]
struct BlockContextOps {
    /// (inst_index, ctx_id, value)
    stores: Vec<(usize, ContextId, ValueId)>,
}

/// Aggregated context info.
struct ContextInfo {
    written_contexts: FxHashSet<ContextId>,
    block_ops: FxHashMap<BlockIdx, BlockContextOps>,
    /// ctx → initial loaded value (from entry region ContextLoad).
    entry_defs: FxHashMap<ContextId, ValueId>,
    /// ctx → root type (from ContextProject instructions).
    ctx_types: FxHashMap<ContextId, Ty>,
}

fn collect_context_info(cfg: &Cfg, insts: &[Inst]) -> ContextInfo {
    let mut val_to_ctx: FxHashMap<ValueId, ContextId> = FxHashMap::default();
    let mut written_contexts = FxHashSet::default();
    let mut block_ops: FxHashMap<BlockIdx, BlockContextOps> = FxHashMap::default();
    let mut entry_defs: FxHashMap<ContextId, ValueId> = FxHashMap::default();
    let mut ctx_types: FxHashMap<ContextId, Ty> = FxHashMap::default();

    for (bi, block) in cfg.blocks.iter().enumerate() {
        let ops = block_ops.entry(BlockIdx(bi)).or_default();

        for &inst_i in &block.inst_indices {
            let inst = &insts[inst_i];
            match &inst.kind {
                InstKind::ContextProject { dst, id, ty } => {
                    val_to_ctx.insert(*dst, *id);
                    ctx_types.entry(*id).or_insert_with(|| ty.clone());
                }
                InstKind::ContextLoad { dst, src } => {
                    if let Some(&ctx_id) = val_to_ctx.get(src) {
                        // Entry block loads = initial definitions.
                        if bi == 0 {
                            entry_defs.entry(ctx_id).or_insert(*dst);
                        }
                    }
                }
                InstKind::ContextStore { dst, value } => {
                    if let Some(&ctx_id) = val_to_ctx.get(dst) {
                        ops.stores.push((inst_i, ctx_id, *value));
                        written_contexts.insert(ctx_id);
                    }
                }
                InstKind::FieldGet { dst, object, .. } => {
                    if let Some(&ctx_id) = val_to_ctx.get(object) {
                        val_to_ctx.insert(*dst, ctx_id);
                    }
                }
                _ => {}
            }
        }
    }

    ContextInfo {
        written_contexts,
        block_ops,
        entry_defs,
        ctx_types,
    }
}

// ── Step 2: SSABuilder execution ────────────────────────────────────

fn run_ssa_builder(
    cfg: &Cfg,
    preds: &FxHashMap<BlockIdx, smallvec::SmallVec<[BlockIdx; 2]>>,
    ctx_info: &ContextInfo,
    val_factory: &mut acvus_utils::LocalFactory<ValueId>,
) -> Vec<crate::ssa::PhiInsertion> {
    let mut ssa = SSABuilder::new();

    let block_label = |bi: BlockIdx| -> Label {
        if bi.0 == 0 {
            ENTRY_BLOCK
        } else {
            cfg.blocks[bi.0].label.unwrap_or(Label(bi.0 as u32))
        }
    };

    // Detect loop headers (backedge target: succ index <= current index).
    let mut loop_headers: FxHashSet<BlockIdx> = FxHashSet::default();
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

    // Define initial values in entry block (guaranteed by Step 0).
    for (&ctx_id, &val) in &ctx_info.entry_defs {
        ssa.define(ENTRY_BLOCK, ctx_id, val);
    }

    // Process blocks: define writes, seal non-loop-headers.
    for (bi, _) in cfg.blocks.iter().enumerate() {
        let block_idx = BlockIdx(bi);
        let label = block_label(block_idx);

        if let Some(ops) = ctx_info.block_ops.get(&block_idx) {
            for &(_, ctx_id, value) in &ops.stores {
                ssa.define(label, ctx_id, value);
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

    // Trigger PHIs at merge points.
    for (block_idx, block_preds) in preds {
        if block_preds.len() > 1 {
            let label = block_label(*block_idx);
            for &ctx_id in &ctx_info.written_contexts {
                let _ = ssa.use_var(label, ctx_id, &mut || val_factory.next());
            }
        }
    }

    ssa.finish()
}

// ── Step 3: Patch instructions ──────────────────────────────────────

fn patch_instructions(
    body: &mut MirBody,
    cfg: &Cfg,
    phi_insertions: &[crate::ssa::PhiInsertion],
    ctx_info: &ContextInfo,
) {
    // PHI lookup tables.
    let mut block_phis: FxHashMap<Label, Vec<&crate::ssa::PhiInsertion>> = FxHashMap::default();
    for phi in phi_insertions {
        block_phis.entry(phi.block).or_default().push(phi);
    }

    let mut jump_extra_args: FxHashMap<(Label, Label), Vec<ValueId>> = FxHashMap::default();
    for phi in phi_insertions {
        for &(pred, val) in &phi.incoming {
            jump_extra_args
                .entry((pred, phi.block))
                .or_default()
                .push(val);
        }
    }

    // Identify ContextStores to remove (in branches superseded by PHI write-back).
    let phi_contexts: FxHashSet<ContextId> = phi_insertions.iter().map(|p| p.context).collect();
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
            Terminator::IterStep { done, .. } => merge_labels.contains(done),
            _ => false,
        };
        if jumps_to_merge && let Some(ops) = ctx_info.block_ops.get(&BlockIdx(bi)) {
            for &(inst_i, ctx_id, _) in &ops.stores {
                if phi_contexts.contains(&ctx_id) {
                    remove_indices.insert(inst_i);
                    // Remove preceding ContextProject if it exists.
                    if inst_i > 0
                        && matches!(body.insts[inst_i - 1].kind, InstKind::ContextProject { .. })
                    {
                        remove_indices.insert(inst_i - 1);
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

                // Write-back ContextStores for PHI values.
                if let Some(phis) = phis_here {
                    for phi in phis {
                        let ty = ctx_info
                            .ctx_types
                            .get(&phi.context)
                            .expect("missing ty for context in PHI write-back")
                            .clone();
                        // Set type for PHI result value.
                        body.val_types.insert(phi.result, ty.clone());
                        let proj = body.val_factory.next();
                        body.val_types.insert(proj, ty.clone());
                        new_insts.push(Inst {
                            span,
                            kind: InstKind::ContextProject {
                                dst: proj,
                                id: phi.context,
                                ty,
                            },
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

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::{compile_script, compile_template};
    use crate::ty::Ty;
    use acvus_utils::Interner;

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
        run(&mut module.main);
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
        run(&mut module.main);
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
        run(&mut module.main);
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
        run(&mut module.main);
        assert_eq!(count_context_stores(&module.main), stores_before);
    }

    #[test]
    fn iter_no_write_no_phi() {
        let i = Interner::new();
        let (mut module, _) = compile_template(
            &i,
            r#"{{ x in @items }}{{ x | to_string }}{{/}}"#,
            &[("items", Ty::List(Box::new(Ty::String)))],
        )
        .unwrap();
        run(&mut module.main);
        assert_eq!(count_context_stores(&module.main), 0);
    }

    #[test]
    fn straight_line_write_preserved() {
        let i = Interner::new();
        let (mut module, _) = compile_script(&i, "@x = 42; @x", &[("x", Ty::Int)]).unwrap();
        let stores_before = count_context_stores(&module.main);
        run(&mut module.main);
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
        // Nested loop must not panic in SSA pass (regression: IterStep CFG terminator).
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
}
