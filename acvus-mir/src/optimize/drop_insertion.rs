//! Drop Insertion — insert `Drop` instructions for non-Copy values.
//!
//! Runs after all optimizations (DCE, SROA, etc.). Inserts `InstKind::Drop`
//! at the point where a non-Copy value's live range ends.
//!
//! Copy types (Int, Float, Bool, Byte, Unit, String, Identity) never need Drop.
//! Only move-only values (UserDefined, containers with move-only elements, etc.)
//! get Drop instructions.
//!
//! Two phases:
//!
//! **Phase 1 — Within-block drops**: Walk each block forward. When a value's last
//! use within the block is found and the value is NOT live-out, insert Drop after
//! that instruction.
//!
//! **Phase 2 — Edge drops**: At branch points, a value may be forwarded to one
//! successor but not another. For each edge A→B, if a value is live-out of A but
//! NOT forwarded to B and NOT live-in to B, insert Drop at the start of B.
//! Example: `if cond → then(v0), else()` — v0 needs Drop at start of else.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::{inst_info, liveness};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Inst, InstKind, Label, ValueId};
use crate::ty::Ty;
use crate::validate::move_check::is_move_only;

/// Insert Drop instructions for non-Copy values at the end of their live ranges.
pub fn insert_drops(cfg: &mut CfgBody, val_types: &FxHashMap<ValueId, Ty>) {
    let liveness = liveness::analyze(cfg);

    // Build label → block index mapping.
    let label_to_block: FxHashMap<Label, usize> = cfg
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label, i))
        .collect();

    // ── Phase 1: within-block drops ────────────────────────────────

    for bi in 0..cfg.blocks.len() {
        let block_idx = BlockIdx(bi);
        let block = &cfg.blocks[bi];

        let mut drops_after: Vec<(usize, ValueId)> = Vec::new();

        for (ii, inst) in block.insts.iter().enumerate() {
            for u in inst_info::uses(&inst.kind) {
                if !liveness.is_live_out(block_idx, u)
                    && is_last_use_in_block(block, ii, u)
                    && needs_drop(u, val_types)
                    && !is_consumed_by_inst(&inst.kind, u)
                {
                    drops_after.push((ii, u));
                }
            }
        }

        // Values defined in this block that are never used, or whose last use
        // is the terminator and it consumes them (no Drop needed).
        let term_uses = terminator_use_set(&block.terminator);
        let already_dropped: FxHashSet<ValueId> = drops_after.iter().map(|(_, v)| *v).collect();

        // Collect all defs in this block.
        let mut all_defs: Vec<ValueId> = block.params.clone();
        for inst in &block.insts {
            all_defs.extend(inst_info::defs(&inst.kind));
        }

        for v in all_defs {
            if !liveness.is_live_out(block_idx, v)
                && !already_dropped.contains(&v)
                && needs_drop(v, val_types)
            {
                // Check if consumed by any instruction or terminator.
                let consumed_by_inst = block
                    .insts
                    .iter()
                    .any(|inst| is_consumed_by_inst(&inst.kind, v));
                let consumed_by_term = is_consumed_by_terminator(&block.terminator, v);

                if consumed_by_inst || consumed_by_term {
                    // Ownership transferred — no Drop needed.
                    continue;
                }

                if term_uses.contains(&v) {
                    // Used by terminator but not consumed (read-only, e.g. cond in JumpIf).
                    // Drop is needed but handled by edge drops or will be
                    // dropped in successor blocks.
                    continue;
                }

                if !is_used_in_block(block, v) {
                    // Unused def — insert drop right after definition.
                    let idx = block
                        .insts
                        .iter()
                        .position(|inst| inst_info::defs(&inst.kind).contains(&v))
                        .unwrap_or(0);
                    drops_after.push((idx, v));
                }
                // If used in block but not consumed, it was already handled
                // in the per-instruction loop above.
            }
        }

        // Sort by insertion point (reverse order to preserve indices when inserting).
        drops_after.sort_by(|a, b| b.0.cmp(&a.0));

        let block = &mut cfg.blocks[bi];
        for (after_idx, val) in drops_after {
            let drop_inst = Inst {
                span: acvus_ast::Span::ZERO,
                kind: InstKind::Drop { src: val },
            };
            block.insts.insert(after_idx + 1, drop_inst);
        }
    }

    // ── Phase 2: edge drops (branch-point) ─────────────────────────

    // For each block, examine the terminator's outgoing edges.
    // If a value is live-out of the block but NOT forwarded to a successor
    // and NOT live-in to that successor, insert Drop at the start of that successor.

    // Collect edge drops: (target_block_idx, values_to_drop).
    let mut edge_drops: FxHashMap<usize, Vec<ValueId>> = FxHashMap::default();

    for bi in 0..cfg.blocks.len() {
        let block_idx = BlockIdx(bi);
        let live_out = liveness
            .live_out
            .get(bi)
            .cloned()
            .unwrap_or_default();

        let block = &cfg.blocks[bi];
        let edges = terminator_edges(&block.terminator);

        for (label, forwarded) in edges {
            let Some(&target_bi) = label_to_block.get(&label) else {
                continue;
            };
            let target_block_idx = BlockIdx(target_bi);

            for &v in &live_out {
                if !forwarded.contains(&v)
                    && !liveness.is_live_in(target_block_idx, v)
                    && needs_drop(v, val_types)
                {
                    edge_drops.entry(target_bi).or_default().push(v);
                }
            }
        }
    }

    // Insert edge drops at the start of target blocks.
    for (bi, vals) in edge_drops {
        let block = &mut cfg.blocks[bi];
        // Deduplicate (a value might be orphaned from multiple predecessors,
        // but should only be dropped once).
        let mut seen = FxHashSet::default();
        let mut drop_insts: Vec<Inst> = Vec::new();
        for v in vals {
            if seen.insert(v) {
                drop_insts.push(Inst {
                    span: acvus_ast::Span::ZERO,
                    kind: InstKind::Drop { src: v },
                });
            }
        }
        // Prepend drops at the start of the block.
        drop_insts.extend(block.insts.drain(..));
        block.insts = drop_insts;
    }
}

/// Check if `val` is used after `at_idx` within the block (instructions + terminator).
fn is_last_use_in_block(block: &crate::cfg::Block, at_idx: usize, val: ValueId) -> bool {
    for inst in &block.insts[at_idx + 1..] {
        if inst_info::uses(&inst.kind).contains(&val) {
            return false;
        }
    }
    if terminator_use_set(&block.terminator).contains(&val) {
        return false;
    }
    true
}

/// Check if `val` is used by any instruction or terminator in the block.
fn is_used_in_block(block: &crate::cfg::Block, val: ValueId) -> bool {
    for inst in &block.insts {
        if inst_info::uses(&inst.kind).contains(&val) {
            return true;
        }
    }
    terminator_use_set(&block.terminator).contains(&val)
}

/// Extract ValueIds directly used by a terminator (not forwarded args).
fn terminator_use_set(term: &Terminator) -> FxHashSet<ValueId> {
    let mut uses = FxHashSet::default();
    match term {
        Terminator::Return(val) => {
            uses.insert(*val);
        }
        Terminator::Jump { args, .. } => {
            uses.extend(args.iter().copied());
        }
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            uses.insert(*cond);
            uses.extend(then_args.iter().copied());
            uses.extend(else_args.iter().copied());
        }
        Terminator::ListStep {
            list,
            index_src,
            done_args,
            ..
        } => {
            uses.insert(*list);
            uses.insert(*index_src);
            uses.extend(done_args.iter().copied());
        }
        Terminator::Fallthrough => {}
    }
    uses
}

/// Outgoing edges from a terminator: (target_label, forwarded_values).
fn terminator_edges(term: &Terminator) -> Vec<(Label, FxHashSet<ValueId>)> {
    match term {
        Terminator::Jump { label, args } => {
            vec![(*label, args.iter().copied().collect())]
        }
        Terminator::JumpIf {
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        } => {
            vec![
                (*then_label, then_args.iter().copied().collect()),
                (*else_label, else_args.iter().copied().collect()),
            ]
        }
        Terminator::ListStep {
            done, done_args, ..
        } => {
            // The "continue" edge is the fallthrough to the next block —
            // dst and index_dst are defined by the terminator and available in the next block.
            // The "done" edge forwards done_args.
            vec![(*done, done_args.iter().copied().collect())]
        }
        Terminator::Return(_) | Terminator::Fallthrough => vec![],
    }
}

/// Does this value need a Drop instruction?
fn needs_drop(val: ValueId, val_types: &FxHashMap<ValueId, Ty>) -> bool {
    val_types
        .get(&val)
        .and_then(|ty| is_move_only(ty))
        .unwrap_or(false)
}

/// Is `val` consumed (ownership transferred) by this instruction?
///
/// Consumed = the instruction takes ownership. No Drop needed after.
/// Read = the instruction borrows. Drop still needed if this is the last use.
fn is_consumed_by_inst(kind: &InstKind, val: ValueId) -> bool {
    match kind {
        // Function calls consume all arguments (ownership transfer to callee).
        InstKind::FunctionCall { callee, args, context_uses, .. } => {
            args.contains(&val)
                || context_uses.iter().any(|(_, v)| *v == val)
                || matches!(callee, crate::ir::Callee::Indirect(f) if *f == val)
        }
        // Spawn consumes args.
        InstKind::Spawn { callee, args, context_uses, .. } => {
            args.contains(&val)
                || context_uses.iter().any(|(_, v)| *v == val)
                || matches!(callee, crate::ir::Callee::Indirect(f) if *f == val)
        }
        // Eval consumes the Handle.
        InstKind::Eval { src, .. } => *src == val,
        // Store consumes the value (not the dst Ref).
        InstKind::Store { value, .. } => *value == val,
        // Cast consumes src (transforms it).
        InstKind::Cast { src, .. } => *src == val,
        // Container constructors consume their elements.
        InstKind::MakeDeque { elements, .. } => elements.contains(&val),
        InstKind::MakeTuple { elements, .. } => elements.contains(&val),
        InstKind::MakeObject { fields, .. } => fields.iter().any(|(_, v)| *v == val),
        InstKind::MakeVariant { payload, .. } => payload.as_ref() == Some(&val),
        InstKind::MakeRange { start, end, .. } => *start == val || *end == val,
        // Closure captures are consumed (moved into closure).
        InstKind::MakeClosure { captures, .. } => captures.contains(&val),
        // FieldSet consumes both object and value (produces new object).
        InstKind::FieldSet { object, value, .. } => *object == val || *value == val,
        // Drop consumes src.
        InstKind::Drop { src } => *src == val,

        // Read-only: these don't consume the value.
        InstKind::FieldGet { .. }
        | InstKind::BinOp { .. }
        | InstKind::UnaryOp { .. }
        | InstKind::Load { .. }
        | InstKind::Clone { .. }
        | InstKind::TestLiteral { .. }
        | InstKind::TestVariant { .. }
        | InstKind::UnwrapVariant { .. }
        | InstKind::TestListLen { .. }
        | InstKind::TestObjectKey { .. }
        | InstKind::TestRange { .. }
        | InstKind::ListIndex { .. }
        | InstKind::ListGet { .. }
        | InstKind::ListSlice { .. }
        | InstKind::ObjectGet { .. }
        | InstKind::TupleIndex { .. }
        | InstKind::ListStep { .. } => false,

        // These don't use values at all.
        InstKind::Const { .. }
        | InstKind::Ref { .. }
        | InstKind::LoadFunction { .. }
        | InstKind::BlockLabel { .. }
        | InstKind::Undef { .. }
        | InstKind::Poison { .. }
        | InstKind::Nop => false,

        // Control flow — handled by terminator, not here.
        InstKind::Jump { .. }
        | InstKind::JumpIf { .. }
        | InstKind::Return(_) => false,
    }
}

/// Is `val` consumed by the terminator?
fn is_consumed_by_terminator(term: &Terminator, val: ValueId) -> bool {
    match term {
        // Return consumes the value (transferred to caller).
        Terminator::Return(v) => *v == val,
        // Jump args are transferred to the target block.
        Terminator::Jump { args, .. } => args.contains(&val),
        // JumpIf: args are transferred, cond is read-only.
        Terminator::JumpIf { then_args, else_args, .. } => {
            then_args.contains(&val) || else_args.contains(&val)
        }
        // ListStep: list and index_src are read, done_args are transferred.
        Terminator::ListStep { done_args, .. } => done_args.contains(&val),
        Terminator::Fallthrough => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use crate::ir::*;
    use acvus_utils::{Interner, LocalFactory, LocalIdOps, QualifiedRef};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn user_defined_ty() -> Ty {
        let i = Interner::new();
        Ty::UserDefined {
            id: QualifiedRef::root(i.intern("TestType")),
            type_args: vec![],
            effect_args: vec![],
        }
    }

    fn make_cfg_with_types(
        insts: Vec<InstKind>,
        types: Vec<(ValueId, Ty)>,
    ) -> (CfgBody, FxHashMap<ValueId, Ty>) {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..20 {
            factory.next();
        }
        let val_types: FxHashMap<ValueId, Ty> = types.into_iter().collect();
        let cfg = promote(MirBody {
            insts: insts
                .into_iter()
                .map(|kind| Inst {
                    span: acvus_ast::Span::ZERO,
                    kind,
                })
                .collect(),
            val_types: val_types.clone(),
            params: Vec::new(),
            captures: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        });
        (cfg, val_types)
    }

    /// Count Drop instructions in the entire CfgBody.
    fn count_drops(cfg: &CfgBody) -> usize {
        cfg.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|inst| matches!(inst.kind, InstKind::Drop { .. }))
            .count()
    }

    /// Collect all Drop targets (src ValueIds).
    fn drop_targets(cfg: &CfgBody) -> Vec<ValueId> {
        cfg.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter_map(|inst| match &inst.kind {
                InstKind::Drop { src } => Some(*src),
                _ => None,
            })
            .collect()
    }

    /// Collect Drop targets within a specific block.
    fn block_drop_targets(cfg: &CfgBody, block_idx: usize) -> Vec<ValueId> {
        cfg.blocks[block_idx]
            .insts
            .iter()
            .filter_map(|inst| match &inst.kind {
                InstKind::Drop { src } => Some(*src),
                _ => None,
            })
            .collect()
    }

    // ── Copy types: no Drop ─────────────────────────────────────────

    #[test]
    fn no_drop_for_copy_types() {
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(42),
                },
                InstKind::Return(v(0)),
            ],
            vec![(v(0), Ty::Int)],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 0);
    }

    // ── Simple linear: move-only value used then dropped ─────────────

    #[test]
    fn drop_after_last_use() {
        // v0 = UserDefined (move-only)
        // v1 = FieldGet(v0, "x")  → v0's last use
        // return v1
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::FieldGet {
                    dst: v(1),
                    object: v(0),
                    field: Interner::new().intern("x"),
                    rest: vec![],
                },
                InstKind::Return(v(1)),
            ],
            vec![(v(0), user_defined_ty()), (v(1), Ty::Int)],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 1);
        assert!(drop_targets(&cfg).contains(&v(0)));
    }

    // ── Value returned: no Drop ──────────────────────────────────────

    #[test]
    fn no_drop_for_returned_value() {
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::Return(v(0)),
            ],
            vec![(v(0), user_defined_ty())],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 0);
    }

    // ── Unused move-only value: dropped immediately ──────────────────

    #[test]
    fn drop_unused_move_only() {
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::Const {
                    dst: v(1),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::Return(v(1)),
            ],
            vec![(v(0), user_defined_ty()), (v(1), Ty::Int)],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 1);
        assert!(drop_targets(&cfg).contains(&v(0)));
    }

    // ── Branch: value used in one arm, dropped in the other ──────────

    #[test]
    fn drop_in_branch_where_not_used() {
        // v0 = UserDefined
        // v1 = Bool (cond)
        // v2 = Int
        // if v1 → then(v0), else()
        // then: v3 = v0, return v3
        // else: return v2
        // → v0 should be dropped in else branch (Phase 2 edge drop).
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::Const {
                    dst: v(1),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::Const {
                    dst: v(2),
                    value: acvus_ast::Literal::Int(99),
                },
                InstKind::JumpIf {
                    cond: v(1),
                    then_label: Label(0),
                    then_args: vec![v(0)],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![v(3)],
                    merge_of: None,
                },
                InstKind::Return(v(3)),
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Return(v(2)),
            ],
            vec![
                (v(0), user_defined_ty()),
                (v(1), Ty::Bool),
                (v(2), Ty::Int),
                (v(3), user_defined_ty()),
            ],
        );

        insert_drops(&mut cfg, &val_types);

        // v0 should be dropped in the else block (block index 2).
        let else_drops = block_drop_targets(&cfg, 2);
        assert!(
            else_drops.contains(&v(0)),
            "v0 should be dropped in else branch, got drops: {:?}",
            else_drops
        );

        // v0 should NOT be dropped in the then block (it's forwarded as v3).
        let then_drops = block_drop_targets(&cfg, 1);
        assert!(
            !then_drops.contains(&v(0)),
            "v0 should NOT be dropped in then branch"
        );
    }

    // ── Both branches get the value: no edge drop ────────────────────

    #[test]
    fn no_edge_drop_when_forwarded_to_both() {
        // v0 = UserDefined, forwarded to both branches
        // if cond → then(v0), else(v0)
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::Const {
                    dst: v(1),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(1),
                    then_label: Label(0),
                    then_args: vec![v(0)],
                    else_label: Label(1),
                    else_args: vec![v(0)],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![v(2)],
                    merge_of: None,
                },
                InstKind::Return(v(2)),
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![v(3)],
                    merge_of: None,
                },
                InstKind::Return(v(3)),
            ],
            vec![
                (v(0), user_defined_ty()),
                (v(1), Ty::Bool),
                (v(2), user_defined_ty()),
                (v(3), user_defined_ty()),
            ],
        );

        insert_drops(&mut cfg, &val_types);
        // v0 is forwarded to both branches → no edge drops.
        // v2, v3 are returned → no drops.
        assert_eq!(count_drops(&cfg), 0);
    }

    // ── Multiple move-only values, different lifetimes ───────────────

    #[test]
    fn multiple_move_only_different_lifetimes() {
        // v0 = UserDefined, used at inst 1 only
        // v1 = UserDefined, used at inst 2 only
        // v2 = Int (result of inst 1)
        // v3 = Int (result of inst 2)
        // return v3
        let i = Interner::new();
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::Const {
                    dst: v(1),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::FieldGet {
                    dst: v(2),
                    object: v(0),
                    field: i.intern("x"),
                    rest: vec![],
                },
                InstKind::FieldGet {
                    dst: v(3),
                    object: v(1),
                    field: i.intern("y"),
                    rest: vec![],
                },
                InstKind::Return(v(3)),
            ],
            vec![
                (v(0), user_defined_ty()),
                (v(1), user_defined_ty()),
                (v(2), Ty::Int),
                (v(3), Ty::Int),
            ],
        );

        insert_drops(&mut cfg, &val_types);
        // v0 dropped after FieldGet(v0), v1 dropped after FieldGet(v1).
        assert_eq!(count_drops(&cfg), 2);
        let targets = drop_targets(&cfg);
        assert!(targets.contains(&v(0)));
        assert!(targets.contains(&v(1)));
    }

    // ── Value used multiple times then dropped ──────────────────────

    #[test]
    fn drop_after_multiple_uses() {
        // v0 = UserDefined
        // v1 = FieldGet(v0, "x")  — first use
        // v2 = FieldGet(v0, "y")  — last use → drop here
        // return v2
        let i = Interner::new();
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::FieldGet {
                    dst: v(1),
                    object: v(0),
                    field: i.intern("x"),
                    rest: vec![],
                },
                InstKind::FieldGet {
                    dst: v(2),
                    object: v(0),
                    field: i.intern("y"),
                    rest: vec![],
                },
                InstKind::Return(v(2)),
            ],
            vec![
                (v(0), user_defined_ty()),
                (v(1), Ty::Int),
                (v(2), Ty::Int),
            ],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 1);
        // v0 should be dropped after the second FieldGet (its last use).
        let targets = drop_targets(&cfg);
        assert!(targets.contains(&v(0)));
    }

    // ── Container with move-only element: needs drop ────────────────

    #[test]
    fn container_with_move_only_needs_drop() {
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::Const {
                    dst: v(1),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::Return(v(1)),
            ],
            vec![
                (v(0), Ty::List(Box::new(user_defined_ty()))), // List<MoveOnly> = move-only
                (v(1), Ty::Int),
            ],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 1);
        assert!(drop_targets(&cfg).contains(&v(0)));
    }

    // ── Container with copy element: no drop ────────────────────────

    #[test]
    fn container_with_copy_no_drop() {
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::Const {
                    dst: v(1),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::Return(v(1)),
            ],
            vec![
                (v(0), Ty::List(Box::new(Ty::Int))), // List<Int> = copy
                (v(1), Ty::Int),
            ],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 0);
    }

    // ── Both branches drop different values ──────────────────────────

    #[test]
    fn both_branches_drop_different_values() {
        // v0 = MoveOnly, v1 = MoveOnly
        // if cond → then(v0), else(v1)
        // then: return v0 → v1 needs drop in then
        // else: return v1 → v0 needs drop in else
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::Const {
                    dst: v(1),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::Const {
                    dst: v(2),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(2),
                    then_label: Label(0),
                    then_args: vec![v(0)],
                    else_label: Label(1),
                    else_args: vec![v(1)],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![v(3)],
                    merge_of: None,
                },
                InstKind::Return(v(3)),
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![v(4)],
                    merge_of: None,
                },
                InstKind::Return(v(4)),
            ],
            vec![
                (v(0), user_defined_ty()),
                (v(1), user_defined_ty()),
                (v(2), Ty::Bool),
                (v(3), user_defined_ty()),
                (v(4), user_defined_ty()),
            ],
        );

        insert_drops(&mut cfg, &val_types);
        // then block should drop v1 (not forwarded to then).
        let then_drops = block_drop_targets(&cfg, 1);
        assert!(then_drops.contains(&v(1)), "v1 should drop in then: {:?}", then_drops);
        // else block should drop v0 (not forwarded to else).
        let else_drops = block_drop_targets(&cfg, 2);
        assert!(else_drops.contains(&v(0)), "v0 should drop in else: {:?}", else_drops);
        // Total: 2 drops.
        assert_eq!(count_drops(&cfg), 2);
    }

    // ── No double drop: value used and dropped only once ─────────────

    #[test]
    fn no_double_drop() {
        // v0 = MoveOnly, used once → exactly 1 Drop.
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::FieldGet {
                    dst: v(1),
                    object: v(0),
                    field: Interner::new().intern("x"),
                    rest: vec![],
                },
                InstKind::Return(v(1)),
            ],
            vec![(v(0), user_defined_ty()), (v(1), Ty::Int)],
        );

        insert_drops(&mut cfg, &val_types);
        let targets = drop_targets(&cfg);
        let v0_drops = targets.iter().filter(|&&t| t == v(0)).count();
        assert_eq!(v0_drops, 1, "v0 should be dropped exactly once");
    }
}
