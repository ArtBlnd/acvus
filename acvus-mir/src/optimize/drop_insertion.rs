//! Drop Insertion - insert `Drop` instructions for non-Copy values.
//!
//! Runs after all optimizations. Inserts `InstKind::Drop` at the point where
//! a move-only value's live range ends.
//!
//! A value that is a word owns nothing and never needs a Drop; every
//! move-only value (`is_move_only`, RFC-0018) gets one.
//!
//! Two phases:
//!
//! **Phase 1 - Within-block drops**: Walk each block forward. When a value's last
//! use within the block is found and the value is NOT live-out, insert Drop after
//! that instruction.
//!
//! **Phase 2 - Edge drops**: At branch points, a value may be forwarded to one
//! successor but not another. For each edge A->B, if a value is live-out of A but
//! NOT forwarded to B and NOT live-in to B, its life ends on that edge; where the
//! drop then runs is `DropSite`.
//!
//! A register a payload has left owns nothing from there on
//! (`move_check::emptied_by`, the statement this pass shares with the move
//! check), and both phases read it: an option's storage after a take that
//! reached its payload through nothing but options, and an unwrap's source
//! in every variant form.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::loans::Loans;
use crate::analysis::{inst_info, liveness};
use crate::cfg::{Block, BlockIdx, CfgBody, Terminator};
use crate::ir::{Inst, InstKind, Label, ValueId};
use crate::ty::Ty;
use crate::validate::move_check::{emptied_by, is_move_only};

/// Insert Drop instructions for non-Copy values at the end of their live ranges.
pub fn insert_drops(cfg: &mut CfgBody, val_types: &FxHashMap<ValueId, Ty>) {
    let liveness = liveness::analyze(cfg);
    let loans = Loans::build(cfg);

    // Build label -> block index mapping.
    let label_to_block: FxHashMap<Label, usize> = cfg
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label, i))
        .collect();

    // -- Phase 1: within-block drops --------------------------------

    for bi in 0..cfg.blocks.len() {
        let block_idx = BlockIdx(bi);
        let block = &cfg.blocks[bi];

        let mut drops_after: Vec<(usize, ValueId)> = Vec::new();

        for (ii, inst) in block.insts.iter().enumerate() {
            for u in loans.uses_with_storage(&inst.kind) {
                if !liveness.is_live_out(block_idx, u)
                    && is_last_use_in_block(block, ii, u, &loans)
                    && needs_drop(u, val_types)
                    && !ends_ownership(&inst.kind, u, val_types)
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
                    .any(|inst| ends_ownership(&inst.kind, v, val_types));
                let consumed_by_term = is_consumed_by_terminator(&block.terminator, v);

                if consumed_by_inst || consumed_by_term {
                    // Ownership transferred - no Drop needed.
                    continue;
                }

                if term_uses.contains(&v) {
                    // Used by terminator but not consumed (read-only, e.g. cond in JumpIf).
                    // Drop is needed but handled by edge drops or will be
                    // dropped in successor blocks.
                    continue;
                }

                if !is_used_in_block(block, v, &loans) {
                    // Unused def - insert drop right after definition.
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

    // -- Phase 2: edge drops (branch-point) -------------------------

    let preds = PredCounts::of(cfg);
    let mut head_drops: FxHashMap<BlockIdx, Vec<ValueId>> = FxHashMap::default();
    let mut splits: Vec<EdgeSplit> = Vec::new();

    for bi in 0..cfg.blocks.len() {
        let live_out = liveness.live_out.get(bi).cloned().unwrap_or_default();
        let emptied = emptied_in(&cfg.blocks[bi], val_types, &loans);
        let edges = terminator_edges(&cfg.blocks[bi].terminator);

        for edge in edges {
            let Some(&target) = label_to_block.get(&edge.target) else {
                continue;
            };
            let target = BlockIdx(target);
            let dying: Vec<ValueId> = live_out
                .iter()
                .copied()
                .filter(|v| {
                    !edge.forwarded.contains(v)
                        && !emptied.contains(v)
                        && !liveness.is_live_in(target, *v)
                        && needs_drop(*v, val_types)
                })
                .collect();
            if dying.is_empty() {
                continue;
            }
            match DropSite::of(preds.count_of(target)) {
                DropSite::TargetHead => head_drops.entry(target).or_default().extend(dying),
                DropSite::SplitEdge => splits.push(EdgeSplit {
                    from: BlockIdx(bi),
                    edge,
                    dying,
                }),
            }
        }
    }

    for (target, vals) in head_drops {
        let block = &mut cfg.blocks[target.0];
        let mut drop_insts: Vec<Inst> = drop_seq(vals);
        drop_insts.extend(block.insts.drain(..));
        block.insts = drop_insts;
    }

    apply_edge_splits(cfg, splits);
}

/// Where the drops of a dying edge run.
enum DropSite {
    /// The edge is its target's only entry, so the target's head is on that
    /// edge and nowhere else.
    TargetHead,
    /// The target is a join. A drop at its head would also run for a sibling
    /// edge along which the value is live and dropped inside its own block, so
    /// the edge gets a block of its own.
    SplitEdge,
}

impl DropSite {
    fn of(target_predecessors: usize) -> Self {
        match target_predecessors {
            0 | 1 => Self::TargetHead,
            _ => Self::SplitEdge,
        }
    }
}

/// Which of a terminator's outgoing edges this is.
#[derive(Clone, Copy)]
enum EdgeSlot {
    Jump,
    Then,
    Else,
    /// The i-th arm of a `Switch`, by position in `arms` (RFC-0051).
    SwitchArm(usize),
    /// A `Switch`'s `default` edge.
    SwitchDefault,
}

/// One outgoing edge of a block's terminator. `forwarded` keeps the order of
/// the jump's arguments: they are the target block's parameters, by position.
struct OutEdge {
    slot: EdgeSlot,
    target: Label,
    forwarded: Vec<ValueId>,
}

/// One edge that needs a block of its own to hold its drops.
struct EdgeSplit {
    from: BlockIdx,
    edge: OutEdge,
    dying: Vec<ValueId>,
}

/// How many edges enter each block, by block index.
struct PredCounts(Vec<usize>);

impl PredCounts {
    fn of(cfg: &CfgBody) -> Self {
        let mut counts = vec![0usize; cfg.blocks.len()];
        for bi in 0..cfg.blocks.len() {
            for succ in cfg.successors(BlockIdx(bi)) {
                counts[succ.0] += 1;
            }
        }
        Self(counts)
    }

    fn count_of(&self, block: BlockIdx) -> usize {
        self.0[block.0]
    }
}

/// One `Drop` per distinct value, in the order given.
fn drop_seq<I>(vals: I) -> Vec<Inst>
where
    I: IntoIterator<Item = ValueId>,
{
    let mut seen = FxHashSet::default();
    vals.into_iter()
        .filter(|v| seen.insert(*v))
        .map(|v| Inst {
            span: acvus_ast::Span::ZERO,
            kind: InstKind::Drop { src: v },
        })
        .collect()
}

/// Give each split edge a block holding its drops and a jump onward, placed
/// directly after the block it leaves so that no `Fallthrough` changes target.
fn apply_edge_splits(cfg: &mut CfgBody, splits: Vec<EdgeSplit>) {
    if splits.is_empty() {
        return;
    }
    let mut next_label = cfg
        .blocks
        .iter()
        .map(|b| b.label)
        .filter(|l| *l != crate::cfg::ENTRY_LABEL)
        .map(|l| l.0 + 1)
        .max()
        .expect("a split edge names a labelled block, so one exists");

    let mut inserted: FxHashMap<BlockIdx, Vec<Block>> = FxHashMap::default();
    for split in splits {
        let label = Label(next_label);
        next_label += 1;
        inserted.entry(split.from).or_default().push(Block {
            label,
            params: vec![],
            insts: drop_seq(split.dying),
            terminator: Terminator::Jump {
                label: split.edge.target,
                args: split.edge.forwarded,
            },
            merge_of: None,
        });
        retarget(
            &mut cfg.blocks[split.from.0].terminator,
            split.edge.slot,
            label,
        );
    }

    let mut blocks: Vec<Block> = Vec::with_capacity(cfg.blocks.len() + inserted.len());
    for (bi, block) in std::mem::take(&mut cfg.blocks).into_iter().enumerate() {
        blocks.push(block);
        blocks.extend(inserted.remove(&BlockIdx(bi)).unwrap_or_default());
    }
    cfg.label_to_block = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label, BlockIdx(i)))
        .collect();
    cfg.blocks = blocks;
}

/// One edge of a terminator, in place.
struct EdgeRef<'a> {
    label: &'a mut Label,
    args: &'a mut Vec<ValueId>,
}

impl<'a> EdgeRef<'a> {
    fn of(term: &'a mut Terminator, slot: EdgeSlot) -> Self {
        match (term, slot) {
            (Terminator::Jump { label, args }, EdgeSlot::Jump)
            | (
                Terminator::JumpIf {
                    then_label: label,
                    then_args: args,
                    ..
                },
                EdgeSlot::Then,
            )
            | (
                Terminator::JumpIf {
                    else_label: label,
                    else_args: args,
                    ..
                },
                EdgeSlot::Else,
            ) => Self { label, args },
            (Terminator::Switch { arms, .. }, EdgeSlot::SwitchArm(i)) => {
                let (_, label, args) = &mut arms[i];
                Self { label, args }
            }
            (Terminator::Switch { default, .. }, EdgeSlot::SwitchDefault) => {
                let (label, args) = default
                    .as_mut()
                    .expect("a SwitchDefault slot names a Switch that has a default");
                Self { label, args }
            }
            (term, _) => panic!("edge slot does not name an edge of {term:?}"),
        }
    }
}

/// Send one edge to the block that now holds its drops. The split block takes
/// no parameters and passes the arguments on itself, so the edge loses them.
fn retarget(term: &mut Terminator, slot: EdgeSlot, to: Label) {
    let edge = EdgeRef::of(term, slot);
    *edge.label = to;
    edge.args.clear();
}

/// Check if `val` is used after `at_idx` within the block (instructions + terminator).
fn is_last_use_in_block(
    block: &crate::cfg::Block,
    at_idx: usize,
    val: ValueId,
    loans: &Loans,
) -> bool {
    for inst in &block.insts[at_idx + 1..] {
        if loans.uses_with_storage(&inst.kind).contains(&val) {
            return false;
        }
    }
    if terminator_use_set(&block.terminator).contains(&val) {
        return false;
    }
    true
}

/// Check if `val` is used by any instruction or terminator in the block.
fn is_used_in_block(block: &crate::cfg::Block, val: ValueId, loans: &Loans) -> bool {
    for inst in &block.insts {
        if loans.uses_with_storage(&inst.kind).contains(&val) {
            return true;
        }
    }
    terminator_use_set(&block.terminator).contains(&val)
}

/// Extract ValueIds directly used by a terminator (not forwarded args).
fn terminator_use_set(term: &Terminator) -> FxHashSet<ValueId> {
    let mut uses = FxHashSet::default();
    match term {
        Terminator::Return { value, order } => {
            uses.insert(*value);
            uses.extend(order.iter().copied());
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
        // The tag a `Switch` reads, and the arguments each edge forwards.
        Terminator::Switch { tag, arms, default } => {
            uses.insert(*tag);
            for (_, _, args) in arms {
                uses.extend(args.iter().copied());
            }
            if let Some((_, args)) = default {
                uses.extend(args.iter().copied());
            }
        }
        Terminator::Fallthrough | Terminator::Diverge => {}
    }
    uses
}

/// The outgoing edges of a terminator. A `Fallthrough` edge carries no label,
/// so it names no edge here.
fn terminator_edges(term: &Terminator) -> Vec<OutEdge> {
    let edge = |slot: EdgeSlot, target: &Label, args: &Vec<ValueId>| OutEdge {
        slot,
        target: *target,
        forwarded: args.clone(),
    };
    match term {
        Terminator::Jump { label, args } => vec![edge(EdgeSlot::Jump, label, args)],
        Terminator::JumpIf {
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        } => vec![
            edge(EdgeSlot::Then, then_label, then_args),
            edge(EdgeSlot::Else, else_label, else_args),
        ],
        Terminator::Switch { arms, default, .. } => arms
            .iter()
            .enumerate()
            .map(|(i, (_, label, args))| edge(EdgeSlot::SwitchArm(i), label, args))
            .chain(
                default
                    .iter()
                    .map(|(label, args)| edge(EdgeSlot::SwitchDefault, label, args)),
            )
            .collect(),
        Terminator::Return { .. } | Terminator::Fallthrough | Terminator::Diverge => vec![],
    }
}

/// Whether `val` leaves this instruction owning nothing, by either road: the
/// instruction took it, or the payload that was all it held left it.
pub(crate) fn ends_ownership(
    kind: &InstKind,
    val: ValueId,
    val_types: &FxHashMap<ValueId, Ty>,
) -> bool {
    is_consumed_by_inst(kind, val) || emptied_by(kind, val_types) == Some(val)
}

/// The registers a block leaves empty: the payload left them, and nothing
/// wrote to them afterwards.
fn emptied_in(
    block: &Block,
    val_types: &FxHashMap<ValueId, Ty>,
    loans: &Loans,
) -> FxHashSet<ValueId> {
    let mut emptied = FxHashSet::default();
    for inst in &block.insts {
        if let Some(register) = emptied_by(&inst.kind, val_types) {
            emptied.insert(register);
            continue;
        }
        for written in loans.storage_effect(&inst.kind).writes {
            emptied.remove(&written);
        }
    }
    emptied
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
        InstKind::FunctionCall { callee, args, .. } => {
            args.contains(&val) || matches!(callee, crate::ir::Callee::Indirect(f) if *f == val)
        }
        // Spawn consumes args.
        InstKind::Spawn { callee, args, .. } => {
            args.contains(&val) || matches!(callee, crate::ir::Callee::Indirect(f) if *f == val)
        }
        // Eval consumes the Handle.
        InstKind::Eval { src, .. } => *src == val,
        // A whole Take moves the value out of its slot (RFC-0018); a Take
        // of a part leaves the rest for a Drop.
        InstKind::Take { target, path, .. } => {
            path.is_empty() && inst_info::storage(target) == Some(val)
        }
        // Assign and Commit consume the value; the reference an Assign
        // goes through is only read.
        InstKind::Assign { value, .. } | InstKind::Commit { value, .. } => *value == val,
        // Cast consumes src (transforms it).
        // Container constructors consume their elements.
        InstKind::MakeArray { elements, .. } => elements.contains(&val),
        InstKind::StringConcat { parts, .. } => parts.contains(&val),
        InstKind::StringEq { .. } | InstKind::StringClone { .. } => false,
        InstKind::MakeTuple { elements, .. } => elements.contains(&val),
        InstKind::MakeObject { fields, .. } => fields.iter().any(|(_, v)| *v == val),
        InstKind::MakeVariant { payload, .. } => payload.as_ref() == Some(&val),
        // Closure captures are consumed (moved into closure).
        InstKind::MakeClosure { captures, .. } => captures.contains(&val),
        // FieldSet consumes both object and value (produces new object).
        InstKind::FieldSet { object, value, .. } => *object == val || *value == val,
        // Drop consumes src.
        InstKind::Drop { src } => *src == val,
        // An unwrap's source is `move_check::emptied_by`'s answer, which
        // both passes read; `false` here is this function declining to give
        // a second one, not a claim that an unwrap keeps its source.
        InstKind::UnwrapVariant { .. } => false,

        // Read-only: these don't consume the value.
        InstKind::FieldGet { .. }
        | InstKind::BinOp { .. }
        | InstKind::UnaryOp { .. }
        | InstKind::Cast { .. }
        | InstKind::TestLiteral { .. }
        | InstKind::TestVariant { .. }
        | InstKind::TestObjectKey { .. }
        | InstKind::ArrayIndex { .. }
        | InstKind::ObjectGet { .. }
        | InstKind::TupleIndex { .. }
        | InstKind::Merge { .. } => false,

        // A slice borrows its container and an `Index` borrows the slice;
        // only the element written through `IndexSet` changes owner.
        InstKind::AsSlice { .. } | InstKind::Index { .. } => false,
        InstKind::IndexSet { value, .. } => *value == val,

        // These don't consume a value; a Ref only reads the reference a
        // place goes through.
        InstKind::Const { .. }
        | InstKind::Ref { .. }
        | InstKind::Fetch { .. }
        | InstKind::LoadFunction { .. }
        | InstKind::BlockLabel { .. }
        | InstKind::Undef { .. }
        | InstKind::Poison { .. }
        | InstKind::Nop => false,

        // Control flow - handled by terminator, not here.
        InstKind::Jump { .. }
        | InstKind::JumpIf { .. }
        | InstKind::Switch { .. }
        | InstKind::Return { .. }
        | InstKind::Diverge => false,
    }
}

/// Is `val` consumed by the terminator?
fn is_consumed_by_terminator(term: &Terminator, val: ValueId) -> bool {
    match term {
        // Return consumes the value (transferred to caller).
        Terminator::Return { value, order } => *value == val || *order == Some(val),
        // Jump args are transferred to the target block.
        Terminator::Jump { args, .. } => args.contains(&val),
        // JumpIf: args are transferred, cond is read-only.
        Terminator::JumpIf {
            then_args,
            else_args,
            ..
        } => then_args.contains(&val) || else_args.contains(&val),
        // A Switch's edge args are transferred; the tag is read-only.
        Terminator::Switch { arms, default, .. } => {
            arms.iter().any(|(_, _, args)| args.contains(&val))
                || default
                    .as_ref()
                    .is_some_and(|(_, args)| args.contains(&val))
        }
        Terminator::Fallthrough | Terminator::Diverge => false,
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

    /// A user-defined type with an identity: a source of its own, so move-only.
    fn user_defined_ty() -> Ty {
        let i = Interner::new();
        Ty::UserDefined {
            id: QualifiedRef::root(i.intern("TestType")),
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![crate::ty::IdentityTerm::Known(
                <crate::ty::IdentityId as acvus_utils::LocalIdOps>::from_raw(0),
            )],
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
            order_param: None,
            task: crate::ty::Task::Sync,
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

    // -- Copy types: no Drop -----------------------------------------

    #[test]
    fn no_drop_for_copy_types() {
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(42),
                },
                InstKind::Return {
                    value: v(0),
                    order: None,
                },
            ],
            vec![(v(0), Ty::I64)],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 0);
    }

    // -- Simple linear: move-only value used then dropped -------------

    #[test]
    fn drop_after_last_use() {
        // v0 = UserDefined (move-only)
        // v1 = FieldGet(v0, "x")  -> v0's last use
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
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            vec![(v(0), user_defined_ty()), (v(1), Ty::I64)],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 1);
        assert!(drop_targets(&cfg).contains(&v(0)));
    }

    // -- Value returned: no Drop --------------------------------------

    #[test]
    fn no_drop_for_returned_value() {
        let (mut cfg, val_types) = make_cfg_with_types(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(0),
                },
                InstKind::Return {
                    value: v(0),
                    order: None,
                },
            ],
            vec![(v(0), user_defined_ty())],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 0);
    }

    // -- Unused move-only value: dropped immediately ------------------

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
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            vec![(v(0), user_defined_ty()), (v(1), Ty::I64)],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 1);
        assert!(drop_targets(&cfg).contains(&v(0)));
    }

    // -- Branch: value used in one arm, dropped in the other ----------

    #[test]
    fn drop_in_branch_where_not_used() {
        // v0 = UserDefined
        // v1 = Bool (cond)
        // v2 = Int
        // if v1 -> then(v0), else()
        // then: v3 = v0, return v3
        // else: return v2
        // -> v0 should be dropped in else branch (Phase 2 edge drop).
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
                InstKind::Return {
                    value: v(3),
                    order: None,
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            vec![
                (v(0), user_defined_ty()),
                (v(1), Ty::Bool),
                (v(2), Ty::I64),
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

    // -- Both branches get the value: no edge drop --------------------

    #[test]
    fn no_edge_drop_when_forwarded_to_both() {
        // v0 = UserDefined, forwarded to both branches
        // if cond -> then(v0), else(v0)
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
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![v(3)],
                    merge_of: None,
                },
                InstKind::Return {
                    value: v(3),
                    order: None,
                },
            ],
            vec![
                (v(0), user_defined_ty()),
                (v(1), Ty::Bool),
                (v(2), user_defined_ty()),
                (v(3), user_defined_ty()),
            ],
        );

        insert_drops(&mut cfg, &val_types);
        // v0 is forwarded to both branches -> no edge drops.
        // v2, v3 are returned -> no drops.
        assert_eq!(count_drops(&cfg), 0);
    }

    // -- Multiple move-only values, different lifetimes ---------------

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
                InstKind::Return {
                    value: v(3),
                    order: None,
                },
            ],
            vec![
                (v(0), user_defined_ty()),
                (v(1), user_defined_ty()),
                (v(2), Ty::I64),
                (v(3), Ty::I64),
            ],
        );

        insert_drops(&mut cfg, &val_types);
        // v0 dropped after FieldGet(v0), v1 dropped after FieldGet(v1).
        assert_eq!(count_drops(&cfg), 2);
        let targets = drop_targets(&cfg);
        assert!(targets.contains(&v(0)));
        assert!(targets.contains(&v(1)));
    }

    // -- Value used multiple times then dropped ----------------------

    #[test]
    fn drop_after_multiple_uses() {
        // v0 = UserDefined
        // v1 = FieldGet(v0, "x")  - first use
        // v2 = FieldGet(v0, "y")  - last use -> drop here
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
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            vec![(v(0), user_defined_ty()), (v(1), Ty::I64), (v(2), Ty::I64)],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 1);
        // v0 should be dropped after the second FieldGet (its last use).
        let targets = drop_targets(&cfg);
        assert!(targets.contains(&v(0)));
    }

    // -- Container with move-only element: needs drop ----------------

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
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            vec![
                (
                    v(0),
                    Ty::Array(Box::new(user_defined_ty()), crate::ty::LenTerm::Known(3)),
                ), // List<MoveOnly> = move-only
                (v(1), Ty::I64),
            ],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 1);
        assert!(drop_targets(&cfg).contains(&v(0)));
    }

    // -- Container with copy element: no drop ------------------------

    #[test]
    fn list_of_ints_moves_and_is_dropped() {
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
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            vec![
                (
                    v(0),
                    Ty::Array(Box::new(Ty::I64), crate::ty::LenTerm::Known(3)),
                ),
                (v(1), Ty::I64),
            ],
        );

        insert_drops(&mut cfg, &val_types);
        assert_eq!(count_drops(&cfg), 1);
    }

    // -- Both branches drop different values --------------------------

    #[test]
    fn both_branches_drop_different_values() {
        // v0 = MoveOnly, v1 = MoveOnly
        // if cond -> then(v0), else(v1)
        // then: return v0 -> v1 needs drop in then
        // else: return v1 -> v0 needs drop in else
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
                InstKind::Return {
                    value: v(3),
                    order: None,
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![v(4)],
                    merge_of: None,
                },
                InstKind::Return {
                    value: v(4),
                    order: None,
                },
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
        assert!(
            then_drops.contains(&v(1)),
            "v1 should drop in then: {:?}",
            then_drops
        );
        // else block should drop v0 (not forwarded to else).
        let else_drops = block_drop_targets(&cfg, 2);
        assert!(
            else_drops.contains(&v(0)),
            "v0 should drop in else: {:?}",
            else_drops
        );
        // Total: 2 drops.
        assert_eq!(count_drops(&cfg), 2);
    }

    // -- No double drop: value used and dropped only once -------------

    #[test]
    fn no_double_drop() {
        // v0 = MoveOnly, used once -> exactly 1 Drop.
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
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            vec![(v(0), user_defined_ty()), (v(1), Ty::I64)],
        );

        insert_drops(&mut cfg, &val_types);
        let targets = drop_targets(&cfg);
        let v0_drops = targets.iter().filter(|&&t| t == v(0)).count();
        assert_eq!(v0_drops, 1, "v0 should be dropped exactly once");
    }
}
