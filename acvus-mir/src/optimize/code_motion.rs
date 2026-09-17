//! Cross-block code motion: an instruction moves only between
//! control-equivalent blocks - except a shared borrow of a storage, which
//! moves under a borrow condition instead - and an Eval sinks toward its
//! use.
//!
//! Two blocks are control-equivalent when the destination dominates the
//! source and the source post-dominates the destination. They then execute
//! under exactly the same condition, so the move changes nothing the program
//! can observe: no raise on a path that did not reach the instruction, no
//! work on a path that did not need it.
//!
//! Until `bb8207f` the criterion was purity instead, and purity is not
//! infallibility: integer division and remainder panic at zero and at
//! `MIN / -1` (RFC-0037), and at that time `+` was checked too, so hoisting
//! `i + 1` out of a loop body made `let i = 250; while i < @n { i = i + 1; }
//! i` with `n: u8 = 255` overflow where the program returns `255`. The
//! arithmetic now wraps as Rust's release build does; the hoist would
//! still return the wrong value, and `/` still panics off the program's
//! path.
//!
//! A Spawn is never moved: the work starts at the Spawn, and issuing it on
//! a path that would not have reached it speculates an effect (RFC-0007).
//! Its operands need only reach the block that issues it, which the
//! equivalence already permits.
//!
//! # A shared borrow of a storage
//!
//! `ref &v` on a variable or an extern parameter with no path under it is
//! the one instruction that needs no control equivalence. It writes one
//! register with the address of another (`ops::storage::ref_var`): it reads
//! nothing, allocates nothing and cannot raise, so no path can observe that
//! it ran. What it needs instead is a borrow condition. `check_borrows`
//! runs before this pass (`graph/optimize.rs`), so a borrow moved above a
//! loop holds its loan through every iteration of a program the checker
//! only saw with the loan inside the body. The move is therefore taken
//! only when no block the borrow would newly span - the blocks the target
//! dominates that still reach the source, the source included, the target
//! excluded because the instruction lands at its end - writes that storage.
//! A write is what `Loans::storage_effect` calls one: an `Assign`, a `Take`
//! out of a storage, a `Ref &mut`, or a call an argument carries a `Mut`
//! loan into - including one that reaches the storage only through a
//! reference, which `Loans::touch` resolves to the loans that reference's
//! region holds.
//!
//! `Ref &mut`, a `Ref` through a reference and a `Ref` with a path stay
//! where they are: the first takes a loan that conflicts with every other,
//! the second is a memory op that stays in order with the other ops through
//! that reference, and the third walks into the value, which a path that
//! would not have reached it can find in a shape the walk does not expect.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::analysis::domtree::{DomTree, PostDomTree};
use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::cfg::{BlockIdx, CfgBody};
use crate::ir::*;
use crate::optimize::context_ops::{context_read, context_written};
use crate::ty::Mutability;

// -- Entry point ----------------------------------------------------

pub fn run(cfg: &mut CfgBody) {
    // Phase 1: Hoist - move Spawn and pure instructions UP.
    loop {
        if !hoist_pass(cfg) {
            break;
        }
    }

    // Phase 2: Sink - move Eval and blocking instructions DOWN.
    sink_pass(cfg);
}

// -- Single hoist pass ----------------------------------------------

/// One iteration: find hoistable instructions, move them to the highest
/// valid dominator. Returns true if anything moved.
fn hoist_pass(cfg: &mut CfgBody) -> bool {
    if cfg.blocks.len() < 2 {
        return false;
    }

    let domtree = DomTree::build(cfg);
    let postdom = PostDomTree::build(cfg);
    let writes = StorageWrites::of(cfg);
    let mut def_block = build_def_block(cfg);

    // -- Collect hoists ---------------------------------------------
    //
    // def_block is updated after each decision, so an operand chain within
    // one block resolves in a single pass. No hoistable kind writes a
    // storage, so the decisions below stay valid against `writes` while the
    // rest of this pass is being decided.

    let mut hoists: Vec<(usize, usize, usize)> = Vec::new(); // (src_block, inst_idx, tgt_block)

    for (bi, block) in cfg.blocks.iter().enumerate() {
        if domtree.idom(BlockIdx(bi)).is_none() {
            continue;
        }

        let reaches_here = ReachingBlocks::of(cfg, BlockIdx(bi));

        for (i, inst) in block.insts.iter().enumerate() {
            let kind = &inst.kind;

            if inst_info::defs(kind).is_empty() {
                continue;
            }

            let source = BlockIdx(bi);
            let target = match hoistable(kind) {
                Hoistable::No => continue,
                Hoistable::ControlEquivalent => {
                    let uses = inst_info::uses(kind);
                    find_highest_target(source, &uses, &domtree, &def_block, |candidate| {
                        postdom.post_dominates(source, candidate)
                    })
                }
                // The storage is the borrow's operand: it must be defined
                // above the target as any other operand must.
                Hoistable::SharedBorrow { storage } => {
                    let uses: SmallVec<[ValueId; 4]> = inst_info::uses(kind)
                        .iter()
                        .copied()
                        .chain(std::iter::once(storage))
                        .collect();
                    find_highest_target(source, &uses, &domtree, &def_block, |candidate| {
                        !writes.written_in_span(&domtree, candidate, &reaches_here, storage)
                    })
                }
            };

            if let Some(target) = target {
                hoists.push((bi, i, target.0));
                for d in inst_info::defs(kind) {
                    def_block.insert(d, target);
                }
            }
        }
    }

    if hoists.is_empty() {
        return false;
    }

    // -- Apply hoists -----------------------------------------------

    let mut to_move: FxHashMap<usize, Vec<Inst>> = FxHashMap::default();
    for &(src_bi, inst_i, tgt_bi) in &hoists {
        to_move
            .entry(tgt_bi)
            .or_default()
            .push(cfg.blocks[src_bi].insts[inst_i].clone());
    }

    // Nop originals.
    for &(bi, i, _) in &hoists {
        cfg.blocks[bi].insts[i].kind = InstKind::Nop;
    }

    // Append to target blocks (before terminator, which is separate).
    for (tgt_bi, insts) in to_move {
        cfg.blocks[tgt_bi].insts.extend(insts);
    }

    true
}

// -- Def-block map --------------------------------------------------

/// Build ValueId -> BlockIdx mapping: where each value is defined.
fn build_def_block(cfg: &CfgBody) -> FxHashMap<ValueId, BlockIdx> {
    let mut def_block = FxHashMap::default();

    for (bi, block) in cfg.blocks.iter().enumerate() {
        let idx = BlockIdx(bi);
        for &p in &block.params {
            def_block.insert(p, idx);
        }
        for inst in &block.insts {
            for d in inst_info::defs(&inst.kind) {
                def_block.insert(d, idx);
            }
        }
    }

    for v in cfg.entry_defs() {
        def_block.entry(v).or_insert(BlockIdx(0));
    }

    def_block
}

// -- Target finding -------------------------------------------------

/// The highest dominator of `block_idx` where every operand is already
/// available and `accept` holds.
///
/// Availability only shrinks as the walk rises - an ancestor is dominated by
/// strictly fewer definitions than its child - so the first ancestor that
/// lacks an operand ends the walk. `accept` is asked at every candidate, not
/// only at the first: it may hold at a block and fail at its parent, and the
/// walk keeps the highest that held.
fn find_highest_target<A>(
    block_idx: BlockIdx,
    uses: &[ValueId],
    domtree: &DomTree,
    def_block: &FxHashMap<ValueId, BlockIdx>,
    accept: A,
) -> Option<BlockIdx>
where
    A: Fn(BlockIdx) -> bool,
{
    let mut best: Option<BlockIdx> = None;
    let mut candidate = domtree.idom(block_idx)?;

    loop {
        // All operands must be available at candidate's body (before terminator).
        let all_available = uses.iter().all(|u| match def_block.get(u) {
            Some(&def_bi) if def_bi == candidate => true,
            Some(&def_bi) => domtree.dominates(def_bi, candidate),
            None => true,
        });
        if !all_available {
            break;
        }

        if accept(candidate) {
            best = Some(candidate);
        }

        match domtree.idom(candidate) {
            Some(parent) => candidate = parent,
            None => break,
        }
    }

    best
}

// -- The blocks a borrow would newly span ---------------------------

/// The blocks from which one block is reachable, that block itself included:
/// a hoist out of it puts the borrow above everything they hold.
struct ReachingBlocks {
    reaches: Vec<bool>,
}

impl ReachingBlocks {
    fn of(cfg: &CfgBody, block: BlockIdx) -> Self {
        let preds = cfg.predecessors();
        let mut reaches = vec![false; cfg.blocks.len()];
        let mut stack = vec![block];
        reaches[block.0] = true;
        while let Some(b) = stack.pop() {
            let Some(ps) = preds.get(&b) else {
                continue;
            };
            for &p in ps {
                if !reaches[p.0] {
                    reaches[p.0] = true;
                    stack.push(p);
                }
            }
        }
        Self { reaches }
    }

    fn contains(&self, block: BlockIdx) -> bool {
        self.reaches[block.0]
    }
}

/// The storages each block writes (`Loans::storage_effect`).
struct StorageWrites {
    per_block: Vec<SmallVec<[ValueId; 4]>>,
}

impl StorageWrites {
    fn of(cfg: &CfgBody) -> Self {
        let loans = Loans::build(cfg);
        Self {
            per_block: cfg
                .blocks
                .iter()
                .map(|block| {
                    block
                        .insts
                        .iter()
                        .flat_map(|inst| loans.storage_effect(&inst.kind).writes)
                        .collect()
                })
                .collect(),
        }
    }

    /// Does anything write `storage` in the blocks a borrow hoisted from the
    /// source to `target` would newly span?
    ///
    /// Those are the blocks `target` dominates that still reach the source -
    /// `reaches_source` carries the latter. `target` itself is not one: the
    /// instruction lands at the end of it, after everything it holds.
    fn written_in_span(
        &self,
        domtree: &DomTree,
        target: BlockIdx,
        reaches_source: &ReachingBlocks,
        storage: ValueId,
    ) -> bool {
        (0..self.per_block.len())
            .map(BlockIdx)
            .filter(|&b| b != target && reaches_source.contains(b) && domtree.dominates(target, b))
            .any(|b| self.per_block[b.0].contains(&storage))
    }
}

// -- Hoistability (allowlist) ---------------------------------------

/// What a hoist of this instruction requires of its target.
enum Hoistable {
    /// It does not move.
    No,
    /// It moves only to a block control-equivalent to its own: it has no
    /// effect and reads nothing a store between the two blocks could change,
    /// but it may cost work or raise, so the set of paths it runs on must
    /// stay what it was.
    ControlEquivalent,
    /// It is a shared borrow of `storage`, and moves to any dominator no
    /// block between which and it writes `storage` (see the module doc).
    SharedBorrow { storage: ValueId },
}

/// Whether the instruction can raise is deliberately not asked of a
/// `ControlEquivalent` kind. Control equivalence already fixes the set of
/// paths it runs on, so a failability test here would reject moves that
/// change nothing. An unknown kind is not movable.
fn hoistable(kind: &InstKind) -> Hoistable {
    match kind {
        // Arithmetic / logic.
        InstKind::BinOp { .. } | InstKind::UnaryOp { .. } => Hoistable::ControlEquivalent,

        // A word constant; a heap value is built where it is used.
        InstKind::Const { value, .. } => {
            match matches!(
                value,
                acvus_ast::Literal::String(_) | acvus_ast::Literal::List(_)
            ) {
                true => Hoistable::No,
                false => Hoistable::ControlEquivalent,
            }
        }
        InstKind::MakeArray { .. }
        | InstKind::MakeObject { .. }
        | InstKind::MakeTuple { .. }
        | InstKind::MakeVariant { .. }
        | InstKind::MakeClosure { .. } => Hoistable::No,

        // The address of a storage, under no path and lent shared: the one
        // move the borrow condition carries rather than control equivalence.
        InstKind::Ref {
            target: RefTarget::Var(storage) | RefTarget::Param(storage),
            path,
            mutability: Mutability::Shared,
            ..
        } if path.is_empty() => Hoistable::SharedBorrow { storage: *storage },
        InstKind::Ref { .. } => Hoistable::No,

        // Field and element access.
        InstKind::FieldGet { .. }
        | InstKind::FieldSet { .. }
        | InstKind::ObjectGet { .. }
        | InstKind::ArrayIndex { .. }
        | InstKind::TupleIndex { .. } => Hoistable::ControlEquivalent,

        // Test predicates.
        InstKind::TestLiteral { .. }
        | InstKind::TestVariant { .. }
        | InstKind::TestObjectKey { .. } => Hoistable::ControlEquivalent,

        // Function reference.
        InstKind::LoadFunction { .. } => Hoistable::ControlEquivalent,

        _ => Hoistable::No,
    }
}

// -- Sink pass -----------------------------------------------------
//
// Sinking widens the distance between a Spawn and the Eval that awaits it,
// which is the window the executor has to run the call concurrently.

/// Position of a use: (block_idx, instruction_index_within_block or TERMINATOR).
fn terminator_uses_vec(term: &crate::cfg::Terminator) -> Vec<ValueId> {
    use crate::cfg::Terminator;
    match term {
        Terminator::Return { value, order } => std::iter::once(*value).chain(*order).collect(),
        Terminator::Jump { args, .. } => args.clone(),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            let mut v = vec![*cond];
            v.extend_from_slice(then_args);
            v.extend_from_slice(else_args);
            v
        }
        Terminator::Fallthrough | Terminator::Diverge => vec![],
    }
}

// -- Sink infrastructure ---------------------------------------------

/// Move each Eval as late as possible within its block. A page op
/// (`Fetch`, `Commit`) is never moved.
///
/// One Eval per iteration, then re-scan: a move invalidates the indices the
/// rest of the scan holds.
fn sink_pass(cfg: &mut CfgBody) {
    let max_iters = cfg.blocks.iter().map(|b| b.insts.len()).sum::<usize>() * 2;
    let mut iters = 0;
    loop {
        if !sink_one(cfg) {
            break;
        }
        iters += 1;
        if iters > max_iters {
            #[cfg(debug_assertions)]
            eprintln!("[sink_pass] hit max iterations ({max_iters}), stopping");
            break;
        }
    }
}

/// Try to sink ONE Eval. Returns true if something moved.
fn sink_one(cfg: &mut CfgBody) -> bool {
    let loans = Loans::build(cfg);
    for bi in 0..cfg.blocks.len() {
        for ii in 0..cfg.blocks[bi].insts.len() {
            let kind = &cfg.blocks[bi].insts[ii].kind;
            let InstKind::Eval { .. } = kind else {
                continue;
            };
            let effect = loans.storage_effect(kind);

            let defs: Vec<ValueId> = inst_info::defs(kind).to_vec();

            // Scan forward for barrier.
            let mut barrier = cfg.blocks[bi].insts.len();

            for jj in (ii + 1)..cfg.blocks[bi].insts.len() {
                let other = &cfg.blocks[bi].insts[jj].kind;

                let other_uses = inst_info::uses(other);
                if defs.iter().any(|d| other_uses.contains(d)) {
                    barrier = jj;
                    break;
                }

                if is_call(other)
                    || is_page_op(other)
                    || effect.conflicts(&loans.storage_effect(other))
                {
                    barrier = jj;
                    break;
                }
            }

            let term_uses = terminator_uses_vec(&cfg.blocks[bi].terminator);
            if defs.iter().any(|d| term_uses.contains(d)) {
                barrier = barrier.min(cfg.blocks[bi].insts.len());
            }

            let target = if barrier > 0 { barrier - 1 } else { ii };
            if target > ii {
                let inst = cfg.blocks[bi].insts.remove(ii);
                cfg.blocks[bi].insts.insert(target, inst);
                return true;
            }
        }
    }
    false
}

fn is_page_op(kind: &InstKind) -> bool {
    context_read(kind).is_some() || context_written(kind).is_some()
}

fn is_call(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::FunctionCall { .. } | InstKind::Spawn { .. } | InstKind::Eval { .. }
    )
}

// -- Tests ----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{self, CfgBody};
    use crate::graph::QualifiedRef;
    use crate::ty::Ty;
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_cfg(insts: Vec<InstKind>, val_count: usize) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..val_count {
            factory.next();
        }
        cfg::promote(MirBody {
            insts: insts
                .into_iter()
                .map(|kind| Inst {
                    span: acvus_ast::Span::ZERO,
                    kind,
                })
                .collect(),
            val_types: FxHashMap::default(),
            params: Vec::new(),
            captures: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
            order_param: None,
        })
    }

    fn io_fn_type(i: &Interner, name: &str) -> (QualifiedRef, Ty) {
        let qref = QualifiedRef::root(i.intern(name));
        (
            qref,
            Ty::Fn {
                params: vec![],
                ret: Box::new(Ty::I64),
                captures: vec![],
                effect: crate::ty::Effect::OPAQUE.into(),
            },
        )
    }

    fn demoted(c: CfgBody) -> MirBody {
        cfg::demote(c)
    }

    fn kinds(body: &MirBody) -> Vec<&InstKind> {
        body.insts.iter().map(|i| &i.kind).collect()
    }

    #[test]
    fn spawn_stays_below_branch() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");

        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Spawn {
                    dst: v(1),
                    callee: Callee::Direct(qref),
                    callee_ty: ty,
                    args: vec![v(0)],
                    order: None,
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let spawn_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Spawn { .. }))
            .unwrap();
        let jumpif_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::JumpIf { .. }))
            .unwrap();
        assert!(
            spawn_idx > jumpif_idx,
            "spawn stays in its block (spawn {spawn_idx}, branch {jumpif_idx})"
        );
    }

    #[test]
    fn block_param_dependency_prevents_hoist() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");

        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(0),
                    then_args: vec![v(0)],
                    else_label: Label(1),
                    else_args: vec![v(0)],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![v(1)],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![v(1)],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![v(2)],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![v(2)],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![v(3)],
                    merge_of: None,
                },
                InstKind::Spawn {
                    dst: v(4),
                    callee: Callee::Direct(qref),
                    callee_ty: ty,
                    args: vec![v(3)],
                    order: None,
                },
                InstKind::Return {
                    value: v(4),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let spawn_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Spawn { .. }))
            .unwrap();
        let merge_idx = k
            .iter()
            .position(|k| {
                matches!(
                    k,
                    InstKind::BlockLabel {
                        label: Label(2),
                        ..
                    }
                )
            })
            .unwrap();
        assert!(spawn_idx > merge_idx, "spawn should stay in merge block");
    }

    fn ref_of(target: RefTarget, path: Vec<PathSeg>, mutability: Mutability) -> InstKind {
        InstKind::Ref {
            dst: v(1),
            target,
            path,
            mutability,
        }
    }

    #[test]
    fn which_borrow_carries_the_condition_and_which_does_not_move() {
        assert!(matches!(
            hoistable(&ref_of(RefTarget::Var(v(0)), vec![], Mutability::Shared)),
            Hoistable::SharedBorrow { storage } if storage == v(0)
        ));
        assert!(matches!(
            hoistable(&ref_of(RefTarget::Param(v(0)), vec![], Mutability::Shared)),
            Hoistable::SharedBorrow { storage } if storage == v(0)
        ));
        assert!(matches!(
            hoistable(&ref_of(RefTarget::Var(v(0)), vec![], Mutability::Mut)),
            Hoistable::No
        ));
        assert!(matches!(
            hoistable(&ref_of(
                RefTarget::Through(v(0)),
                vec![],
                Mutability::Shared
            )),
            Hoistable::No
        ));
        assert!(matches!(
            hoistable(&ref_of(
                RefTarget::Var(v(0)),
                vec![PathSeg::Payload],
                Mutability::Shared
            )),
            Hoistable::No
        ));
    }

    #[test]
    fn a_heap_constant_is_never_hoisted() {
        assert!(matches!(
            hoistable(&InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::String("a".into()),
            }),
            Hoistable::No
        ));
        assert!(matches!(
            hoistable(&InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(1),
            }),
            Hoistable::ControlEquivalent
        ));
        assert!(matches!(
            hoistable(&InstKind::MakeArray {
                dst: v(1),
                elements: vec![],
            }),
            Hoistable::No
        ));
    }

    #[test]
    fn eval_not_hoisted() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");

        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(qref),
                    callee_ty: ty,
                    args: vec![],
                    order: None,
                },
                InstKind::Const {
                    dst: v(5),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(5),
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    order: None,
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let eval_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Eval { .. }))
            .unwrap();
        let merge_idx = k
            .iter()
            .position(|k| {
                matches!(
                    k,
                    InstKind::BlockLabel {
                        label: Label(2),
                        ..
                    }
                )
            })
            .unwrap();
        assert!(eval_idx > merge_idx, "eval must stay in merge block");
    }

    #[test]
    fn spawn_is_not_hoisted_across_a_branch() {
        let i = Interner::new();
        let (qref1, ty1) = io_fn_type(&i, "io1");
        let (qref2, ty2) = io_fn_type(&i, "io2");

        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(qref1),
                    callee_ty: ty1,
                    args: vec![],
                    order: None,
                },
                InstKind::Const {
                    dst: v(5),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(5),
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Spawn {
                    dst: v(1),
                    callee: Callee::Direct(qref2),
                    callee_ty: ty2,
                    args: vec![],
                    order: None,
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let spawn2_idx = k
            .iter()
            .rposition(|k| matches!(k, InstKind::Spawn { .. }))
            .unwrap();
        let merge_idx = k
            .iter()
            .position(|k| {
                matches!(
                    k,
                    InstKind::BlockLabel {
                        label: Label(2),
                        ..
                    }
                )
            })
            .unwrap();
        assert!(
            spawn2_idx > merge_idx,
            "a Spawn stays in its block: hoisting it would issue the call on a path that skips it"
        );
    }

    #[test]
    fn multi_level_hoist() {
        // B0 -> diamond -> B3 -> diamond -> B6
        // The pure op in B6 hoists directly to B0 in one pass.
        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(3),
                    then_args: vec![],
                    else_label: Label(4),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(3),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(5),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(4),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(5),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(5),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::UnaryOp {
                    dst: v(1),
                    op: acvus_ast::UnaryOp::Not,
                    operand: v(0),
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let op_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::UnaryOp { .. }))
            .unwrap();
        let first_jumpif = k
            .iter()
            .position(|k| matches!(k, InstKind::JumpIf { .. }))
            .unwrap();
        assert!(
            op_idx < first_jumpif,
            "pure op should be hoisted to B0 via highest-target"
        );
    }

    #[test]
    fn pure_instruction_hoisted() {
        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::BinOp {
                    dst: v(1),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(0),
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let binop_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { .. }))
            .unwrap();
        let jumpif_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::JumpIf { .. }))
            .unwrap();
        assert!(
            binop_idx < jumpif_idx,
            "pure BinOp should be hoisted before branch"
        );
    }

    /// The shape the pass exists for: the Spawn's operand rises to the block
    /// that dominates the branch, and the Spawn itself does not follow it.
    #[test]
    fn a_spawn_operand_rises_to_the_dominator_and_the_spawn_stays() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");

        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::BinOp {
                    dst: v(1),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(0),
                },
                InstKind::Spawn {
                    dst: v(2),
                    callee: Callee::Direct(qref),
                    callee_ty: ty,
                    args: vec![v(1)],
                    order: None,
                },
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let binop_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { .. }))
            .unwrap();
        let jumpif_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::JumpIf { .. }))
            .unwrap();
        let spawn_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Spawn { .. }))
            .unwrap();
        assert!(
            binop_idx < jumpif_idx,
            "the operand rises: merge and entry are control-equivalent (operand {binop_idx}, branch {jumpif_idx})"
        );
        assert!(
            jumpif_idx < spawn_idx,
            "the spawn stays where it is issued (branch {jumpif_idx}, spawn {spawn_idx})"
        );
    }

    /// The loop head reaches the exit without the body, so the body is not
    /// control-equivalent to it.
    #[test]
    fn a_loop_body_instruction_does_not_rise_into_the_head() {
        let mut cfg = make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::Jump {
                    label: Label(0),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::JumpIf {
                    cond: v(0),
                    then_label: Label(1),
                    then_args: vec![],
                    else_label: Label(2),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::BinOp {
                    dst: v(1),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(0),
                },
                InstKind::Jump {
                    label: Label(0),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                    merge_of: None,
                },
                InstKind::Return {
                    value: v(0),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let binop_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { .. }))
            .unwrap();
        let jumpif_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::JumpIf { .. }))
            .unwrap();
        assert!(
            jumpif_idx < binop_idx,
            "the body's operation stays below the test (branch {jumpif_idx}, operation {binop_idx})"
        );
    }

    /// `v` is a storage, `v(2)` a `&mut` of it taken above the loop, and the
    /// body holds a shared borrow of `v` and writes it - through `v(2)`, so
    /// the write names `v` only through the reference's region. The shared
    /// borrow stays in the body; with `writes_through_the_reference` false,
    /// the same program hoists it to the entry.
    fn a_loop_that_borrows_a_storage(writes_through_the_reference: bool) -> Vec<InstKind> {
        let mut body = vec![InstKind::Ref {
            dst: v(4),
            target: RefTarget::Var(v(1)),
            path: vec![],
            mutability: Mutability::Shared,
        }];
        if writes_through_the_reference {
            body.push(InstKind::Assign {
                target: RefTarget::Through(v(2)),
                path: vec![],
                value: v(0),
            });
        }

        let mut insts = vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Assign {
                target: RefTarget::Var(v(1)),
                path: vec![],
                value: v(0),
            },
            InstKind::Ref {
                dst: v(2),
                target: RefTarget::Var(v(1)),
                path: vec![],
                mutability: Mutability::Mut,
            },
            InstKind::Const {
                dst: v(3),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::Jump {
                label: Label(0),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
                merge_of: None,
            },
            InstKind::JumpIf {
                cond: v(3),
                then_label: Label(1),
                then_args: vec![],
                else_label: Label(2),
                else_args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
                merge_of: None,
            },
        ];
        insts.extend(body);
        insts.extend([
            InstKind::Jump {
                label: Label(0),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            },
            InstKind::Return {
                value: v(0),
                order: None,
            },
        ]);
        insts
    }

    /// Where the shared borrow of `v(1)` sits, relative to the loop's test.
    fn shared_borrow_is_below_the_test(insts: Vec<InstKind>) -> bool {
        let mut cfg = make_cfg(insts, 10);
        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let borrow = k
            .iter()
            .position(|k| {
                matches!(
                    k,
                    InstKind::Ref {
                        mutability: Mutability::Shared,
                        ..
                    }
                )
            })
            .unwrap();
        let test = k
            .iter()
            .position(|k| matches!(k, InstKind::JumpIf { .. }))
            .unwrap();
        borrow > test
    }

    #[test]
    fn a_borrow_stays_when_a_reference_the_loop_holds_writes_the_storage() {
        assert!(
            shared_borrow_is_below_the_test(a_loop_that_borrows_a_storage(true)),
            "the write through the `&mut` names the storage in its region"
        );
    }

    #[test]
    fn the_same_loop_without_that_write_lets_the_borrow_out() {
        assert!(
            !shared_borrow_is_below_the_test(a_loop_that_borrows_a_storage(false)),
            "nothing the borrow would span writes the storage"
        );
    }

    // -- Sink tests --------------------------------------------------

    /// Eval is sunk past pure computation to just before its result is used.
    ///
    /// Before: Spawn, Eval, BinOp(uses eval result), Return
    /// After:  Spawn, BinOp...(doesn't use eval), Eval, BinOp(uses eval result), Return
    #[test]
    fn eval_sunk_past_independent_computation() {
        let i = Interner::new();
        let (fetch_id, fetch_ty) = io_fn_type(&i, "fetch");

        // v0 = Spawn fetch
        // v1 = Eval v0              <- should sink
        // v2 = BinOp(v3, v3)       <- independent of eval result
        // v4 = BinOp(v1, v2)       <- uses eval result
        // Return v4
        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fetch_id),
                    callee_ty: fetch_ty,
                    args: vec![],
                    order: None,
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    order: None,
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(3),
                    right: v(3),
                },
                InstKind::BinOp {
                    dst: v(4),
                    op: acvus_ast::BinOp::Add,
                    left: v(1),
                    right: v(2),
                },
                InstKind::Return {
                    value: v(4),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);

        let eval_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Eval { .. }))
            .unwrap();
        let use_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { left, .. } if *left == v(1)))
            .unwrap();
        let independent_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { left, .. } if *left == v(3)))
            .unwrap();

        assert!(
            independent_idx < eval_idx,
            "independent computation should be before eval (was {independent_idx}, eval at {eval_idx})"
        );
        assert!(
            eval_idx < use_idx,
            "eval should be before its use (eval at {eval_idx}, use at {use_idx})"
        );
    }

    #[test]
    fn fetch_not_sunk_past_call() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));
        let f = QualifiedRef::root(i.intern("f"));
        let mut cfg = make_cfg(
            vec![
                InstKind::Fetch {
                    dst: v(1),
                    context: ctx,
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(5),
                    right: v(5),
                },
                InstKind::FunctionCall {
                    dst: v(3),
                    callee: Callee::Direct(f),
                    callee_ty: Ty::error(),
                    args: vec![],
                    order: None,
                },
                InstKind::BinOp {
                    dst: v(6),
                    op: acvus_ast::BinOp::Add,
                    left: v(1),
                    right: v(2),
                },
                InstKind::Return {
                    value: v(6),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let load_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Fetch { .. }))
            .unwrap();
        let call_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::FunctionCall { .. }))
            .unwrap();
        assert!(
            load_idx < call_idx,
            "fetch must stay before the call (fetch {load_idx}, call {call_idx})"
        );
    }

    #[test]
    fn eval_not_sunk_past_fetch() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));
        let (qref, ty) = io_fn_type(&i, "io_fn");
        let mut cfg = make_cfg(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(qref),
                    callee_ty: ty,
                    args: vec![],
                    order: None,
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    order: None,
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(5),
                    right: v(5),
                },
                InstKind::Fetch {
                    dst: v(4),
                    context: ctx,
                },
                InstKind::BinOp {
                    dst: v(6),
                    op: acvus_ast::BinOp::Add,
                    left: v(1),
                    right: v(4),
                },
                InstKind::Return {
                    value: v(6),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let eval_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Eval { .. }))
            .unwrap();
        let load_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Fetch { .. }))
            .unwrap();
        assert!(
            eval_idx < load_idx,
            "eval must stay before the fetch (eval {eval_idx}, fetch {load_idx})"
        );
    }

    /// A fetch is not moved, even past independent computation.
    #[test]
    fn fetch_stays_in_place() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));

        // v1 = fetch @x
        // v2 = BinOp(v5, v5)   <- independent
        // commit @x = v5
        // v6 = BinOp(v1, v2)   <- uses the fetched value
        // Return v6
        let mut cfg = make_cfg(
            vec![
                InstKind::Fetch {
                    dst: v(1),
                    context: ctx,
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(5),
                    right: v(5),
                },
                InstKind::Commit {
                    context: ctx,
                    value: v(5),
                },
                InstKind::BinOp {
                    dst: v(6),
                    op: acvus_ast::BinOp::Add,
                    left: v(1),
                    right: v(2),
                },
                InstKind::Return {
                    value: v(6),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);

        let load_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Fetch { .. }))
            .unwrap();
        let store_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Commit { .. }))
            .unwrap();
        let independent_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { left, .. } if *left == v(5)))
            .unwrap();

        assert!(
            load_idx < independent_idx,
            "fetch is not moved (fetch {load_idx}, independent {independent_idx})"
        );
        assert!(
            load_idx < store_idx,
            "fetch stays before the commit of the same context (fetch {load_idx}, commit {store_idx})"
        );
    }

    /// A commit is not moved, even past independent computation.
    #[test]
    fn commit_stays_in_place() {
        let i = Interner::new();
        let ctx = QualifiedRef::root(i.intern("x"));

        // commit @x = v5
        // v2 = BinOp(v5, v5)       <- independent
        // v4 = fetch @x
        // v6 = BinOp(v4, v2)
        // Return v6
        let mut cfg = make_cfg(
            vec![
                InstKind::Commit {
                    context: ctx,
                    value: v(5),
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(5),
                    right: v(5),
                },
                InstKind::Fetch {
                    dst: v(4),
                    context: ctx,
                },
                InstKind::BinOp {
                    dst: v(6),
                    op: acvus_ast::BinOp::Add,
                    left: v(4),
                    right: v(2),
                },
                InstKind::Return {
                    value: v(6),
                    order: None,
                },
            ],
            10,
        );

        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);

        let store_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Commit { .. }))
            .unwrap();
        let load_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::Fetch { .. }))
            .unwrap();
        let independent_idx = k
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { left, .. } if *left == v(5)))
            .unwrap();

        assert!(
            store_idx < independent_idx,
            "commit is not moved (commit {store_idx}, independent {independent_idx})"
        );
        assert!(
            store_idx < load_idx,
            "commit stays before the fetch of the same context (commit {store_idx}, fetch {load_idx})"
        );
    }
}
