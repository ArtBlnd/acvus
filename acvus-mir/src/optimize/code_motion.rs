//! Cross-block code motion: an instruction moves only between
//! control-equivalent blocks - except a shared borrow of a storage, which
//! moves under a borrow condition instead - and an Eval sinks toward its
//! use. A shared borrow that lands beside one of its own kind is then
//! merged into it, under the same borrow condition read within one block.
//!
//! Two blocks are control-equivalent when the destination dominates the
//! source and the source post-dominates the destination. They then execute
//! under exactly the same condition, so the move changes nothing the program
//! can observe: no raise on a path that did not reach the instruction, no
//! work on a path that did not need it.
//!
//! Equivalence is "executes iff", not "executes as often". A loop's exit
//! post-dominates its header, so post-dominance alone lets an instruction
//! written after a loop move into the header and run once per iteration.
//! Every target is therefore also held to the source's loop depth, the
//! number of natural loops containing the block, so that no move - a shared
//! borrow's included - lands deeper in the loop nest than it started. What
//! a natural loop is, and what a value defined outside one is called, lives
//! in `analysis::loops`, which `optimize::lsr` reads from as well.
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
//! out of a storage, or a call an argument carries a `Mut` loan into -
//! including one that reaches the storage only through a reference, which
//! `Loans::touch` resolves to the loans that reference's region holds. A
//! `Ref &mut` is not one of them: it writes nothing, and `storage_effect`
//! reads it as a read. A value live in such a block that holds a `&mut`
//! loan on the storage bars the move as a write does.
//!
//! `Ref &mut`, a `Ref` through a reference and a `Ref` with a path stay
//! where they are: the first takes a loan that conflicts with every other,
//! the second is a memory op that stays in order with the other ops through
//! that reference, and the third walks into the value, which a path that
//! would not have reached it can find in a shape the walk does not expect.
//!
//! # A second shared borrow in one block
//!
//! Two shared borrows of one storage in one block name the same address
//! when they borrow the same thing - the same kind of instruction, reading
//! the same values, at the same type (`BorrowKey`) - and nothing between
//! them takes the storage exclusively: the second is the first, and its
//! uses read the first's value. An `as_slice` of `m[z]` and one of `m[one]`
//! both reach the storage `m` and borrow two different elements of it, so
//! the values each reads are part of what it is a borrow of. A block is a
//! straight line, so this needs no dominance and no reachability - only a
//! walk over the block, dropping a storage's borrow at every instruction
//! that takes it exclusively (`taken_exclusively`: the writes above, plus a
//! `&mut` borrow, which takes the storage for as long as the reference it
//! makes lives even though it writes nothing).
//!
//! The merge runs after the hoist, not before it. In the attention kernel
//! all four pairs are the hoist's work - each second borrow stood inside a
//! loop until the hoist lifted it into the block that already held the
//! first - so a merge before the hoist would find nothing.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::analysis::domtree::{DomTree, PostDomTree};
use crate::analysis::loops::{LoopDepth, NaturalLoop, natural_loops_innermost_first};
use std::mem::{Discriminant, discriminant};

use crate::analysis::inst_info;
use crate::analysis::loans::{Loan, Loans};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::*;
use crate::optimize::const_dedup::{remap_uses, remap_val, remap_vec};
use crate::optimize::context_ops::{context_read, context_written};
use crate::ty::{Mutability, Ty};

// -- Entry point ----------------------------------------------------

pub fn run(cfg: &mut CfgBody) {
    // Phase 1: the hoist, the merge and the dedup to a fixed point. Each
    // is the others' opportunity, in both directions. In the attention
    // kernel every one of the merge's four pairs is the hoist's work -
    // each second borrow stood inside a loop until the hoist lifted it
    // into the block that already held the first. In the log kernel the
    // traffic runs the other way: `a[li]` is indexed off a slice of `a`
    // that the hoist has to lift out of both blocks and the merge to make
    // one before the two `Index` name the same thing, and only then does
    // the dedup leave the inner one's `AsSlice` an operand from outside
    // the loop for the next hoist to lift.
    loop {
        let mut moved = hoist_pass(cfg);
        moved |= merge_pass(cfg);
        moved |= dedup_pass(cfg);
        moved |= coalesce_pass(cfg);
        if !moved {
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
    let depth = LoopDepth::of(cfg, &domtree);
    let loans = Loans::build(cfg);
    let writes = StorageWrites::of(cfg, &loans);
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
            let target = match hoistable(&loans, kind) {
                Hoistable::No => continue,
                Hoistable::ControlEquivalent => {
                    let uses = inst_info::uses(kind);
                    find_highest_target(source, &uses, &domtree, &depth, &def_block, |candidate| {
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
                    find_highest_target(source, &uses, &domtree, &depth, &def_block, |candidate| {
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

// -- An address already named ---------------------------------------

/// A checked `Index` never moves: it may panic, and RFC-0007 keeps an
/// instruction that can raise on the path that wrote it. So an `AsSlice`
/// of `a[i]` stays inside a loop for as long as the `Index` naming `a[i]`
/// does, however invariant in that loop the pair is. This pass is the way
/// out that does not move it: where a dominating block already indexes the
/// same element, the inner `Index` becomes the outer one, which takes
/// nothing off the path it was written on - the earlier instruction ran
/// before it, and panicked there or not at all.
fn names_an_address(kind: &InstKind) -> bool {
    match kind {
        InstKind::Index {
            mode: IndexMode::Ref,
            ..
        } => true,
        InstKind::Ref {
            target: RefTarget::Through(_),
            path,
            mutability: Mutability::Shared,
            ..
        } => path.is_empty(),
        _ => false,
    }
}

fn names_the_same(a: &InstKind, b: &InstKind) -> bool {
    match (a, b) {
        (
            InstKind::Index {
                slice: sa,
                index: ia,
                mode: ma,
                ..
            },
            InstKind::Index {
                slice: sb,
                index: ib,
                mode: mb,
                ..
            },
        ) => sa == sb && ia == ib && ma == mb,
        (
            InstKind::Ref {
                target: ta,
                path: pa,
                mutability: ma,
                ..
            },
            InstKind::Ref {
                target: tb,
                path: pb,
                mutability: mb,
                ..
            },
        ) => ta == tb && pa == pb && ma == mb,
        _ => false,
    }
}

fn backing_storages(loans: &Loans, kind: &InstKind) -> SmallVec<[ValueId; 2]> {
    let through = match kind {
        InstKind::Index { slice, .. } => *slice,
        InstKind::Ref {
            target: RefTarget::Through(reference),
            ..
        } => *reference,
        _ => return SmallVec::new(),
    };
    loans
        .names(through)
        .iter()
        .filter_map(|loan| loan.storage.slot())
        .collect()
}

/// Where one instruction stands in a body.
#[derive(Clone, Copy, PartialEq, Eq)]
struct At {
    block: BlockIdx,
    inst: usize,
}

impl At {
    fn kind<'a>(&self, cfg: &'a CfgBody) -> &'a InstKind {
        &cfg.blocks[self.block.0].insts[self.inst].kind
    }

    /// Whether this instruction runs before `later` on every path that
    /// reaches it.
    fn runs_before(&self, later: At, domtree: &DomTree) -> bool {
        match self.block == later.block {
            true => self.inst < later.inst,
            false => domtree.dominates(self.block, later.block),
        }
    }
}

/// One iteration: every address-naming instruction a dominating one already
/// named becomes that one. Returns true if anything was replaced, so that a
/// chain - the `Index`, then the `Ref` through it - resolves a link a pass.
fn dedup_pass(cfg: &mut CfgBody) -> bool {
    let domtree = DomTree::build(cfg);
    let loans = Loans::build(cfg);
    let writes = StorageWrites::of(cfg, &loans);

    let named: Vec<At> = cfg
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(bi, block)| {
            block
                .insts
                .iter()
                .enumerate()
                .filter(|(_, inst)| names_an_address(&inst.kind))
                .map(move |(inst, _)| At {
                    block: BlockIdx(bi),
                    inst,
                })
        })
        .collect();

    let mut merged: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    let mut replaced: Vec<At> = Vec::new();

    for &at in &named {
        let kind = at.kind(cfg);
        let reaches_here = ReachingBlocks::of(cfg, at.block);
        let storages = backing_storages(&loans, kind);

        let earlier = named
            .iter()
            .copied()
            .filter(|other| {
                *other != at && !replaced.contains(other) && other.runs_before(at, &domtree)
            })
            .find(|other| {
                names_the_same(other.kind(cfg), kind)
                    && !storages.iter().any(|&storage| {
                        writes.written_in_closed_span(&domtree, other.block, &reaches_here, storage)
                    })
            });

        let Some(earlier) = earlier else {
            continue;
        };
        let defines = |kind: &InstKind| {
            *inst_info::defs(kind)
                .first()
                .expect("an address-naming instruction defines its destination")
        };
        merged.insert(defines(kind), defines(earlier.kind(cfg)));
        replaced.push(at);
    }

    if merged.is_empty() {
        return false;
    }

    for at in &replaced {
        cfg.blocks[at.block.0].insts[at.inst].kind = InstKind::Nop;
    }
    for block in cfg.blocks.iter_mut() {
        for inst in block.insts.iter_mut() {
            remap_uses(&mut inst.kind, &merged);
        }
        remap_terminator(&mut block.terminator, &merged);
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

/// The highest dominator of `source` where every operand is already
/// available, the loop nest is no deeper than at `source`, and `accept`
/// holds.
///
/// Availability only shrinks as the walk rises - an ancestor is dominated by
/// strictly fewer definitions than its child - so the first ancestor that
/// lacks an operand ends the walk. The depth clause and `accept` are asked at
/// every candidate, not only at the first: either may hold at a block and
/// fail at its parent, and the walk keeps the highest that held.
fn find_highest_target<A>(
    source: BlockIdx,
    uses: &[ValueId],
    domtree: &DomTree,
    depth: &LoopDepth,
    def_block: &FxHashMap<ValueId, BlockIdx>,
    accept: A,
) -> Option<BlockIdx>
where
    A: Fn(BlockIdx) -> bool,
{
    let mut best: Option<BlockIdx> = None;
    let mut candidate = domtree.idom(source)?;

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

        if depth.at(candidate) <= depth.at(source) && accept(candidate) {
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

/// The storages each block writes (`Loans::storage_effect`), and the ones a
/// value live in it holds a `&mut` loan on: a shared borrow does not live
/// across either (RFC-0018 rule 8).
struct StorageWrites {
    per_block: Vec<SmallVec<[ValueId; 4]>>,
}

impl StorageWrites {
    fn of(cfg: &CfgBody, loans: &Loans) -> Self {
        let live = crate::analysis::liveness::analyze_with(cfg, loans);
        let held_mutably = |value: ValueId| {
            loans
                .holds(value)
                .filter(|loan| loan.mutability == Mutability::Mut)
                .filter_map(|loan| loan.storage.slot())
                .collect::<SmallVec<[ValueId; 2]>>()
        };
        Self {
            per_block: cfg
                .blocks
                .iter()
                .enumerate()
                .map(|(at, block)| {
                    let live = live.live_in[at].iter().copied().chain(
                        block
                            .insts
                            .iter()
                            .flat_map(|inst| inst_info::uses(&inst.kind)),
                    );
                    block
                        .insts
                        .iter()
                        .flat_map(|inst| loans.storage_effect(&inst.kind).writes)
                        .chain(live.flat_map(held_mutably))
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

    /// The same question between two instructions rather than above one,
    /// with the block each stands in counted whole.
    ///
    /// Neither end block is read instruction by instruction - not an
    /// omission, a decision. A cycle inside the span runs either of them
    /// again, so a write below the later instruction in its own block, or
    /// above the earlier one in its own, still reaches the span; the
    /// position would have to be paired with a second reachability walk to
    /// be worth anything, and the callers ask about a storage a loop does
    /// not write at all.
    fn written_in_closed_span(
        &self,
        domtree: &DomTree,
        earlier: BlockIdx,
        reaches_later: &ReachingBlocks,
        storage: ValueId,
    ) -> bool {
        (0..self.per_block.len())
            .map(BlockIdx)
            .filter(|&b| reaches_later.contains(b) && domtree.dominates(earlier, b))
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
fn hoistable(loans: &Loans, kind: &InstKind) -> Hoistable {
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

        // A shared `AsSlice` is the same borrow one level down: pure,
        // infallible, and a projection of the storage its container names
        // (RFC-0047 rule 3). The container is a reference value, so the storage
        // behind it is the one its region holds; a container whose region
        // holds anything but one storage is not named, and does not move.
        // An exclusive take does not move at all.
        InstKind::AsSlice {
            container,
            mutability: Mutability::Shared,
            ..
        } => match loans.names(*container) {
            [
                Loan {
                    storage,
                    mutability: Mutability::Shared,
                },
            ] => match storage.slot() {
                Some(storage) => Hoistable::SharedBorrow { storage },
                None => Hoistable::No,
            },
            _ => Hoistable::No,
        },
        InstKind::AsSlice { .. } => Hoistable::No,

        // A checked `Index` may panic (RFC-0007), and an `IndexSet` writes
        // the slice's storage.
        InstKind::Index { .. } | InstKind::IndexSet { .. } => Hoistable::No,

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

// -- One slice per container and per loop ----------------------------

/// A slice of a storage taken inside a loop: the `Ref` that names the
/// storage, and the `AsSlice` that projects it.
#[derive(Clone, Copy)]
struct SliceOfStorage {
    reference: At,
    slice: At,
    mutability: Mutability,
}

/// RFC-0047 rule 2 is why one exclusive slice can serve the reads as well: a
/// slice is a pointer and a length, and an element write leaves both where
/// they were. What moves them is a write to the container's shape - a
/// `push`, a `pop`, an `Assign` of a whole new container - and the loans
/// call that a write of the storage while an `IndexSet` through the slice
/// is a write of the storage too. The two are told apart here by which
/// instruction made the write rather than by a second predicate: every
/// touch of the storage inside the loop has to be one of the slices
/// collected below, the `Ref` under one, or an `Index`/`IndexSet` through
/// one, and anything else disqualifies the storage.
///
/// Splitting the mutabilities instead would put a live `&[T]` across an
/// `as_slice_mut` of its own container, which `validate::borrow_check`
/// refuses - `conflicts(Shared, Touch::Reference(Mut))` - so the read
/// slice and the write slice become one or neither leaves.
fn coalesce_pass(cfg: &mut CfgBody) -> bool {
    let domtree = DomTree::build(cfg);
    let mut loops = natural_loops_innermost_first(cfg, &domtree);
    loops.reverse();

    for loop_ in &loops {
        let [preheader] = loop_.entering[..] else {
            continue;
        };
        let taken = slices_taken_in(cfg, loop_);
        for (storage, pairs) in taken {
            if !pairs.iter().any(|p| p.mutability == Mutability::Mut) {
                continue;
            }
            if !every_touch_goes_through(cfg, loop_, storage, &pairs) {
                continue;
            }
            coalesce(cfg, preheader, &pairs);
            return true;
        }
    }
    false
}

/// Every `AsSlice` inside the loop whose container is a `Ref` to a storage
/// taken inside the loop for that `AsSlice` alone, by the storage.
fn slices_taken_in(cfg: &CfgBody, loop_: &NaturalLoop) -> Vec<(ValueId, Vec<SliceOfStorage>)> {
    let readers = use_counts(cfg);
    let mut by_storage: Vec<(ValueId, Vec<SliceOfStorage>)> = Vec::new();

    for block in loop_.blocks() {
        for (ii, inst) in cfg.blocks[block.0].insts.iter().enumerate() {
            let InstKind::AsSlice {
                container,
                mutability,
                ..
            } = inst.kind
            else {
                continue;
            };
            let Some(reference) = definition_in(cfg, loop_, container) else {
                continue;
            };
            let InstKind::Ref {
                target: RefTarget::Var(storage) | RefTarget::Param(storage),
                path,
                ..
            } = &reference.kind(cfg)
            else {
                continue;
            };
            let readers = *readers
                .get(&container)
                .expect("the AsSlice at hand is itself a use of its container");
            if !path.is_empty() || readers != 1 {
                continue;
            }
            let pair = SliceOfStorage {
                reference,
                slice: At { block, inst: ii },
                mutability,
            };
            match by_storage.iter_mut().find(|(s, _)| s == storage) {
                Some((_, pairs)) => pairs.push(pair),
                None => by_storage.push((*storage, vec![pair])),
            }
        }
    }
    by_storage
}

fn use_counts(cfg: &CfgBody) -> FxHashMap<ValueId, usize> {
    let mut counts: FxHashMap<ValueId, usize> = FxHashMap::default();
    for block in &cfg.blocks {
        let uses = block
            .insts
            .iter()
            .flat_map(|inst| inst_info::uses(&inst.kind).to_vec())
            .chain(terminator_uses_vec(&block.terminator));
        for u in uses {
            *counts.entry(u).or_default() += 1;
        }
    }
    counts
}

fn definition_in(cfg: &CfgBody, loop_: &NaturalLoop, value: ValueId) -> Option<At> {
    loop_.blocks().find_map(|block| {
        cfg.blocks[block.0]
            .insts
            .iter()
            .position(|inst| inst_info::defs(&inst.kind).contains(&value))
            .map(|inst| At { block, inst })
    })
}

/// Whether the loop reaches the storage only through the slices collected
/// of it: the `Ref` under one, the `AsSlice` itself, or an `Index` /
/// `IndexSet` of one. Anything else - a call taking the container, an
/// `Assign` of a new one, a `push` - and the slice the loop would carry
/// may no longer name the container's elements.
fn every_touch_goes_through(
    cfg: &CfgBody,
    loop_: &NaturalLoop,
    storage: ValueId,
    pairs: &[SliceOfStorage],
) -> bool {
    let loans = Loans::build(cfg);
    let slices: Vec<ValueId> = pairs
        .iter()
        .map(|p| {
            *inst_info::defs(p.slice.kind(cfg))
                .first()
                .expect("an AsSlice defines its destination")
        })
        .collect();

    loop_.blocks().all(|block| {
        cfg.blocks[block.0]
            .insts
            .iter()
            .enumerate()
            .all(|(inst, held)| {
                let at = At { block, inst };
                if pairs.iter().any(|p| p.reference == at || p.slice == at) {
                    return true;
                }
                let through = match &held.kind {
                    InstKind::Index { slice, .. } | InstKind::IndexSet { slice, .. } => {
                        slices.contains(slice)
                    }
                    _ => false,
                };
                if through {
                    return true;
                }
                let effect = loans.storage_effect(&held.kind);
                !effect.reads.contains(&storage) && !effect.writes.contains(&storage)
            })
    })
}

/// Put one exclusive slice of the storage in the preheader and let every
/// read and write of the loop go through it.
fn coalesce(cfg: &mut CfgBody, preheader: BlockIdx, pairs: &[SliceOfStorage]) {
    let template = pairs
        .iter()
        .find(|p| p.mutability == Mutability::Mut)
        .expect("a coalesced storage is written through one of its slices");

    let mut reference =
        cfg.blocks[template.reference.block.0].insts[template.reference.inst].clone();
    let mut slice = cfg.blocks[template.slice.block.0].insts[template.slice.inst].clone();
    let old_reference = *inst_info::defs(&reference.kind)
        .first()
        .expect("a Ref defines its destination");
    let old_slice = *inst_info::defs(&slice.kind)
        .first()
        .expect("an AsSlice defines its destination");
    let new_reference = cfg.val_factory.next();
    let new_slice = cfg.val_factory.next();
    let ty_of = |v: ValueId, cfg: &CfgBody| {
        cfg.val_types
            .get(&v)
            .cloned()
            .expect("the pipeline types every defined value")
    };
    let reference_ty = ty_of(old_reference, cfg);
    let slice_ty = ty_of(old_slice, cfg);
    cfg.val_types.insert(new_reference, reference_ty);
    cfg.val_types.insert(new_slice, slice_ty);

    let renamed: FxHashMap<ValueId, ValueId> =
        [(old_reference, new_reference), (old_slice, new_slice)]
            .into_iter()
            .collect();
    remap_defs(&mut reference.kind, &renamed);
    remap_defs(&mut slice.kind, &renamed);
    remap_uses(&mut slice.kind, &renamed);
    cfg.blocks[preheader.0].insts.extend([reference, slice]);

    let mut merged: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    for pair in pairs {
        let dst = *inst_info::defs(pair.slice.kind(cfg))
            .first()
            .expect("an AsSlice defines its destination");
        merged.insert(dst, new_slice);
        cfg.blocks[pair.reference.block.0].insts[pair.reference.inst].kind = InstKind::Nop;
        cfg.blocks[pair.slice.block.0].insts[pair.slice.inst].kind = InstKind::Nop;
    }
    for block in cfg.blocks.iter_mut() {
        for inst in block.insts.iter_mut() {
            remap_uses(&mut inst.kind, &merged);
        }
        remap_terminator(&mut block.terminator, &merged);
    }
}

fn remap_defs(kind: &mut InstKind, remap: &FxHashMap<ValueId, ValueId>) {
    match kind {
        InstKind::Ref { dst, .. } | InstKind::AsSlice { dst, .. } => remap_val(dst, remap),
        other => panic!("a coalesced slice is a Ref or an AsSlice, not {other:?}"),
    }
}

// -- Merge pass -----------------------------------------------------

/// What a shared borrow borrows, under a storage: the kind of instruction
/// that took it, every value that instruction reads, and the type of the
/// reference it makes. Two borrows of one storage name one address only
/// when all three agree.
///
/// `operands` is what tells `as_slice(m[z])` from `as_slice(m[one])`: both
/// reach the storage `m`, in the same kind, at the same type, and they
/// name two different elements of it. A `ref &v` reads no value, so its
/// operands are empty and two of them are one borrow as before.
#[derive(PartialEq)]
struct BorrowKey<'a> {
    taken_by: Discriminant<InstKind>,
    operands: SmallVec<[ValueId; 2]>,
    ty: Option<&'a Ty>,
}

impl<'a> BorrowKey<'a> {
    fn of(kind: &InstKind, dst: ValueId, val_types: &'a FxHashMap<ValueId, Ty>) -> Self {
        Self {
            taken_by: discriminant(kind),
            operands: inst_info::uses(kind).iter().copied().collect(),
            ty: val_types.get(&dst),
        }
    }
}

/// Within one block, a shared borrow of a storage becomes the borrow an
/// earlier instruction of that block already took of the same thing, unless
/// something between the two takes the storage exclusively.
///
/// A block is a straight line, so the instructions between the two borrows
/// are exactly what runs between them: the question needs no dominance and
/// no reachability, only a walk.
fn merge_pass(cfg: &mut CfgBody) -> bool {
    let loans = Loans::build(cfg);
    let CfgBody {
        blocks, val_types, ..
    } = cfg;
    let mut merged: FxHashMap<ValueId, ValueId> = FxHashMap::default();

    for block in blocks.iter_mut() {
        let mut borrows_of: FxHashMap<ValueId, Vec<(BorrowKey<'_>, ValueId)>> =
            FxHashMap::default();

        for inst in block.insts.iter_mut() {
            for storage in taken_exclusively(&loans, &inst.kind) {
                borrows_of.remove(&storage);
            }

            // The instruction the hoist calls `SharedBorrow`, asked of the
            // same classifier so that the two rules cannot drift apart.
            let Hoistable::SharedBorrow { storage } = hoistable(&loans, &inst.kind) else {
                continue;
            };
            let dst = *inst_info::defs(&inst.kind)
                .first()
                .expect("a shared borrow defines its destination");

            let key = BorrowKey::of(&inst.kind, dst, val_types);
            let held = borrows_of.entry(storage).or_default();
            let earlier = held
                .iter()
                .find(|(held_key, _)| *held_key == key)
                .map(|(_, earlier)| *earlier);

            match earlier {
                Some(earlier) => {
                    merged.insert(dst, earlier);
                    inst.kind = InstKind::Nop;
                }
                None => held.push((key, dst)),
            }
        }
    }

    if merged.is_empty() {
        return false;
    }

    // A merged value is read wherever the block it was defined in dominates,
    // and the value it becomes is defined earlier in that same block, so
    // every one of those reads is still dominated by a definition.
    for block in cfg.blocks.iter_mut() {
        for inst in block.insts.iter_mut() {
            remap_uses(&mut inst.kind, &merged);
        }
        remap_terminator(&mut block.terminator, &merged);
    }
    true
}

/// The storages an instruction takes exclusively: a shared borrow of one of
/// them does not survive it.
///
/// `Loans::storage_effect` names the writes - an `Assign`, a `Take` out of a
/// storage, a call an argument carries a `Mut` loan into, including one that
/// reaches the storage only through a reference. A `Ref &mut` is not among
/// them: it writes nothing, and `storage_effect` calls it a read. It takes
/// the storage all the same, for as long as the reference it makes lives,
/// and `validate::borrow_check` refuses a shared loan held across it
/// (`conflicts(Shared, Touch::Reference(Mut))`). So the exclusive takes are
/// the writes and the `&mut` borrows together.
fn taken_exclusively(loans: &Loans, kind: &InstKind) -> SmallVec<[ValueId; 2]> {
    let mut taken = loans.storage_effect(kind).writes;
    if let InstKind::Ref {
        target,
        mutability: Mutability::Mut,
        ..
    } = kind
    {
        // An exclusive `AsSlice` reaches here as a write, through
        // `storage_effect`; a `Ref &mut` writes nothing and does not.
        match target {
            RefTarget::Var(storage) | RefTarget::Param(storage) => taken.push(*storage),
            RefTarget::Through(reference) => {
                taken.extend(
                    loans
                        .names(*reference)
                        .iter()
                        .filter_map(|l| l.storage.slot()),
                );
            }
        }
    }
    taken
}

fn remap_terminator(term: &mut Terminator, remap: &FxHashMap<ValueId, ValueId>) {
    match term {
        Terminator::Jump { args, .. } => remap_vec(args, remap),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        }
        | Terminator::Diamond {
            cond,
            then_args,
            else_args,
            ..
        } => {
            remap_val(cond, remap);
            remap_vec(then_args, remap);
            remap_vec(else_args, remap);
        }
        Terminator::Return { value, order, .. } => {
            remap_val(value, remap);
            if let Some(order) = order {
                remap_val(order, remap);
            }
        }
        Terminator::For {
            source,
            body_args,
            exit_args,
            ..
        } => {
            source.for_each_use(|v| remap_val(v, remap));
            remap_vec(body_args, remap);
            remap_vec(exit_args, remap);
        }
        Terminator::Switch { tag, arms, default } => {
            remap_val(tag, remap);
            for (_, _, args) in arms.iter_mut() {
                remap_vec(args, remap);
            }
            if let Some((_, args)) = default {
                remap_vec(args, remap);
            }
        }
        Terminator::Fallthrough | Terminator::Diverge => {}
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
        Terminator::Return { value, order, .. } => std::iter::once(*value).chain(*order).collect(),
        Terminator::Jump { args, .. } => args.clone(),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        }
        | Terminator::Diamond {
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
        Terminator::For {
            source,
            body_args,
            exit_args,
            ..
        } => {
            let mut v = source.uses().to_vec();
            v.extend_from_slice(body_args);
            v.extend_from_slice(exit_args);
            v
        }
        Terminator::Switch { tag, arms, default } => {
            let mut v = vec![*tag];
            for (_, _, args) in arms {
                v.extend_from_slice(args);
            }
            if let Some((_, args)) = default {
                v.extend_from_slice(args);
            }
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
///
/// The loop has no iteration cap - not an omission, a decision. Every
/// `sink_one` that moves strictly decreases `tests::sink_measure`, the sum
/// over every Eval of the positions left below it in its block, so the pass
/// ends within that measure's initial value, which
/// `the_sink_ends_within_its_measure` asserts on a body whose Evals must
/// move in cascade.
fn sink_pass(cfg: &mut CfgBody) {
    while sink_one(cfg) {}
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
    use crate::ir::IndexBound;
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
            demoted_diamonds: Default::default(),
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
            task: crate::ty::Task::Sync,
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
                flows: crate::ty::Flows::Every.into(),
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
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
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
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![v(1)],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![v(2)],
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![v(2)],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![v(3)],
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

    /// How the hoist classifies the last instruction of a body, with the
    /// regions that body gives its values.
    fn classified(insts: Vec<InstKind>) -> Hoistable {
        let last = insts.last().expect("a body to classify").clone();
        let cfg = make_cfg(insts, 8);
        hoistable(&Loans::build(&cfg), &last)
    }

    fn as_slice_of(container: ValueId, mutability: Mutability) -> InstKind {
        InstKind::AsSlice {
            dst: v(2),
            container,
            mutability,
            instance: ExternInstance {
                id: QualifiedRef::root(Interner::new().intern("as_slice")),
                instance: 0,
            },
        }
    }

    #[test]
    fn which_borrow_carries_the_condition_and_which_does_not_move() {
        assert!(matches!(
            classified(vec![ref_of(RefTarget::Var(v(0)), vec![], Mutability::Shared)]),
            Hoistable::SharedBorrow { storage } if storage == v(0)
        ));
        assert!(matches!(
            classified(vec![ref_of(RefTarget::Param(v(0)), vec![], Mutability::Shared)]),
            Hoistable::SharedBorrow { storage } if storage == v(0)
        ));
        assert!(matches!(
            classified(vec![ref_of(RefTarget::Var(v(0)), vec![], Mutability::Mut)]),
            Hoistable::No
        ));
        assert!(matches!(
            classified(vec![ref_of(
                RefTarget::Through(v(0)),
                vec![],
                Mutability::Shared
            )]),
            Hoistable::No
        ));
        assert!(matches!(
            classified(vec![ref_of(
                RefTarget::Var(v(0)),
                vec![PathSeg::Payload],
                Mutability::Shared
            )]),
            Hoistable::No
        ));
    }

    #[test]
    fn a_shared_as_slice_is_a_borrow_of_the_storage_its_container_names() {
        assert!(matches!(
            classified(vec![
                ref_of(RefTarget::Var(v(0)), vec![], Mutability::Shared),
                as_slice_of(v(1), Mutability::Shared),
            ]),
            Hoistable::SharedBorrow { storage } if storage == v(0)
        ));
    }

    #[test]
    fn an_exclusive_as_slice_does_not_move() {
        assert!(matches!(
            classified(vec![
                ref_of(RefTarget::Var(v(0)), vec![], Mutability::Mut),
                as_slice_of(v(1), Mutability::Mut),
            ]),
            Hoistable::No
        ));
    }

    #[test]
    fn an_as_slice_of_a_container_no_one_storage_backs_does_not_move() {
        assert!(matches!(
            classified(vec![as_slice_of(v(7), Mutability::Shared)]),
            Hoistable::No
        ));
    }

    #[test]
    fn indexing_does_not_move() {
        assert!(matches!(
            classified(vec![InstKind::Index {
                dst: v(3),
                slice: v(2),
                index: v(1),
                mode: IndexMode::Copy,
                bound: IndexBound::Checked,
            }]),
            Hoistable::No
        ));
        assert!(matches!(
            classified(vec![InstKind::IndexSet {
                slice: v(2),
                index: v(1),
                value: v(0),
                bound: IndexBound::Checked,
            }]),
            Hoistable::No
        ));
    }

    /// A cycle with two entries has no natural loop and so no depth. The
    /// lowering emits `while` and `while let`, neither of which can produce
    /// one; this is what the pass does if a front end ever does.
    #[test]
    #[should_panic(expected = "irreducible control flow")]
    fn an_irreducible_cycle_is_refused() {
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
                },
                InstKind::Jump {
                    label: Label(1),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                },
                InstKind::Jump {
                    label: Label(0),
                    args: vec![],
                },
            ],
            10,
        );

        run(&mut cfg);
    }

    #[test]
    fn a_heap_constant_is_never_hoisted() {
        assert!(matches!(
            classified(vec![InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::String("a".into()),
            }]),
            Hoistable::No
        ));
        assert!(matches!(
            classified(vec![InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(1),
            }]),
            Hoistable::ControlEquivalent
        ));
        assert!(matches!(
            classified(vec![InstKind::MakeArray {
                dst: v(1),
                elements: vec![],
            }]),
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
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
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
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
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
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
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
                },
                InstKind::Jump {
                    label: Label(5),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(4),
                    params: vec![],
                },
                InstKind::Jump {
                    label: Label(5),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(5),
                    params: vec![],
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
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                },
                InstKind::BinOp {
                    dst: v(1),
                    op: crate::ir::BinOp::Add,
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
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                },
                InstKind::Jump {
                    label: Label(2),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(2),
                    params: vec![],
                },
                InstKind::BinOp {
                    dst: v(1),
                    op: crate::ir::BinOp::Add,
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
                },
                InstKind::BinOp {
                    dst: v(1),
                    op: crate::ir::BinOp::Add,
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
                restores: false,
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
                restores: false,
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

    /// The shape RFC-0047 rule 8 exists for: a loop that reads `v(1)` through
    /// an `AsSlice` and an `Index`, with or without a write to the
    /// container in the body.
    fn a_loop_that_slices_a_container(writes_the_container: bool) -> Vec<InstKind> {
        let mut body = vec![
            InstKind::Ref {
                dst: v(4),
                target: RefTarget::Var(v(1)),
                path: vec![],
                mutability: Mutability::Shared,
            },
            as_slice_of(v(4), Mutability::Shared),
            InstKind::Index {
                dst: v(6),
                slice: v(2),
                index: v(0),
                mode: IndexMode::Copy,
                bound: IndexBound::Checked,
            },
        ];
        if writes_the_container {
            body.push(InstKind::Assign {
                target: RefTarget::Var(v(1)),
                path: vec![],
                value: v(0),
                restores: false,
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
                restores: false,
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
            },
            InstKind::Return {
                value: v(0),
                order: None,
            },
        ]);
        insts
    }

    /// The log kernel's shape (`benches/logs.rs`): an outer loop over `li`
    /// whose body takes `&a[li]` - the source wrote `len(&a[li])` there -
    /// and an inner loop over `i` that reads `a[li][i]`, which is the same
    /// `&a[li]` again, sliced and indexed.
    fn a_loop_that_reindexes_the_row(writes_the_container: bool) -> Vec<InstKind> {
        let index_ref = |dst, slice| InstKind::Index {
            dst,
            slice,
            index: v(0),
            mode: IndexMode::Ref,
            bound: IndexBound::Checked,
        };
        let ref_through = |dst, reference| InstKind::Ref {
            dst,
            target: RefTarget::Through(reference),
            path: vec![],
            mutability: Mutability::Shared,
        };
        let label = |n| InstKind::BlockLabel {
            label: Label(n),
            params: vec![],
        };
        let jump = |n| InstKind::Jump {
            label: Label(n),
            args: vec![],
        };
        let test = |then_label, else_label| InstKind::JumpIf {
            cond: v(3),
            then_label: Label(then_label),
            then_args: vec![],
            else_label: Label(else_label),
            else_args: vec![],
        };

        let mut inner = vec![
            index_ref(v(14), v(11)),
            ref_through(v(15), v(14)),
            InstKind::AsSlice {
                dst: v(16),
                container: v(15),
                mutability: Mutability::Shared,
                instance: ExternInstance {
                    id: QualifiedRef::root(Interner::new().intern("as_slice")),
                    instance: 0,
                },
            },
            InstKind::Index {
                dst: v(17),
                slice: v(16),
                index: v(0),
                mode: IndexMode::Copy,
                bound: IndexBound::Checked,
            },
        ];
        if writes_the_container {
            inner.push(InstKind::Assign {
                target: RefTarget::Var(v(1)),
                path: vec![],
                value: v(0),
                restores: false,
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
                restores: false,
            },
            InstKind::Const {
                dst: v(3),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::Ref {
                dst: v(10),
                target: RefTarget::Var(v(1)),
                path: vec![],
                mutability: Mutability::Shared,
            },
            InstKind::AsSlice {
                dst: v(11),
                container: v(10),
                mutability: Mutability::Shared,
                instance: ExternInstance {
                    id: QualifiedRef::root(Interner::new().intern("as_slice")),
                    instance: 0,
                },
            },
            jump(0),
            label(0),
            test(1, 2),
            label(1),
            index_ref(v(12), v(11)),
            ref_through(v(13), v(12)),
            jump(3),
            label(3),
            test(4, 5),
            label(4),
        ];
        insts.extend(inner);
        insts.extend([
            jump(3),
            label(5),
            jump(0),
            label(2),
            InstKind::Return {
                value: v(0),
                order: None,
            },
        ]);
        insts
    }

    /// The loop depth of the block each instruction stands in after the pass.
    struct Depths {
        slice: usize,
        read: usize,
    }

    fn depths_of_the_slice_and_the_read(insts: Vec<InstKind>) -> Depths {
        let mut cfg = make_cfg(insts, 24);
        run(&mut cfg);
        let domtree = DomTree::build(&cfg);
        let depth = LoopDepth::of(&cfg, &domtree);
        let block_of = |wanted: fn(&InstKind) -> bool| {
            cfg.blocks
                .iter()
                .position(|block| block.insts.iter().any(|inst| wanted(&inst.kind)))
                .map(|bi| depth.at(BlockIdx(bi)))
                .expect("the body still holds the instruction")
        };
        Depths {
            slice: block_of(
                |kind| matches!(kind, InstKind::AsSlice { container, .. } if *container != v(10)),
            ),
            read: block_of(|kind| {
                matches!(
                    kind,
                    InstKind::Index {
                        mode: IndexMode::Copy,
                        ..
                    }
                )
            }),
        }
    }

    fn index_refs_left(insts: Vec<InstKind>) -> usize {
        let mut cfg = make_cfg(insts, 24);
        run(&mut cfg);
        cfg.blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter(|inst| {
                matches!(
                    inst.kind,
                    InstKind::Index {
                        mode: IndexMode::Ref,
                        ..
                    }
                )
            })
            .count()
    }

    #[test]
    fn the_row_a_loop_reindexes_is_the_row_the_block_above_it_named() {
        assert_eq!(
            index_refs_left(a_loop_that_reindexes_the_row(false)),
            1,
            "the inner `&a[li]` is the one the outer body already named"
        );
        let Depths { slice, read } =
            depths_of_the_slice_and_the_read(a_loop_that_reindexes_the_row(false));
        assert!(
            slice < read,
            "the row's slice leaves the inner loop the element read stays in \
             (slice at depth {slice}, read at depth {read})"
        );
    }

    #[test]
    fn a_write_to_the_container_keeps_both_the_index_and_the_slice_inside() {
        assert_eq!(
            index_refs_left(a_loop_that_reindexes_the_row(true)),
            2,
            "the body writes the container, so the row may have moved"
        );
        let Depths { slice, read } =
            depths_of_the_slice_and_the_read(a_loop_that_reindexes_the_row(true));
        assert_eq!(
            slice, read,
            "the slice stays beside the read it serves (slice {slice}, read {read})"
        );
    }

    /// Where the `AsSlice` sits, relative to the loop's test.
    fn as_slice_is_below_the_test(insts: Vec<InstKind>) -> bool {
        let mut cfg = make_cfg(insts, 10);
        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let taken = k
            .iter()
            .position(|k| matches!(k, InstKind::AsSlice { .. }))
            .expect("the body still takes a slice");
        let test = k
            .iter()
            .position(|k| matches!(k, InstKind::JumpIf { .. }))
            .expect("the loop still tests");
        taken > test
    }

    #[test]
    fn a_slice_of_a_container_the_loop_does_not_write_leaves_the_loop() {
        assert!(
            !as_slice_is_below_the_test(a_loop_that_slices_a_container(false)),
            "nothing the borrow would span writes the container"
        );
    }

    #[test]
    fn a_slice_stays_where_the_loop_writes_the_container() {
        assert!(
            as_slice_is_below_the_test(a_loop_that_slices_a_container(true)),
            "the body writes the storage the slice borrows"
        );
    }

    #[test]
    fn an_index_never_leaves_the_loop() {
        let mut cfg = make_cfg(a_loop_that_slices_a_container(false), 10);
        run(&mut cfg);
        let body = demoted(cfg);
        let k = kinds(&body);
        let indexed = k
            .iter()
            .position(|k| matches!(k, InstKind::Index { .. }))
            .expect("the body still indexes");
        let test = k
            .iter()
            .position(|k| matches!(k, InstKind::JumpIf { .. }))
            .expect("the loop still tests");
        assert!(indexed > test, "a checked Index may panic (RFC-0007)");
    }

    // -- Merge tests -------------------------------------------------

    /// One block that borrows `v(1)` shared, optionally takes it `&mut`,
    /// and borrows it shared again. `check_borrows` refuses the `&mut`
    /// version at the source, so the pair lives here rather than in
    /// `acvus-mir-test`.
    fn a_block_that_borrows_twice(takes_it_mut_between: bool) -> Vec<InstKind> {
        let mut insts = vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Assign {
                target: RefTarget::Var(v(1)),
                path: vec![],
                value: v(0),
                restores: false,
            },
            InstKind::Ref {
                dst: v(2),
                target: RefTarget::Var(v(1)),
                path: vec![],
                mutability: Mutability::Shared,
            },
        ];
        if takes_it_mut_between {
            insts.push(InstKind::Ref {
                dst: v(3),
                target: RefTarget::Var(v(1)),
                path: vec![],
                mutability: Mutability::Mut,
            });
        }
        insts.extend([
            InstKind::Ref {
                dst: v(4),
                target: RefTarget::Var(v(1)),
                path: vec![],
                mutability: Mutability::Shared,
            },
            InstKind::Return {
                value: v(0),
                order: None,
            },
        ]);
        insts
    }

    fn shared_borrows_left(insts: Vec<InstKind>) -> usize {
        let mut cfg = make_cfg(insts, 10);
        run(&mut cfg);
        let body = demoted(cfg);
        kinds(&body)
            .iter()
            .filter(|k| {
                matches!(
                    k,
                    InstKind::Ref {
                        mutability: Mutability::Shared,
                        ..
                    }
                )
            })
            .count()
    }

    #[test]
    fn a_borrow_does_not_survive_an_exclusive_take_of_the_storage() {
        assert_eq!(
            shared_borrows_left(a_block_that_borrows_twice(true)),
            2,
            "the `&mut` takes the storage for the life of the reference it makes"
        );
    }

    #[test]
    fn the_same_block_without_that_take_keeps_one_borrow() {
        assert_eq!(
            shared_borrows_left(a_block_that_borrows_twice(false)),
            1,
            "nothing between the two takes the storage"
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
                    op: crate::ir::BinOp::Add,
                    left: v(3),
                    right: v(3),
                },
                InstKind::BinOp {
                    dst: v(4),
                    op: crate::ir::BinOp::Add,
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
                    op: crate::ir::BinOp::Add,
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
                    op: crate::ir::BinOp::Add,
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
                    op: crate::ir::BinOp::Add,
                    left: v(5),
                    right: v(5),
                },
                InstKind::Fetch {
                    dst: v(4),
                    context: ctx,
                },
                InstKind::BinOp {
                    dst: v(6),
                    op: crate::ir::BinOp::Add,
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
                    op: crate::ir::BinOp::Add,
                    left: v(5),
                    right: v(5),
                },
                InstKind::Commit {
                    context: ctx,
                    value: v(5),
                },
                InstKind::BinOp {
                    dst: v(6),
                    op: crate::ir::BinOp::Add,
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
                    op: crate::ir::BinOp::Add,
                    left: v(5),
                    right: v(5),
                },
                InstKind::Fetch {
                    dst: v(4),
                    context: ctx,
                },
                InstKind::BinOp {
                    dst: v(6),
                    op: crate::ir::BinOp::Add,
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

    /// The sink's measure: over every Eval, the positions left below it in
    /// its block.
    fn sink_measure(cfg: &CfgBody) -> usize {
        cfg.blocks
            .iter()
            .flat_map(|block| {
                block
                    .insts
                    .iter()
                    .enumerate()
                    .filter(|(_, inst)| matches!(inst.kind, InstKind::Eval { .. }))
                    .map(move |(index, _)| block.insts.len() - 1 - index)
            })
            .sum()
    }

    /// Three Evals in one block, each used after the Eval below it: the
    /// first Eval reaches its use only after the third, then the second,
    /// has moved out of its way, since an Eval is a call and so a barrier
    /// to the Eval above it.
    fn three_evals_whose_uses_interleave() -> CfgBody {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");
        let spawn = |dst, args| InstKind::Spawn {
            dst,
            callee: Callee::Direct(qref),
            callee_ty: ty.clone(),
            args,
            order: None,
        };
        let eval = |dst, src| InstKind::Eval {
            dst,
            src,
            order: None,
        };
        let add = |dst, left, right| InstKind::BinOp {
            dst,
            op: crate::ir::BinOp::Add,
            left,
            right,
        };
        make_cfg(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                spawn(v(1), vec![v(0)]),
                spawn(v(2), vec![v(0)]),
                spawn(v(3), vec![v(0)]),
                eval(v(4), v(1)),
                eval(v(5), v(2)),
                eval(v(6), v(3)),
                add(v(7), v(4), v(0)),
                add(v(8), v(5), v(7)),
                add(v(9), v(6), v(8)),
                InstKind::Return {
                    value: v(9),
                    order: None,
                },
            ],
            10,
        )
    }

    /// The index of every Eval in the one block, and of the instruction
    /// that first uses what it defines.
    fn evals_and_their_first_uses(cfg: &CfgBody) -> Vec<(usize, usize)> {
        let insts = &cfg.blocks[0].insts;
        insts
            .iter()
            .enumerate()
            .filter_map(|(index, inst)| {
                let InstKind::Eval { dst, .. } = inst.kind else {
                    return None;
                };
                let use_index = insts[index + 1..]
                    .iter()
                    .position(|other| inst_info::uses(&other.kind).contains(&dst))
                    .map(|offset| index + 1 + offset)
                    .unwrap_or_else(|| panic!("no use of {dst:?} below index {index}"));
                Some((index, use_index))
            })
            .collect()
    }

    #[test]
    fn every_eval_lands_immediately_before_its_first_use() {
        let mut cfg = three_evals_whose_uses_interleave();
        assert_eq!(
            evals_and_their_first_uses(&cfg),
            vec![(4, 7), (5, 8), (6, 9)],
            "the body before the pass"
        );

        run(&mut cfg);

        assert_eq!(
            evals_and_their_first_uses(&cfg),
            vec![(4, 5), (6, 7), (8, 9)],
            "the body after the pass"
        );
    }

    #[test]
    fn the_sink_ends_within_its_measure() {
        let mut cfg = three_evals_whose_uses_interleave();
        let bound = sink_measure(&cfg);
        let mut moves = 0;
        while sink_one(&mut cfg) {
            moves += 1;
            assert!(
                moves <= bound,
                "the sink moved {moves} times, past its measure of {bound}"
            );
        }
        assert_eq!(moves, 2, "the third Eval moves, then the second");
    }
}
