//! The editor (ir-core-structure §7, D47): passes' only way to change IR.
//! One generic engine, `Editor<S>`; an editor's kind is a `Scope`, a type
//! that declares what the editor may change (`CHANGES`), the facts its
//! legality checks read (`Needs`), and the invariants every edit keeps
//! (`Keeps`). Methods exist only where the scope's sealed capability does.
//!
//! The editor lives in `ir::cfg` so that the Rewriter is visible to it alone.
//! The facts it reads are traits of this module (`needs`), implemented by
//! `passes` over its analysis manager, so imports run from `passes` to `ir`
//! only.

mod cursor;
mod dominators;
mod keeps;
mod needs;
mod scopes;
#[cfg(test)]
mod tests;

pub use cursor::Cursor;
pub use keeps::{
    DefDominatesUses, Invariant, Invariants, OrderChainWhole, SplitTakenOnce, StructureKept, TypesAgree, Violation,
};
pub use needs::{DominanceFacts, Facts, LoanFacts, NoFacts, OrderFacts};
pub use scopes::{
    MotionFacts, Cfg, EditsCfg, EditsLoops, EditsOrder, Effect, Loop, Motion, MovesInsts, PromotesStorage, Remove, RemovesInsts,
    ReplacesValues, Sccp, Scope, Storage, Value,
};

use std::collections::BTreeSet;

use acvus_ast::Span;

use super::core::InstShift;
use super::{Block, BlockId, Body, BodyId, Changes, Inst, Module, Op, Rewriter, Site};
use crate::ir::{ValueId, fn_effect};
use crate::ty::{Reissue, SplitCost, Ty};

/// An instruction of the body being edited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstRef {
    block: BlockId,
    index: usize,
}

impl InstRef {
    pub fn block(self) -> BlockId {
        self.block
    }
    pub fn index(self) -> usize {
        self.index
    }
}

/// Where an instruction is inserted or moved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    Before(InstRef),
    End(BlockId),
}

/// A replacement whose value does not dominate a use it would take over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DoesNotDominate {
    pub use_at: Site,
}

/// Why an instruction cannot be removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unremovable {
    StillUsed,
    /// It may trap, and whether a run traps is its meaning (compilation §1).
    Traps,
    /// It writes storage.
    Writes,
    /// It loads or commits a context, or calls a callee whose reissue is not
    /// pure: an effect the declarations do not let vanish.
    Effect,
}

/// Why an instruction cannot move to a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotMove {
    /// An operand, order token included, is not defined where it would stand.
    OperandNotDominating(ValueId),
    /// A use of what it defines, order token included, would not be reached
    /// from where it would stand.
    UseNotDominated(Site),
    /// It may trap or has an effect, and the new place does not run on
    /// exactly the runs the old one does (compilation §1).
    ChangesWhereItRuns,
    /// It would cross a write to storage its loans name.
    CrossesConflictingWrite(InstRef),
}

/// Why a call cannot be split (effects §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotSplit {
    /// Its callee's effect does not allow a split.
    NotSplittable,
}

/// Why a storage slot cannot be promoted to values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotPromote {
    /// A reference to it escapes the promotable uses.
    Escapes(InstRef),
}

/// The editor over one body, of scope `S`. `needs` answers its legality
/// checks and is refreshed from the change record after every edit, so no
/// check reads a stale fact.
pub struct Editor<'m, S>
where
    S: Scope,
{
    module: &'m mut Module,
    body: BodyId,
    needs: &'m mut S::Needs,
    changes: Changes,
    walk: Walk,
}

/// Where a cursor stands: the next instruction it yields is at `next` of
/// `block`, or in a later block when `next` is past `block`'s end. A
/// `block` past the body's last block is the end of the walk.
#[derive(Debug, Clone, Copy)]
struct Walk {
    block: BlockId,
    next: usize,
}

impl Walk {
    fn start() -> Walk {
        Walk { block: BlockId::of_index(0), next: 0 }
    }

    /// Keeps the walk on the same instructions across an edit that put one
    /// in or took one out.
    fn shift(&mut self, shift: InstShift) {
        match shift {
            InstShift::Inserted { block, index } if block == self.block && index < self.next => self.next += 1,
            InstShift::Removed { block, index } if block == self.block && index < self.next => self.next -= 1,
            InstShift::Inserted { .. } | InstShift::Removed { .. } => {}
        }
    }

    /// The walk past the last of `blocks` blocks.
    fn end(blocks: usize) -> Walk {
        Walk { block: BlockId::of_index(blocks), next: 0 }
    }

    /// Keeps the walk across a removal of blocks: a walk in a removed block
    /// goes on at the start of the first kept block after it, and ends
    /// where no kept block follows.
    fn rename(&mut self, renamed: &[Option<BlockId>]) {
        if let Some(Some(block)) = renamed.get(self.block.index()) {
            self.block = *block;
            return;
        }
        let kept_after = renamed.get(self.block.index()..).into_iter().flatten().flatten().next();
        *self = match kept_after {
            Some(block) => Walk { block: *block, next: 0 },
            None => Walk::end(renamed.iter().flatten().count()),
        };
    }
}

impl<'m, S> Editor<'m, S>
where
    S: Scope,
{
    /// # Panics
    /// `body` is not a body of `module`.
    pub fn new(module: &'m mut Module, body: BodyId, needs: &'m mut S::Needs) -> Self {
        module.assert_body(body);
        Editor { module, body, needs, changes: Changes::default(), walk: Walk::start() }
    }

    /// A cursor over the body's instructions in block order; it stays valid
    /// across every edit made through it.
    pub fn cursor(&mut self) -> Cursor<'_, 'm, S> {
        self.walk = Walk::start();
        Cursor::new(self)
    }

    /// Every change the edits made. In debug and test builds, each edit has
    /// already been checked against `S::CHANGES` and `S::Keeps`.
    pub fn finish(self) -> Changes {
        self.changes
    }

    fn body_ref(&self) -> &Body {
        self.module.body(self.body)
    }

    /// # Panics
    /// `inst` names no instruction of the body.
    fn inst(&self, inst: InstRef) -> &Inst {
        let insts = self.body_ref().block(inst.block).insts();
        insts
            .get(inst.index)
            .unwrap_or_else(|| panic!("{inst:?} names no instruction; {:?} holds {}", inst.block, insts.len()))
    }

    /// The block and index a position names; a block's end is its length.
    ///
    /// # Panics
    /// `at` names no instruction of the body.
    fn place(&self, at: Position) -> (BlockId, usize) {
        match at {
            Position::Before(inst) => {
                self.inst(inst);
                (inst.block, inst.index)
            }
            Position::End(block) => (block, self.body_ref().block(block).insts().len()),
        }
    }

    /// Makes one edit through the Rewriter, then checks it (debug and test
    /// builds) and refreshes the needs from what it changed.
    fn edit<R>(&mut self, edit: impl FnOnce(&mut Rewriter<'_>) -> R) -> R {
        let mut rewriter = self.module.rewrite(self.body);
        let made = edit(&mut rewriter);
        let changes = rewriter.finish();
        self.after_edit(changes);
        made
    }

    fn after_edit(&mut self, changes: Changes) {
        #[cfg(debug_assertions)]
        self.check_edit(&changes);
        self.needs.refresh(self.module, self.body, &changes);
        for aspect in changes.aspects() {
            self.changes.record(aspect);
        }
    }

    #[cfg(debug_assertions)]
    fn check_edit(&self, changes: &Changes) {
        let scope = std::any::type_name::<S>();
        for aspect in changes.aspects() {
            assert!(
                S::CHANGES.contains(&aspect),
                "an edit of scope {scope} changed {aspect:?}, outside the scope's {:?}; the editor is at fault",
                S::CHANGES
            );
        }
        if let Err(Violation { invariant, detail }) = S::Keeps::check(self.body_ref(), self.module.types()) {
            panic!("an edit of scope {scope} broke {invariant}: {detail}");
        }
    }

    fn shifted(&mut self, shifts: impl IntoIterator<Item = InstShift>) {
        for shift in shifts {
            self.walk.shift(shift);
        }
    }
}

impl<S> Editor<'_, S>
where
    S: ReplacesValues,
    S::Needs: DominanceFacts,
{
    /// Every use of `of` takes `with`; refused where `with` does not
    /// dominate a use.
    ///
    /// # Panics
    /// `of` and `with` differ in type.
    pub fn replace_uses(&mut self, of: ValueId, with: ValueId) -> Result<(), DoesNotDominate> {
        let body = self.body_ref();
        let (of_ty, with_ty) = (body.ty_of(of), body.ty_of(with));
        assert!(of_ty == with_ty, "{of:?} of {of_ty:?} is replaced only by a value of its type, not {with:?} of {with_ty:?}");
        let mut sites: Vec<Site> = body.users(of).collect();
        if of == with || sites.is_empty() {
            return Ok(());
        }
        sites.sort_by_key(|site| site_order(*site));
        for site in sites {
            if self.needs.dominates(with, site_position(site)) {
                continue;
            }
            return Err(DoesNotDominate { use_at: site });
        }
        self.edit(|rewriter| rewriter.replace_all_uses(of, with));
        Ok(())
    }

    /// Inserts a pure instruction.
    ///
    /// # Panics
    /// `op` is effectful, or a `Spawn`, `TakeEffect` or `Merge`, which only
    /// the core makes; it yields no value; or `at` names no position of the
    /// body.
    pub fn insert(&mut self, at: Position, op: Op, ty: Ty, span: Span) -> ValueId {
        let (block, index) = self.place(at);
        let result = self.edit(|rewriter| rewriter.insert(block, index, op, ty, span));
        self.shifted([InstShift::Inserted { block, index }]);
        result
    }
}

impl<S> Editor<'_, S>
where
    S: RemovesInsts,
{
    /// Removes an instruction whose result nothing uses, that neither
    /// traps, nor writes storage, nor has an effect. Removing a dead store
    /// is not this operation's.
    pub fn remove(&mut self, inst: InstRef) -> Result<(), Unremovable> {
        let removed = self.inst(inst);
        if removed.result().is_some_and(|result| self.body_ref().use_count(result) > 0) {
            return Err(Unremovable::StillUsed);
        }
        if removed.op().may_trap() {
            return Err(Unremovable::Traps);
        }
        if removed.op().writes_storage() {
            return Err(Unremovable::Writes);
        }
        if self.has_effect(removed) {
            return Err(Unremovable::Effect);
        }
        let shifts = self
            .edit(|rewriter| rewriter.remove_inst(inst.block, inst.index))
            .expect("an instruction whose result nothing uses is removed");
        self.shifted(shifts);
        Ok(())
    }
}

impl<S> Editor<'_, S>
where
    S: Scope,
{
    /// Whether `inst` has an effect (effects §1, context §3): it stands on
    /// the order chain, loads or commits a context, or calls, starts or
    /// takes a call whose callee's reissue is not pure or that writes a
    /// context. `Idempotent` is safe to repeat, not to skip, and `Opaque`
    /// never vanishes.
    fn has_effect(&self, inst: &Inst) -> bool {
        if inst.order().is_some() || inst.op().touches_context() {
            return true;
        }
        let callee_ty = match inst.op() {
            Op::FunctionCall { callee_ty, .. } | Op::Spawn { callee_ty, .. } => *callee_ty,
            Op::TakeEffect { src } => {
                let Site::Inst(block, index) = self.defining_site(*src) else {
                    panic!("a taking's handle is a start's result (effects §6)")
                };
                match self.body_ref().block(block).insts()[index].op() {
                    Op::Spawn { callee_ty, .. } => *callee_ty,
                    other => panic!("a taking's handle is a start's result (effects §6), not {other:?}'s"),
                }
            }
            _ => return false,
        };
        let Some(effect) = fn_effect(self.module.types(), callee_ty) else {
            panic!("a call's callee type is a function type")
        };
        effect.reissue() != Reissue::Pure || !effect.writes().is_empty()
    }

    fn defining_site(&self, value: ValueId) -> Site {
        match self.body_ref().def_of(value) {
            super::body::Def::Inst(block, index) => Site::Inst(block, index),
            def => panic!("{value:?} is defined at {def:?}, not by an instruction"),
        }
    }
}

impl<S> Editor<'_, S>
where
    S: EditsCfg,
{
    /// Every edge into `from` goes to `to`; structure follows (core rule 6).
    ///
    /// # Panics
    /// The parameters of `from` and `to` differ.
    pub fn retarget(&mut self, from: BlockId, to: BlockId) {
        self.edit(|rewriter| rewriter.retarget(from, to));
    }

    /// A branch whose condition is known takes its one arm. A `Diamond`
    /// whose arm is known becomes a jump into that arm, and its join stays a
    /// block of the body; no structure is recovered afterwards (D45).
    ///
    /// # Panics
    /// `block` does not end in a `JumpIf` or a `Diamond`.
    pub fn take_arm(&mut self, block: BlockId, then: bool) {
        self.edit(|rewriter| rewriter.take_arm(block, then));
    }

    /// Removes blocks no edge reaches. A block a surviving `Diamond` names
    /// as its join, or a surviving loop as a stage or its exit, stays, as
    /// does every block such a block reaches or takes a value from, so the
    /// body stays whole.
    pub fn prune(&mut self) {
        let keep = kept_blocks(self.body_ref());
        if keep.iter().all(|kept| *kept) {
            return;
        }
        let renamed = self.edit(|rewriter| rewriter.remove_blocks(&keep));
        self.walk.rename(&renamed);
    }
}

/// The blocks the entry reaches through edges and the structural names of
/// the terminators it reaches, closed over the blocks that define what a
/// kept block reads.
fn kept_blocks(body: &Body) -> Vec<bool> {
    let mut keep = vec![false; body.blocks().count()];
    keep[body.entry().index()] = true;
    let mut pending = vec![body.entry()];
    while let Some(block) = pending.pop() {
        let block: &Block = body.block(block);
        let mut named: Vec<BlockId> = Vec::new();
        block.terminator().for_each_block(|to| named.push(to));
        let operands = block.insts().iter().flat_map(Inst::operands).chain(block.terminator().operands());
        named.extend(operands.filter_map(|value| match body.def_of(value) {
            super::body::Def::Inst(defined, _) | super::body::Def::BlockParam(defined, _) => Some(defined),
            _ => None,
        }));
        for block in named {
            if !keep[block.index()] {
                keep[block.index()] = true;
                pending.push(block);
            }
        }
    }
    keep
}

impl<S> Editor<'_, S>
where
    S: MovesInsts,
    S::Needs: DominanceFacts + LoanFacts,
{
    /// Moves `inst` to stand at `at`, with its operands and what it
    /// defines, order tokens included; the order chain is their def-use, so
    /// dominance keeps it whole and no token is rethreaded (D54). It is
    /// refused where an operand is not defined at `at`; where `at` does not
    /// come first on every path to a use of what it defines; where it may
    /// trap or has an effect and `at` does not run on exactly the runs its
    /// place does (compilation §1); or where the way between its place and
    /// `at` crosses a write to storage it or that write names (ownership §5).
    ///
    /// # Panics
    /// `inst` or `at` names no position of the body.
    pub fn move_to(&mut self, inst: InstRef, at: Position) -> Result<(), CannotMove> {
        let (to_block, to_index) = self.place(at);
        let moved = self.inst(inst);
        if to_block == inst.block && (to_index == inst.index || to_index == inst.index + 1) {
            return Ok(());
        }
        for operand in moved.operands() {
            if !self.needs.dominates(operand, at) {
                return Err(CannotMove::OperandNotDominating(operand));
            }
        }
        let body = self.body_ref();
        let mut uses: Vec<Site> = moved.defs().into_iter().flat_map(|value| body.users(value)).collect();
        uses.sort_by_key(|site| site_order(*site));
        if let Some(site) = uses.into_iter().find(|site| !self.needs.position_dominates(at, *site)) {
            return Err(CannotMove::UseNotDominated(site));
        }
        let pinned = moved.op().may_trap() || self.has_effect(moved);
        if pinned && !self.needs.control_equivalent(Position::Before(inst), at) {
            return Err(CannotMove::ChangesWhereItRuns);
        }
        let crossed = crossed_insts(body, inst, to_block, to_index);
        let conflicting = crossed
            .into_iter()
            .find(|crossed| self.needs.conflicts(inst, *crossed) || self.needs.conflicts(*crossed, inst));
        if let Some(crossed) = conflicting {
            return Err(CannotMove::CrossesConflictingWrite(crossed));
        }

        let shifts = self.edit(|rewriter| rewriter.move_inst(inst.block, inst.index, to_block, to_index));
        self.shifted(shifts);
        Ok(())
    }
}

impl<S> Editor<'_, S>
where
    S: EditsOrder,
    S::Needs: OrderFacts,
{
    /// Splits a call into its start and its taking (effects §6).
    ///
    /// # Panics
    /// `call` is not a call.
    pub fn split_call(&mut self, call: InstRef) -> Result<(), CannotSplit> {
        let inst = self.inst(call);
        let Op::FunctionCall { callee_ty, .. } = inst.op() else {
            panic!("{call:?} is {:?}, not a call", inst.op())
        };
        let result = inst.result().expect("a call yields a value");
        let Some(effect) = fn_effect(self.module.types(), *callee_ty) else {
            panic!("a call's callee type is a function type")
        };
        match effect.split() {
            SplitCost::Unavailable => return Err(CannotSplit::NotSplittable),
            SplitCost::WithCost(_) => {}
        }
        self.edit(|rewriter| rewriter.split_call(result));
        self.shifted([InstShift::Inserted { block: call.block, index: call.index }]);
        Ok(())
    }
}

impl<S> Editor<'_, S>
where
    S: PromotesStorage,
    S::Needs: LoanFacts,
{
    /// Promotes a slot to values (SSA construction); refused where a
    /// reference to it escapes.
    ///
    /// # Panics
    /// `slot` is not a slot of the body, or something other than a
    /// whole-slot `Take` or `Assign` names it though no reference escapes.
    pub fn promote(&mut self, slot: ValueId) -> Result<(), CannotPromote> {
        if let Some(at) = self.needs.escapes(slot) {
            return Err(CannotPromote::Escapes(at));
        }
        let shifts = self.edit(|rewriter| rewriter.promote(slot));
        self.shifted(shifts);
        Ok(())
    }
}

// `EditsLoops`' operations are defined with the loop design (compilation
// §4), after optimize's design; none exists yet.

/// An edit made on the body directly, checked and refreshed as any edit is,
/// for tests that provoke what no operation of the editor does.
#[cfg(test)]
impl<S> Editor<'_, S>
where
    S: Scope,
{
    fn edit_directly(&mut self, edit: impl FnOnce(&mut Module) -> Changes) {
        let changes = edit(self.module);
        self.after_edit(changes);
    }
}

/// Uses in block order, a block's instructions before its terminator.
fn site_order(site: Site) -> (BlockId, usize) {
    match site {
        Site::Inst(block, index) => (block, index),
        Site::Term(block) => (block, usize::MAX),
    }
}

/// Where a use stands, as a position a definition must dominate: before its
/// instruction, or at its block's end for its terminator.
fn site_position(site: Site) -> Position {
    match site {
        Site::Inst(block, index) => Position::Before(InstRef { block, index }),
        Site::Term(block) => Position::End(block),
    }
}

/// The instructions a move from `from` to position `to_index` of `to`
/// passes over, for its loans: those on a way between the two places. The way runs from the
/// new place to the old one for a move up a block, from the old place to the
/// new one for a move down it, and both ways between blocks; it may pass the
/// old place, and it meets the new place only at its end. Around a loop the
/// way takes in every instruction of the loop, which is what a move into or
/// out of the loop passes. Cost: O(the body's instructions and edges).
fn crossed_insts(body: &Body, from: InstRef, to: BlockId, to_index: usize) -> Vec<InstRef> {
    let points = Points::of(body);
    let new_place = points.id(to, to_index);
    let same_block = from.block == to;
    let mut crossed = BTreeSet::new();
    let mut collect = |forward: &[bool], backward: &[bool]| {
        for (block, insts) in body.blocks().map(|(id, block)| (id, block.insts().len())) {
            for index in (0..insts).filter(|index| (block, *index) != (from.block, from.index)) {
                if forward[points.id(block, index)] && backward[points.id(block, index + 1)] {
                    crossed.insert((block, index));
                }
            }
        }
    };
    if !same_block || to_index <= from.index {
        let forward = points.reach(body, new_place, None, Direction::Forward);
        let backward = points.reach(body, points.id(from.block, from.index), Some(new_place), Direction::Backward);
        collect(&forward, &backward);
    }
    if !same_block || to_index > from.index + 1 {
        let forward = points.reach(body, points.id(from.block, from.index + 1), Some(new_place), Direction::Forward);
        let backward = points.reach(body, new_place, None, Direction::Backward);
        collect(&forward, &backward);
    }
    crossed.into_iter().map(|(block, index)| InstRef { block, index }).collect()
}

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Backward,
}

/// The program points of a body: before each instruction of each block, and
/// at its end, numbered block after block.
struct Points {
    offsets: Vec<usize>,
    count: usize,
}

impl Points {
    fn of(body: &Body) -> Points {
        let mut offsets = Vec::new();
        let mut count = 0;
        for (_, block) in body.blocks() {
            offsets.push(count);
            count += block.insts().len() + 1;
        }
        Points { offsets, count }
    }

    fn id(&self, block: BlockId, index: usize) -> usize {
        self.offsets[block.index()] + index
    }

    /// The points reachable from `start` in `direction` without stepping
    /// onto `avoided`.
    fn reach(&self, body: &Body, start: usize, avoided: Option<usize>, direction: Direction) -> Vec<bool> {
        let preds = body.predecessors();
        let mut reached = vec![false; self.count];
        reached[start] = true;
        let mut pending = vec![start];
        while let Some(point) = pending.pop() {
            let block = BlockId::of_index(self.offsets.partition_point(|offset| *offset <= point) - 1);
            let index = point - self.offsets[block.index()];
            let len = body.block(block).insts().len();
            let mut next: Vec<usize> = Vec::new();
            match direction {
                Direction::Forward if index < len => next.push(point + 1),
                Direction::Forward => {
                    next.extend(body.block(block).terminator().edges().iter().map(|edge| self.id(edge.to, 0)))
                }
                Direction::Backward if index > 0 => next.push(point - 1),
                Direction::Backward => next.extend(preds[&block].iter().map(|pred| {
                    self.id(*pred, body.block(*pred).insts().len())
                })),
            }
            for point in next {
                if Some(point) != avoided && !reached[point] {
                    reached[point] = true;
                    pending.push(point);
                }
            }
        }
        reached
    }
}
