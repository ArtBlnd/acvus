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
pub use needs::{DischargeFacts, Unobserved, DominanceFacts, Facts, LoanFacts, NoFacts, OrderFacts};
pub use scopes::{
    MotionFacts, Cfg, EditsCfg, EditsLoops, EditsOrder, Effect, Loop, Motion, MovesInsts, PromotesStorage, Remove, RemovesInsts,
    ReplacesValues, Sccp, Scope, Storage, Value, Discharge, Discharges, Drops, Inline, Inlines, InsertsDrops,
};

use std::collections::BTreeSet;

use acvus_ast::Span;

use rustc_hash::FxHashSet;

use super::body::{Def, InstId};
use super::core::{InstShift, structural_fields};
use super::{Block, BlockId, Body, BodyId, Changes, Inst, Module, Op, OrderUse, Rewriter, Site, Terminator};
use crate::ir::{Callee, IndexBound, RefTarget, SwitchKey, ValueId, fn_effect};
use crate::ty::{Concrete, Reissue, SplitCost, Ty, Types};

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
    /// A use a promotion cannot rewrite names it.
    Escapes(Site),
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
    needs: &'m mut S::Needs<'m>,
    changes: Changes,
    walk: Walk,
    /// Every instruction the current cursor has yielded. Cost: O(1) per
    /// yield, and O(instructions yielded) held.
    yielded: FxHashSet<InstId>,
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

    /// Keeps the walk on the same instructions across a merge of `block`,
    /// whose instructions now follow the `base` instructions of `pred`:
    /// a walk inside `block` goes on at the same instruction in `pred` where
    /// `pred` comes first, and otherwise at the block after `block`, which
    /// the removal of `block` gives it, reaching `pred` in its turn; a walk
    /// past the end of `pred`, where `block` came before it and so was
    /// walked already, goes on past what `block` brought.
    fn merge(&mut self, block: BlockId, pred: BlockId, base: usize, moved: usize) {
        if self.block == block && pred.index() < block.index() {
            *self = Walk { block: pred, next: base + self.next };
        } else if self.block == pred && self.next >= base && block.index() < pred.index() {
            self.next = base + moved;
        }
    }

    /// Keeps the walk on the same instructions across an edit that moved
    /// the instructions of `block` from `at` on to the start of `tail`.
    fn split(&mut self, block: BlockId, at: usize, tail: BlockId) {
        if self.block == block && self.next > at {
            *self = Walk { block: tail, next: self.next - at };
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
    pub fn new(module: &'m mut Module, body: BodyId, needs: &'m mut S::Needs<'m>) -> Self {
        module.assert_body(body);
        Editor { module, body, needs, changes: Changes::default(), walk: Walk::start(), yielded: FxHashSet::default() }
    }

    /// A cursor over the body's instructions in block order; it stays valid
    /// across every edit made through it.
    pub fn cursor(&mut self) -> Cursor<'_, 'm, S> {
        self.walk = Walk::start();
        self.yielded.clear();
        Cursor::new(self)
    }

    /// Every change the edits made. In debug and test builds, each edit has
    /// already been checked against `S::CHANGES` and `S::Keeps`.
    pub fn finish(self) -> Changes {
        self.changes
    }

    /// The body as the last edit left it. Cost: O(1).
    pub fn body(&self) -> &Body {
        self.body_ref()
    }

    /// The module's one type table. Cost: O(1).
    pub fn types(&self) -> &Types<Concrete> {
        self.module.types()
    }

    /// The instruction whose result or yielded token `value` is; none for a
    /// value defined elsewhere, such as a parameter or a slot. Cost: O(1).
    ///
    /// # Panics
    /// `value` is not a value of the body.
    pub fn defining_inst(&self, value: ValueId) -> Option<InstRef> {
        match self.body_ref().def_of(value) {
            super::body::Def::Inst(block, index) => Some(InstRef { block, index }),
            super::body::Def::Param(_)
            | super::body::Def::Capture(_)
            | super::body::Def::OrderParam
            | super::body::Def::Slot { .. }
            | super::body::Def::BlockParam(..)
            | super::body::Def::UnplacedParam(_) => None,
        }
    }

    /// The instruction at `index` of `block` as the last edit left it;
    /// none past the block's end. Cost: O(1).
    ///
    /// # Panics
    /// `block` is not a block of the body.
    pub fn inst_at(&self, block: BlockId, index: usize) -> Option<InstRef> {
        (index < self.body_ref().block(block).insts().len()).then_some(InstRef { block, index })
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

impl<'m, S> Editor<'m, S>
where
    S: ReplacesValues,
    S::Needs<'m>: DominanceFacts,
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
    /// Every edge into `from` goes to `to`, with its arguments; structure
    /// follows (core rule 6), so a `Diamond`'s or a `Switch`'s join or a
    /// loop's stage or exit that is `from` becomes `to`. Refused as `Entry`
    /// where `from` is the entry, which no edge enters; as `ParamsDisagree`
    /// where the two blocks' parameters differ in count or type; and as
    /// `ParamsUsedBeyond` at the first use, in block order, of a parameter
    /// of `from` other than `from`'s own terminator, which would be left
    /// without its definition.
    pub fn retarget(&mut self, from: BlockId, to: BlockId) -> Result<(), CannotRetarget> {
        let body = self.body_ref();
        if from == body.entry() {
            return Err(CannotRetarget::Entry);
        }
        let param_tys = |block: BlockId| -> Vec<Ty> { body.block(block).params().iter().map(|param| body.ty_of(*param)).collect() };
        if param_tys(from) != param_tys(to) {
            return Err(CannotRetarget::ParamsDisagree);
        }
        let mut beyond: Vec<Site> = body
            .block(from)
            .params()
            .iter()
            .flat_map(|param| body.users(*param))
            .filter(|site| *site != Site::Term(from))
            .collect();
        beyond.sort_by_key(|site| site_order(*site));
        if let Some(site) = beyond.first() {
            return Err(CannotRetarget::ParamsUsedBeyond(*site));
        }
        self.edit(|rewriter| rewriter.retarget(from, to));
        Ok(())
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

    /// A `Switch` whose tag is known takes the arm of `key`, or its default
    /// where no arm has that key, and becomes a jump into it; its join
    /// stays a block of the body, and no structure is recovered afterwards
    /// (D45, D57).
    ///
    /// # Panics
    /// `block` does not end in a `Switch`, or no arm has `key` and the
    /// `Switch` has no default.
    pub fn take_case(&mut self, block: BlockId, key: &SwitchKey) {
        self.edit(|rewriter| rewriter.take_case(block, key));
    }

    /// Removes blocks no edge reaches. A block a surviving `Diamond` or
    /// `Switch` names as its join, or a surviving loop as a stage or its
    /// exit, stays, as does every block such a block reaches or takes a
    /// value from, so the body stays whole.
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

impl<'m, S> Editor<'m, S>
where
    S: MovesInsts,
    S::Needs<'m>: DominanceFacts + LoanFacts,
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

impl<'m, S> Editor<'m, S>
where
    S: EditsOrder,
    S::Needs<'m>: OrderFacts,
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

impl<'m, S> Editor<'m, S>
where
    S: PromotesStorage,
    S::Needs<'m>: LoanFacts,
{
    /// Promotes a slot to values (SSA construction); refused where a use
    /// other than a whole-slot `Take` or `Assign` names it.
    ///
    /// # Panics
    /// `slot` is not a slot of the body, or such a use names it though the
    /// facts say none does.
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

// ---- D55: the general capabilities G1-G4 and the narrow operations S1-S3 ----

/// Why an instruction cannot be discharged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotDischarge {
    /// No analysis proves it unobservable.
    NotProven,
    /// The proven fact is not the one this instruction needs (a check needs
    /// `NeverFires`, a store or load `NeverRead`, a commit `NeverWritten`).
    WrongFact,
}

/// Why two effects cannot trade places.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotSwap {
    /// They are not adjacent on one token chain.
    NotAdjacent,
    /// The declarations keep their order (effects §1).
    Ordered,
    /// The later one reads this value, defined from the earlier one up to
    /// it, so it cannot run first.
    DependsOn(ValueId),
}

/// Why the edges into a block cannot go to another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotRetarget {
    /// It is the entry, which no edge enters.
    Entry,
    /// A parameter of it is used here, beyond its own terminator; the use
    /// would be left without its definition.
    ParamsUsedBeyond(Site),
    /// The two blocks' parameters differ in count or type.
    ParamsDisagree,
}

/// Why a block cannot merge into its predecessor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotMerge {
    /// It has other predecessors, or its predecessor other successors.
    NotStraight,
    /// It is a `Diamond`'s join or a loop's stage or exit (D45).
    Structural,
}

/// Why a slot cannot be split into its fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotSplitSlot {
    /// Its type is not structural; an extern-defined type is never split
    /// (compilation §3).
    NotStructural,
    /// A use a split cannot rewrite names it.
    Escapes(Site),
}

/// Why a call cannot be inlined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotInline {
    /// Its callee is not a body of this module.
    NoBody,
    /// The callee is the caller, directly.
    Recursive,
}

/// Why a drop cannot stand at a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotDrop {
    /// The value is used after the position.
    UsedAfter(Site),
}

impl<'m, S> Editor<'m, S>
where
    S: ReplacesValues,
    S::Needs<'m>: DominanceFacts,
{
    /// G3: the operands at `site` that are `of` take `with` instead, and no
    /// other use changes; refused where `with` does not dominate `site`.
    ///
    /// # Panics
    /// `of` and `with` differ in type, or `site` does not read `of`.
    pub fn replace_use(&mut self, site: Site, of: ValueId, with: ValueId) -> Result<(), DoesNotDominate> {
        let body = self.body_ref();
        let (of_ty, with_ty) = (body.ty_of(of), body.ty_of(with));
        assert!(of_ty == with_ty, "{of:?} of {of_ty:?} is replaced only by a value of its type, not {with:?} of {with_ty:?}");
        assert!(site_operands(body, site).contains(&of), "{site:?} does not read {of:?}");
        if of == with {
            return Ok(());
        }
        if !self.needs.dominates(with, site_position(site)) {
            return Err(DoesNotDominate { use_at: site });
        }
        self.edit(|rewriter| rewriter.replace_uses_at(site, of, with));
        Ok(())
    }
}

impl<S> Editor<'_, S>
where
    S: EditsCfg,
{
    /// G2: a new empty block on the edge `from -> to`; it takes parameters
    /// like `to`, the edge passes its arguments to it and it passes them
    /// on. Structure follows the edge: where the edge enters a loop's body
    /// or leaves by its exit, the new block is that body or exit.
    ///
    /// # Panics
    /// `from` has no edge to `to`, or more than one, so the edge is not
    /// named.
    pub fn split_edge(&mut self, from: BlockId, to: BlockId) -> BlockId {
        let positions: Vec<usize> = self
            .body_ref()
            .block(from)
            .terminator()
            .edges()
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.to == to)
            .map(|(position, _)| position)
            .collect();
        let [position] = positions[..] else {
            panic!("{from:?} has {} edges to {to:?}; split_edge names exactly one", positions.len())
        };
        self.edit(|rewriter| rewriter.split_edge(from, position))
    }

    /// G2: `block` joins its sole predecessor, which enters it by a `Jump`,
    /// its only edge. Refused where `block` is a `Diamond`'s or a `Switch`'s
    /// join, or a loop's stage or exit, which never merge away (D45, D57);
    /// and where another block or the body's start enters it, or its
    /// predecessor ends otherwise than in that jump.
    pub fn merge_into_predecessor(&mut self, block: BlockId) -> Result<(), CannotMerge> {
        let body = self.body_ref();
        if body.is_structural(block) {
            return Err(CannotMerge::Structural);
        }
        let preds = body.predecessors();
        let [pred] = preds[&block][..] else {
            return Err(CannotMerge::NotStraight);
        };
        let jumps = matches!(body.block(pred).terminator(), Terminator::Jump { .. });
        if !jumps || pred == block || block == body.entry() {
            return Err(CannotMerge::NotStraight);
        }
        let (base, moved) = (body.block(pred).insts().len(), body.block(block).insts().len());
        let renamed = self.edit(|rewriter| rewriter.merge_into_predecessor(block, pred));
        self.walk.merge(block, pred, base, moved);
        self.walk.rename(&renamed);
        Ok(())
    }
}

/// Every operand of the instruction or terminator at `site`.
fn site_operands(body: &Body, site: Site) -> smallvec::SmallVec<[ValueId; 4]> {
    match site {
        Site::Inst(block, index) => body.block(block).insts()[index].operands(),
        Site::Term(block) => body.block(block).terminator().operands(),
    }
}

impl<'m, S> Editor<'m, S>
where
    S: EditsOrder,
    S::Needs<'m>: OrderFacts,
{
    /// G4: two effects adjacent on one chain trade places, where the
    /// declarations let them (effects §1). `b` consumes what `a` yields, and
    /// nothing else does; afterwards `b` consumes what `a` consumed, `a`
    /// consumes what `b` yields, and what used `b`'s yield uses `a`'s. `b`
    /// moves to stand just before `a`, so each token is still defined
    /// before its uses (effects §3).
    ///
    /// Refused as `NotAdjacent` where either is not an effectful call on a
    /// chain, `b` does not consume `a`'s yield alone, or the two do not
    /// stand in one block with `a` first; as `Ordered` where the
    /// declarations keep their order; and as `DependsOn` where `b` reads a
    /// value defined from `a` up to it.
    pub fn swap(&mut self, a: InstRef, b: InstRef) -> Result<(), CannotSwap> {
        let (first, second) = (self.inst(a), self.inst(b));
        let (Some(OrderUse::Edge(yielded)), Some(OrderUse::Edge(consumed))) = (first.order(), second.order()) else {
            return Err(CannotSwap::NotAdjacent);
        };
        let body = self.body_ref();
        let chained = consumed.before == yielded.after && body.use_count(yielded.after) == 1;
        if !chained || a.block != b.block || a.index >= b.index {
            return Err(CannotSwap::NotAdjacent);
        }
        if !self.needs.may_reorder(a, b) {
            return Err(CannotSwap::Ordered);
        }
        let mut read_between = None;
        second.op().for_each_operand(|value| {
            let between =
                matches!(body.def_of(value), Def::Inst(block, index) if block == a.block && (a.index..b.index).contains(&index));
            if between && read_between.is_none() {
                read_between = Some(value);
            }
        });
        if let Some(value) = read_between {
            return Err(CannotSwap::DependsOn(value));
        }
        let shifts = self.edit(|rewriter| rewriter.swap_effects(a.block, a.index, b.index));
        self.shifted(shifts);
        Ok(())
    }
}

impl<'m, S> Editor<'m, S>
where
    S: Discharges,
    S::Needs<'m>: DischargeFacts,
{
    /// G1: discharges what the core would keep, on the fact that proves it
    /// unobservable: a check on `NeverFires`, a store (`Assign`) or a load
    /// (`Fetch`) on `NeverRead`, a commit on `NeverWritten`. A `Check` or
    /// `CheckSteps` is removed; an `Index` or `IndexSet` whose bound is
    /// checked keeps its place, and its bound becomes proven; every other
    /// kind is removed, and an effect's token is handed on to what used the
    /// token it yields. The body is unchanged on a refusal.
    ///
    /// # Panics
    /// `inst` is none of those kinds, or an index whose bound is proven
    /// already; or the facts call a load never read whose result has a use.
    pub fn discharge(&mut self, inst: InstRef) -> Result<(), CannotDischarge> {
        let checked_bound = matches!(
            self.inst(inst).op(),
            Op::Index { bound: IndexBound::Checked, .. } | Op::IndexSet { bound: IndexBound::Checked, .. }
        );
        let needed = match self.inst(inst).op() {
            Op::Check { .. } | Op::CheckSteps { .. } => Unobserved::NeverFires,
            _ if checked_bound => Unobserved::NeverFires,
            Op::Assign { .. } | Op::Fetch { .. } => Unobserved::NeverRead,
            Op::Commit { .. } => Unobserved::NeverWritten,
            other => panic!("discharge removes a check, a store, a load or a commit, not {other:?}"),
        };
        match self.needs.discharged(inst) {
            None => Err(CannotDischarge::NotProven),
            Some(fact) if fact != needed => Err(CannotDischarge::WrongFact),
            Some(_) if checked_bound => {
                self.edit(|rewriter| rewriter.prove_bound(inst.block, inst.index));
                Ok(())
            }
            Some(_) => {
                let shifts = self
                    .edit(|rewriter| rewriter.remove_inst(inst.block, inst.index))
                    .expect("a load the facts call never read has no use");
                self.shifted(shifts);
                Ok(())
            }
        }
    }
}

impl<'m, S> Editor<'m, S>
where
    S: PromotesStorage,
    S::Needs<'m>: LoanFacts,
{
    /// S1: a slot of a tuple or a structural object becomes one slot per
    /// field, in field order, and every `Take`, `Assign` and `Ref` of it
    /// goes to the slot of the field its path names first. Refused as
    /// `NotStructural` for any other type, an extern-defined one among them
    /// (compilation §3), and as `Escapes` at the first use that names the
    /// slot whole.
    ///
    /// # Panics
    /// `slot` is not a slot of the body.
    pub fn split_slot(&mut self, slot: ValueId) -> Result<Vec<ValueId>, CannotSplitSlot> {
        let body = self.body_ref();
        assert!(matches!(body.def_of(slot), Def::Slot { .. }), "{slot:?} is not a slot of the body");
        let Some(fields) = structural_fields(self.module.types(), body.ty_of(slot)) else {
            return Err(CannotSplitSlot::NotStructural);
        };
        let mut sites: Vec<Site> = body.users(slot).collect();
        sites.sort_by_key(|site| site_order(*site));
        let names_a_field = |site: Site| {
            let Site::Inst(block, index) = site else {
                return false;
            };
            let inst = &body.block(block).insts()[index];
            let once = inst.operands().iter().filter(|value| **value == slot).count() == 1;
            let field = inst.op().place().is_some_and(|(target, path)| {
                *target == RefTarget::Var(slot) && path.first().is_some_and(|seg| fields.iter().any(|(named, _)| named == seg))
            });
            once && field
        };
        if let Some(site) = sites.into_iter().find(|site| !names_a_field(*site)) {
            return Err(CannotSplitSlot::Escapes(site));
        }
        Ok(self.edit(|rewriter| rewriter.split_slot(slot)))
    }
}

impl<S> Editor<'_, S>
where
    S: Inlines,
{
    /// S2: the call's callee body stands in its place. The callee is known
    /// where the call is indirect through a `MakeClosure` of this module:
    /// its parameters take the arguments, its captures take the closure's
    /// captured values, its token chain continues the caller's, and its
    /// returns jump to a new continuation block, whose parameter is the
    /// call's result and which holds what followed the call. Refused as
    /// `NoBody` where the callee is not such a body, and as `Recursive`
    /// where it is the caller's own body.
    ///
    /// # Panics
    /// `call` is not a `FunctionCall`.
    pub fn inline(&mut self, call: InstRef) -> Result<(), CannotInline> {
        let inst = self.inst(call);
        let Op::FunctionCall { callee, .. } = inst.op() else {
            panic!("{call:?} is {:?}, not a call", inst.op())
        };
        let Callee::Indirect(function) = callee else {
            return Err(CannotInline::NoBody);
        };
        let body = self.body_ref();
        let Def::Inst(block, index) = body.def_of(*function) else {
            return Err(CannotInline::NoBody);
        };
        let Op::MakeClosure { body: callee_body, captures } = body.block(block).insts()[index].op() else {
            return Err(CannotInline::NoBody);
        };
        if *callee_body == self.body {
            return Err(CannotInline::Recursive);
        }
        let (callee_body, captures) = (*callee_body, captures.clone());
        let continuation = self.edit(|rewriter| rewriter.inline_call(call.block, call.index, callee_body, &captures));
        self.walk.split(call.block, call.index + 1, continuation);
        self.shifted([InstShift::Removed { block: call.block, index: call.index }]);
        Ok(())
    }
}

impl<'m, S> Editor<'m, S>
where
    S: InsertsDrops,
    S::Needs<'m>: DominanceFacts,
{
    /// S3: a `Drop` of `value` at `at`, where nothing uses it after:
    /// refused as `UsedAfter` at the first use, in block order, that a path
    /// from `at` reaches, the instruction at `at` included.
    ///
    /// # Panics
    /// `value` is not defined at `at`, or is an order token, which no drop
    /// ends; or `at` names no position of the body.
    pub fn insert_drop(&mut self, value: ValueId, at: Position, span: Span) -> Result<(), CannotDrop> {
        let (block, index) = self.place(at);
        assert!(self.needs.dominates(value, at), "a drop of {value:?} stands where {value:?} is defined");
        let body = self.body_ref();
        let points = Points::of(body);
        let reached = points.reach(body, points.id(block, index), None, Direction::Forward);
        let follows = |site: Site| match site {
            Site::Inst(block, index) => reached[points.id(block, index)],
            Site::Term(block) => reached[points.id(block, body.block(block).insts().len())],
        };
        let mut uses: Vec<Site> = body.users(value).collect();
        uses.sort_by_key(|site| site_order(*site));
        if let Some(site) = uses.into_iter().find(|site| follows(*site)) {
            return Err(CannotDrop::UsedAfter(site));
        }
        self.edit(|rewriter| rewriter.insert_drop(block, index, value, span));
        self.shifted([InstShift::Inserted { block, index }]);
        Ok(())
    }
}
