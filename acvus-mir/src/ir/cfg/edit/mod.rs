//! The editor (ir-core-structure §7, D47): passes' only way to change IR.
//! One generic engine, `Editor<S>`; an editor's kind is a `Scope`, a type
//! that declares what the editor may change (`CHANGES`), the facts its
//! legality checks read (`Needs`), and the invariants every edit keeps
//! (`Keeps`). Methods exist only where the scope's sealed capability does.
//!
//! The editor lives in `ir::cfg` so that the Rewriter is visible to it alone.
//! The facts it reads are traits of this module (`needs`), implemented by
//! `passes` over its analysis manager, so imports run from `passes` to `ir`
//! only. Each check lends the facts the body as it stands, and each edit
//! tells them what it changed, so a fact is recomputed on the first check
//! that reads it after an edit made it stale (D47).

mod cursor;
mod dominators;
mod keeps;
mod needs;
mod reach;
mod scopes;
#[cfg(test)]
mod tests;

pub use cursor::Cursor;
pub use keeps::{
    DefDominatesUses, Invariant, Invariants, NeverDiverges, OrderChainWhole, SplitTakenOnce, StructureKept, TypesAgree, Violation,
};
pub use needs::{Asked, ContentFacts, DischargeFacts, Unobserved, DominanceFacts, Facts, LoanFacts, NoFacts, OrderFacts};
pub use scopes::{
    MotionFacts, Cfg, EditsCfg, EditsLoops, EditsOrder, Effect, Loop, Motion, MovesInsts, PromotesStorage, Remove, RemovesInsts,
    ReplacesValues, Sccp, Scope, Storage, Value, Discharge, Discharges, Drops, Inline, Inlines, InsertsDrops, Clones,
    ClonesReads, DropFacts,
};

use acvus_ast::Span;

use rustc_hash::FxHashSet;

use super::body::{Def, InstId};
use super::core::{InstShift, structural_fields};
use super::order::At;
use super::{Block, BlockId, Body, BodyId, Changes, Inst, Module, Op, OrderUse, Rewriter, Site, Terminator};
use crate::ir::parts;
use crate::ir::{Callee, IndexBound, PathSeg, RefTarget, SwitchKey, ValueId, fn_effect};
use crate::ty::{Concrete, Mutability, Reissue, SplitCost, Ty, TyKind, TypeArg, Types};
use reach::{BlockPoint, Direction, PointKey, Reached};

/// An instruction of the body being edited, by its identity, which no edit
/// changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstRef {
    id: InstId,
}

impl InstRef {
    pub fn id(self) -> InstId {
        self.id
    }
}

/// Where an instruction is inserted or moved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    Before(InstRef),
    End(BlockId),
}

impl Position {
    fn at(self) -> At {
        match self {
            Position::Before(inst) => At::Before(inst.id),
            Position::End(block) => At::End(block),
        }
    }
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
    /// It may trap, has an effect, writes storage or moves a value out of
    /// a place, and the new place does not run on exactly the runs the old
    /// one does (compilation §1).
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
/// checks; every edit tells it what changed, and it recomputes a fact on
/// the first check after that reads it, so no check reads a stale fact.
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
    /// The body's predecessors and structural labels, kept from their first
    /// use until an edit changes the CFG's shape.
    shape: Option<ShapeIndex>,
}

/// What the CFG's shape alone decides, for the legality checks that read it
/// block by block.
struct ShapeIndex {
    predecessors: rustc_hash::FxHashMap<BlockId, smallvec::SmallVec<[BlockId; 2]>>,
    structural: Vec<bool>,
}

impl ShapeIndex {
    /// Cost: O(blocks + edges + stages).
    fn of(body: &Body) -> ShapeIndex {
        ShapeIndex { predecessors: body.predecessors(), structural: body.structural_blocks() }
    }
}

/// Where a cursor stands: in `block`, just after the instruction `after`,
/// or at the block's start where `after` is none. The next instruction it
/// yields is the one that follows there, or one of a later block. A
/// `block` past the body's last block is the end of the walk.
#[derive(Debug, Clone, Copy)]
struct Walk {
    block: BlockId,
    after: Option<InstId>,
}

impl Walk {
    fn start() -> Walk {
        Walk { block: BlockId::of_index(0), after: None }
    }

    /// Keeps the walk in place across an edit that took `left.id` out of
    /// its place: a walk just after it stands just after what stood before
    /// it.
    fn left(&mut self, left: InstShift) {
        if self.after == Some(left.id) {
            self.after = left.prev;
        }
    }

    /// Keeps the walk with the instruction it stands after, wherever an
    /// edit took that instruction's block.
    fn settle(&mut self, body: &Body) {
        if let Some(after) = self.after {
            self.block = body.block_of(after);
        }
    }

    /// Keeps the walk on the same instructions across a merge of `block`
    /// into `pred`, whose last instruction before the merge was
    /// `pred_tail`: a walk at `block`'s start goes on after `pred_tail`
    /// where `pred` comes first, and otherwise at the block after `block`,
    /// which the removal of `block` gives it, reaching `pred` in its turn.
    fn merge(&mut self, block: BlockId, pred: BlockId, pred_tail: Option<InstId>) {
        if self.block != block || self.after.is_some() {
            return;
        }
        if pred.index() < block.index() {
            *self = Walk { block: pred, after: pred_tail };
        }
    }

    /// The walk past the last of `blocks` blocks.
    fn end(blocks: usize) -> Walk {
        Walk { block: BlockId::of_index(blocks), after: None }
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
            Some(block) => Walk { block: *block, after: None },
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
        Editor {
            module,
            body,
            needs,
            changes: Changes::default(),
            walk: Walk::start(),
            yielded: FxHashSet::default(),
            shape: None,
        }
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
            Def::Inst(id) => Some(InstRef { id }),
            Def::Param(_)
            | Def::Capture(_)
            | Def::OrderParam
            | Def::Slot { .. }
            | Def::BlockParam(..)
            | Def::UnplacedParam(_) => None,
        }
    }

    /// The instruction `id` names, as the editor names it. Cost: O(1).
    ///
    /// # Panics
    /// `id` names no instruction of the body now.
    pub fn inst_ref(&self, id: InstId) -> InstRef {
        assert!(self.body_ref().contains_inst(id), "{id:?} names no instruction of the body");
        InstRef { id }
    }

    /// The first instruction of `block`. Cost: O(1).
    pub fn first_inst(&self, block: BlockId) -> Option<InstRef> {
        self.body_ref().first_inst(block).map(|id| InstRef { id })
    }

    /// The instruction after `inst` in its block. Cost: O(1).
    pub fn next_inst(&self, inst: InstRef) -> Option<InstRef> {
        self.body_ref().next_in_block(inst.id).map(|id| InstRef { id })
    }

    /// The block `inst` stands in. Cost: O(1).
    pub fn block_of(&self, inst: InstRef) -> BlockId {
        self.body_ref().block_of(inst.id)
    }

    /// The instruction `inst` names. Cost: O(1).
    pub fn inst_of(&self, inst: InstRef) -> &Inst {
        self.inst(inst)
    }

    /// Every instruction of `block`, in order, as the body stands now.
    /// Cost: O(its instructions).
    pub fn insts_of(&self, block: BlockId) -> Vec<InstRef> {
        self.body_ref().block(block).insts().map(|inst| InstRef { id: inst.id() }).collect()
    }

    fn body_ref(&self) -> &Body {
        self.module.body(self.body)
    }

    /// The shape index of the body as it stands, made on first use after
    /// the last edit of the CFG's shape.
    fn shape(&mut self) -> &ShapeIndex {
        let body = self.module.body(self.body);
        self.shape.get_or_insert_with(|| ShapeIndex::of(body))
    }

    /// # Panics
    /// `inst` names no instruction of the body.
    fn inst(&self, inst: InstRef) -> &Inst {
        self.body_ref().inst(inst.id)
    }

    /// Makes one edit through the Rewriter, then checks it (debug and test
    /// builds) and tells the needs what it changed.
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
        if changes.touched(super::Aspect::CfgShape) {
            self.shape = None;
        }
        self.needs.refresh(self.body, &changes);
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

    /// Keeps the cursor's place across instructions an edit took out.
    fn shifted(&mut self, shifts: impl IntoIterator<Item = InstShift>) {
        for shift in shifts {
            self.walk.left(shift);
        }
        let body = self.module.body(self.body);
        self.walk.settle(body);
    }

    /// The point `at` names.
    fn point(&self, at: Position) -> BlockPoint {
        match at {
            Position::Before(inst) => BlockPoint { block: self.body_ref().block_of(inst.id), before: Some(inst.id) },
            Position::End(block) => BlockPoint { block, before: None },
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
        let body = self.module.body(self.body);
        let (of_ty, with_ty) = (body.ty_of(of), body.ty_of(with));
        assert!(of_ty == with_ty, "{of:?} of {of_ty:?} is replaced only by a value of its type, not {with:?} of {with_ty:?}");
        let mut sites: Vec<Site> = body.users(of).collect();
        if of == with || sites.is_empty() {
            return Ok(());
        }
        sites.sort_by_key(|site| site_order(body, *site));
        let asked = Asked::new(self.module, self.body);
        for site in sites {
            if self.needs.dominates(asked, with, site_position(site)) {
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
        self.edit(|rewriter| rewriter.insert(at.at(), op, ty, span))
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
            .edit(|rewriter| rewriter.remove_inst(inst.id))
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
    /// the order chain, as a volatile context's place operation does, or it
    /// calls, starts or takes a call whose callee's reissue is not pure or
    /// that writes a context. `Idempotent` is safe to repeat, not to skip,
    /// and `Opaque` never vanishes.
    fn has_effect(&self, inst: &Inst) -> bool {
        if inst.order().is_some() {
            return true;
        }
        let callee_ty = match inst.op() {
            Op::FunctionCall { callee_ty, .. } | Op::Spawn { callee_ty, .. } => *callee_ty,
            Op::TakeEffect { src } => {
                let Def::Inst(start) = self.body_ref().def_of(*src) else {
                    panic!("a taking's handle is a start's result (effects §6)")
                };
                match self.body_ref().inst(start).op() {
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
        beyond.sort_by_key(|site| site_order(body, *site));
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
    let mut keep = vec![false; body.block_count()];
    keep[body.entry().index()] = true;
    let mut pending = vec![body.entry()];
    while let Some(block) = pending.pop() {
        let block: Block<'_> = body.block(block);
        let mut named: Vec<BlockId> = Vec::new();
        block.terminator().for_each_block(|to| named.push(to));
        let operands = block.insts().flat_map(Inst::operands).chain(block.terminator().operands());
        named.extend(operands.filter_map(|value| match body.def_of(value) {
            Def::Inst(defined) => Some(body.block_of(defined)),
            Def::BlockParam(defined, _) => Some(defined),
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
    /// come first on every path to a use of what it defines; where running
    /// it on other runs is observable (`Op::bound_to_its_paths`, or an
    /// effect) and `at` does not run on exactly the runs its place does
    /// (compilation §1); or where the way between its place and
    /// `at` crosses a write to storage it or that write names (ownership §5).
    ///
    /// # Panics
    /// `inst` or `at` names no position of the body.
    pub fn move_to(&mut self, inst: InstRef, at: Position) -> Result<(), CannotMove> {
        if self.stays(inst, at) {
            return Ok(());
        }
        self.can_move(inst, at)?;
        let shift = self.edit(|rewriter| rewriter.move_inst(inst.id, at.at()));
        self.shifted([shift]);
        Ok(())
    }

    /// The check `move_to` makes, without the move: `Ok` exactly where
    /// `move_to(inst, at)` would move `inst`, or leave it where it stands.
    /// Cost: the facts' queries, and where `inst` names storage, the walk
    /// between its place and `at`, O(blocks + edges + the instructions it
    /// passes).
    ///
    /// # Panics
    /// `inst` or `at` names no position of the body.
    pub fn can_move(&mut self, inst: InstRef, at: Position) -> Result<(), CannotMove> {
        if self.stays(inst, at) {
            return Ok(());
        }
        let module: &Module = &*self.module;
        let asked = Asked::new(module, self.body);
        let body = module.body(self.body);
        let moved = body.inst(inst.id);
        for operand in moved.operands() {
            if !self.needs.dominates(asked, operand, at) {
                return Err(CannotMove::OperandNotDominating(operand));
            }
        }
        let mut uses: Vec<Site> = moved.defs().into_iter().flat_map(|value| body.users(value)).collect();
        uses.sort_by_key(|site| site_order(body, *site));
        for site in uses {
            if !self.needs.position_dominates(asked, at, site) {
                return Err(CannotMove::UseNotDominated(site));
            }
        }
        let result_ty = moved.result().map(|result| body.ty_of(result));
        let pinned = moved.op().bound_to_its_paths(result_ty, module.types()) || self.has_effect(moved);
        if pinned && !self.needs.control_equivalent(asked, Position::Before(inst), at) {
            return Err(CannotMove::ChangesWhereItRuns);
        }
        if !self.needs.names_storage(asked, inst) {
            return Ok(());
        }
        for crossed in crossed_insts(body, inst.id, at.at()) {
            let crossed = InstRef { id: crossed };
            if self.needs.conflicts(asked, inst, crossed) || self.needs.conflicts(asked, crossed, inst) {
                return Err(CannotMove::CrossesConflictingWrite(crossed));
            }
        }
        Ok(())
    }

    /// Whether `at` is where `inst` stands already: just before it, or just
    /// after it.
    fn stays(&self, inst: InstRef, at: Position) -> bool {
        let body = self.body_ref();
        match at {
            Position::Before(before) => before.id == inst.id || body.next_in_block(inst.id) == Some(before.id),
            Position::End(block) => body.last_inst(block) == Some(inst.id),
        }
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
        if self.walk.after == Some(call.id) {
            let Def::Inst(taking) = self.body_ref().def_of(result) else {
                panic!("a split call's result is its taking's")
            };
            self.walk.after = Some(taking);
        }
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
        let asked = Asked::new(self.module, self.body);
        if let Some(at) = self.needs.escapes(asked, slot) {
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
fn site_order(body: &Body, site: Site) -> (BlockId, PointKey) {
    match site {
        Site::Inst(inst) => (body.block_of(inst), PointKey::before(body, inst)),
        Site::Term(block) => (block, PointKey::END),
    }
}

/// Where a use stands, as a position a definition must dominate: before its
/// instruction, or at its block's end for its terminator.
fn site_position(site: Site) -> Position {
    match site {
        Site::Inst(id) => Position::Before(InstRef { id }),
        Site::Term(block) => Position::End(block),
    }
}

/// The instructions a move of `from` to `to` passes over, for its loans:
/// those on a way between the two places. The way runs from the new place
/// to the old one for a move up a block, from the old place to the new one
/// for a move down it, and both ways between blocks; it may pass the old
/// place, and it meets the new place only at its end. Around a loop the
/// way takes in every instruction of the loop, which is what a move into or
/// out of the loop passes.
///
/// Each walk runs block by block over runs of points, so the cost is
/// O(blocks + edges) for the walks and O(the instructions passed over, and
/// those skipped at the start of a run) for the answer, in block order.
fn crossed_insts(body: &Body, from: InstId, to: At) -> Vec<InstId> {
    let from_block = body.block_of(from);
    let new_place = match to {
        At::Before(inst) => BlockPoint { block: body.block_of(inst), before: Some(inst) },
        At::End(block) => BlockPoint { block, before: None },
    };
    let old_place = BlockPoint { block: from_block, before: Some(from) };
    let after_old = BlockPoint::after(body, from);
    let same_block = from_block == new_place.block;
    let (new_key, old_key, after_old_key) = (new_place.key(body), old_place.key(body), after_old.key(body));
    let mut ways: Vec<Way> = Vec::new();
    if !same_block || new_key <= old_key {
        ways.push(Way {
            forward: Reached::walk(body, new_place, None, Direction::Forward),
            backward: Reached::walk(body, old_place, Some(new_place), Direction::Backward),
        });
    }
    if !same_block || new_key > after_old_key {
        ways.push(Way {
            forward: Reached::walk(body, after_old, Some(new_place), Direction::Forward),
            backward: Reached::walk(body, new_place, None, Direction::Backward),
        });
    }
    let mut crossed = Vec::new();
    for (block, _) in body.blocks() {
        let mut passed: Vec<InstId> = Vec::new();
        for way in &ways {
            passed.extend(reach::passed(body, block, &way.forward, &way.backward));
        }
        passed.sort_by_key(|inst| PointKey::before(body, *inst));
        passed.dedup();
        crossed.extend(passed.into_iter().filter(|inst| *inst != from));
    }
    crossed
}

/// One way between two places: the points reached going forward from its
/// start, and those reached going backward from its end.
struct Way {
    forward: Reached,
    backward: Reached,
}

// ---- D55: the general capabilities G1-G4 and the narrow operations S1-S3 ----

/// Why an instruction cannot be discharged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotDischarge {
    /// No analysis proves it unobservable.
    NotProven,
    /// The proven fact is not the one this instruction needs (a check needs
    /// `NeverFires`, a store `NeverRead`).
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

/// Why a block parameter cannot be removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotRemoveParam {
    /// Something reads it.
    StillUsed,
    /// A loop's terminator fills it itself, so no argument goes with it.
    Supplied,
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
    /// The storage's content may be read after the position.
    ContentReadAfter,
}

/// Whose part `drop_part` ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartOf {
    /// A slot or a by-value parameter's own storage.
    Place(ValueId),
    Value(ValueId),
}

/// Why a read cannot become a clone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotCloneRead {
    /// The place the `Take` reads is not a `String`.
    NotAString,
}

impl<'m, S> Editor<'m, S>
where
    S: ReplacesValues,
    S::Needs<'m>: DominanceFacts,
{
    /// G3, by operand: the operand at `index` of `site`, in the order
    /// `Inst::operands` and `Terminator::operands` give, which is `of`, takes
    /// `with` instead, and no other operand changes; refused where `with`
    /// does not dominate `site`.
    ///
    /// # Panics
    /// `of` and `with` differ in type, or that operand is not `of`.
    pub fn replace_operand(&mut self, site: Site, index: usize, of: ValueId, with: ValueId) -> Result<(), DoesNotDominate> {
        let body = self.body_ref();
        let (of_ty, with_ty) = (body.ty_of(of), body.ty_of(with));
        assert!(of_ty == with_ty, "{of:?} of {of_ty:?} is replaced only by a value of its type, not {with:?} of {with_ty:?}");
        assert!(site_operands(body, site).get(index) == Some(&of), "operand {index} of {site:?} is not {of:?}");
        if of == with {
            return Ok(());
        }
        let asked = Asked::new(self.module, self.body);
        if !self.needs.dominates(asked, with, site_position(site)) {
            return Err(DoesNotDominate { use_at: site });
        }
        self.edit(|rewriter| rewriter.replace_operand_at(site, index, of, with));
        Ok(())
    }

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
        let asked = Asked::new(self.module, self.body);
        if !self.needs.dominates(asked, with, site_position(site)) {
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

    /// G2: a new empty block that every edge from each block of `from`
    /// into `to` now goes through; it takes parameters like `to`, each edge
    /// passes its own arguments to it, and it passes them on to `to`.
    /// Structure follows the edges, as `split_edge`'s does. A block with
    /// several edges into `to` is listed once, and all of them are
    /// redirected.
    ///
    /// # Panics
    /// `from` is empty, lists a block twice, or lists one with no edge to
    /// `to`.
    pub fn split_edges(&mut self, from: &[BlockId], to: BlockId) -> BlockId {
        let mut edges: Vec<(BlockId, usize)> = Vec::new();
        for (at, block) in from.iter().enumerate() {
            assert!(!from[..at].contains(block), "{block:?} is listed twice; split_edges names each block once");
            let before = edges.len();
            let terminator = self.body_ref().block(*block).terminator();
            let into = terminator.edges().iter().enumerate().filter(|(_, edge)| edge.to == to).map(|(position, _)| (*block, position)).collect::<Vec<_>>();
            edges.extend(into);
            assert!(edges.len() > before, "{block:?} has no edge to {to:?}");
        }
        assert!(!edges.is_empty(), "split_edges names at least one block");
        self.edit(|rewriter| rewriter.split_edges(&edges))
    }

    /// A parameter of a block that nothing reads goes, together with its
    /// argument on every edge into the block; each argument's value then
    /// ends on its edge unless it is read beyond. Refused as `StillUsed`
    /// where something reads it, and as `Supplied` where a loop's
    /// terminator fills it.
    ///
    /// # Panics
    /// `param` is not a block parameter of the body.
    pub fn remove_param(&mut self, param: ValueId) -> Result<(), CannotRemoveParam> {
        let body = self.body_ref();
        let Def::BlockParam(block, index) = body.def_of(param) else {
            panic!("{param:?} is not a block parameter of the body")
        };
        if body.use_count(param) > 0 {
            return Err(CannotRemoveParam::StillUsed);
        }
        let supplied = body
            .blocks()
            .flat_map(|(_, from)| from.terminator().edges())
            .any(|edge| edge.to == block && index < edge.supplied);
        if supplied {
            return Err(CannotRemoveParam::Supplied);
        }
        self.edit(|rewriter| rewriter.remove(param))
            .expect("a parameter nothing reads is removed");
        Ok(())
    }

    /// G2: `block` joins its sole predecessor, which enters it by a `Jump`,
    /// its only edge. Refused where `block` is a `Diamond`'s or a `Switch`'s
    /// join, or a loop's stage or exit, which never merge away (D45, D57);
    /// and where another block or the body's start enters it, or its
    /// predecessor ends otherwise than in that jump.
    pub fn merge_into_predecessor(&mut self, block: BlockId) -> Result<(), CannotMerge> {
        let shape = self.shape();
        if shape.structural[block.index()] {
            return Err(CannotMerge::Structural);
        }
        let [pred] = shape.predecessors[&block][..] else {
            return Err(CannotMerge::NotStraight);
        };
        let body = self.body_ref();
        let jumps = matches!(body.block(pred).terminator(), Terminator::Jump { .. });
        if !jumps || pred == block || block == body.entry() {
            return Err(CannotMerge::NotStraight);
        }
        let pred_tail = body.last_inst(pred);
        let renamed = self.edit(|rewriter| rewriter.merge_into_predecessor(block, pred));
        self.walk.merge(block, pred, pred_tail);
        self.walk.rename(&renamed);
        let body = self.module.body(self.body);
        self.walk.settle(body);
        Ok(())
    }
}

/// Every operand of the instruction or terminator at `site`.
fn site_operands(body: &Body, site: Site) -> smallvec::SmallVec<[ValueId; 4]> {
    match site {
        Site::Inst(inst) => body.inst(inst).operands(),
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
        let block = body.block_of(a.id);
        if !chained || block != body.block_of(b.id) || !body.comes_before(a.id, b.id) {
            return Err(CannotSwap::NotAdjacent);
        }
        let asked = Asked::new(self.module, self.body);
        if !self.needs.may_reorder(asked, a, b) {
            return Err(CannotSwap::Ordered);
        }
        let body = self.body_ref();
        let mut read_between = None;
        body.inst(b.id).op().for_each_operand(|value| {
            let between = match body.def_of(value) {
                Def::Inst(at) => body.block_of(at) == block && !body.comes_before(at, a.id) && body.comes_before(at, b.id),
                _ => false,
            };
            if between && read_between.is_none() {
                read_between = Some(value);
            }
        });
        if let Some(value) = read_between {
            return Err(CannotSwap::DependsOn(value));
        }
        let shift = self.edit(|rewriter| rewriter.swap_effects(a.id, b.id));
        self.shifted([shift]);
        Ok(())
    }
}

impl<'m, S> Editor<'m, S>
where
    S: Discharges,
    S::Needs<'m>: DischargeFacts,
{
    /// G1: discharges what the core would keep, on the fact that proves it
    /// unobservable: a check on `NeverFires`, a store (`Assign`) on
    /// `NeverRead`. A `Check` or `CheckSteps` is removed; an `Index` or
    /// `IndexSet` whose bound is checked keeps its place, and its bound
    /// becomes proven; a store is removed. The body is unchanged on a
    /// refusal.
    ///
    /// # Panics
    /// `inst` is none of those kinds, or an index whose bound is proven
    /// already.
    pub fn discharge(&mut self, inst: InstRef) -> Result<(), CannotDischarge> {
        let checked_bound = matches!(
            self.inst(inst).op(),
            Op::Index { bound: IndexBound::Checked, .. } | Op::IndexSet { bound: IndexBound::Checked, .. }
        );
        let needed = match self.inst(inst).op() {
            Op::Check { .. } | Op::CheckSteps { .. } => Unobserved::NeverFires,
            _ if checked_bound => Unobserved::NeverFires,
            Op::Assign { .. } => Unobserved::NeverRead,
            other => panic!("discharge removes a check or a store, not {other:?}"),
        };
        let asked = Asked::new(self.module, self.body);
        match self.needs.discharged(asked, inst) {
            None => Err(CannotDischarge::NotProven),
            Some(fact) if fact != needed => Err(CannotDischarge::WrongFact),
            Some(_) if checked_bound => {
                self.edit(|rewriter| rewriter.prove_bound(inst.id));
                Ok(())
            }
            Some(_) => {
                let shifts = self
                    .edit(|rewriter| rewriter.remove_inst(inst.id))
                    .expect("a store yields nothing");
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
        sites.sort_by_key(|site| site_order(body, *site));
        let names_a_field = |site: Site| {
            let Site::Inst(id) = site else {
                return false;
            };
            let inst = body.inst(id);
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
        let Def::Inst(made) = body.def_of(*function) else {
            return Err(CannotInline::NoBody);
        };
        // A closure of one body is that body; one of several is chosen only
        // through a `Specialize`, whose code this does not follow.
        let Op::MakeClosure { bodies, captures } = body.inst(made).op() else {
            return Err(CannotInline::NoBody);
        };
        let [callee_body] = &bodies[..] else {
            return Err(CannotInline::NoBody);
        };
        if *callee_body == self.body {
            return Err(CannotInline::Recursive);
        }
        let (callee_body, captures) = (*callee_body, captures.clone());
        let left = InstShift { id: call.id, prev: body.prev_in_block(call.id) };
        self.edit(|rewriter| rewriter.inline_call(call.id, callee_body, &captures));
        self.shifted([left]);
        Ok(())
    }
}

impl<'m, S> Editor<'m, S>
where
    S: InsertsDrops,
    S::Needs<'m>: DropFacts,
{
    /// S3: a `Drop` of `value` at `at`, where nothing uses it after:
    /// refused as `UsedAfter` at the first use, in block order, that a path
    /// from `at` reaches without passing `value`'s definition again, the
    /// instruction at `at` included. A use reached only through the
    /// definition, as around a loop that defines the value, reads the value
    /// that definition makes anew, not the one dropped.
    ///
    /// # Panics
    /// `value` is not defined at `at`, or is an order token, which no drop
    /// ends; or `at` names no position of the body.
    pub fn insert_drop(&mut self, value: ValueId, at: Position, span: Span) -> Result<(), CannotDrop> {
        let asked = Asked::new(self.module, self.body);
        assert!(self.needs.dominates(asked, value, at), "a drop of {value:?} stands where {value:?} is defined");
        if let Some(site) = self.first_use_after(value, at, |_| true) {
            return Err(CannotDrop::UsedAfter(site));
        }
        self.edit(|rewriter| rewriter.insert_drop(at.at(), value, span));
        Ok(())
    }

    /// S3, for a storage: the whole content of `place`, a slot or a by-value
    /// parameter's own storage, is taken out at `at` and dropped there, as
    /// a `Take` of it followed by a `Drop` of what it took. Refused as
    /// `ContentReadAfter` where the content may be read from `at` on.
    ///
    /// # Panics
    /// `place` is neither a slot nor a parameter of the body, or `at` names
    /// no position of the body.
    pub fn drop_content(&mut self, place: ValueId, at: Position, span: Span) -> Result<(), CannotDrop> {
        let target = match self.body_ref().def_of(place) {
            Def::Slot { .. } => RefTarget::Var(place),
            Def::Param(_) => RefTarget::Param(place),
            def => panic!("{place:?} is defined at {def:?}, and only a slot or a parameter has its own storage"),
        };
        let asked = Asked::new(self.module, self.body);
        if self.needs.content_read_after(asked, place, at) {
            return Err(CannotDrop::ContentReadAfter);
        }
        let ty = self.body_ref().ty_of(place);
        self.edit(|rewriter| {
            let taken = Op::Take { target, path: Vec::new(), taken_out: false };
            let content = rewriter.insert(at.at(), taken, ty, span);
            rewriter.insert_drop(at.at(), content, span);
        });
        Ok(())
    }

    /// S3, for a part (ownership §6): the part of `of` at `path` is taken
    /// out at `at` and dropped there. A place's part is a `Take` of it. A
    /// value's is the extraction that names it: a `FieldGet` for a path of
    /// fields, and a `TupleIndex` or an `ArrayIndex` for one index. Refused
    /// as `ContentReadAfter` where the place's part may be read from `at`
    /// on, and as `UsedAfter` at the first use of the value that touches the
    /// part and that a path from `at` reaches, found as `insert_drop` finds
    /// a use.
    ///
    /// # Panics
    /// A place that is neither a slot nor a parameter; a value not defined
    /// at `at`, or its path neither all fields nor one index; `path` naming
    /// no part of its holder; or `at` naming no position of the body.
    pub fn drop_part(&mut self, of: PartOf, path: &[PathSeg], at: Position, span: Span) -> Result<(), CannotDrop> {
        let asked = Asked::new(self.module, self.body);
        let taken = match of {
            PartOf::Place(place) => {
                let target = match self.body_ref().def_of(place) {
                    Def::Slot { .. } => RefTarget::Var(place),
                    Def::Param(_) => RefTarget::Param(place),
                    def => panic!("{place:?} is defined at {def:?}, and only a slot or a parameter has its own storage"),
                };
                if self.needs.part_read_after(asked, place, path, at) {
                    return Err(CannotDrop::ContentReadAfter);
                }
                (place, Op::Take { target, path: path.to_vec(), taken_out: false })
            }
            PartOf::Value(value) => {
                assert!(
                    self.needs.dominates(asked, value, at),
                    "a drop of part of {value:?} stands where {value:?} is defined"
                );
                let body = self.body_ref();
                if let Some(site) = self.first_use_after(value, at, |site| match site {
                    Site::Inst(id) => touches_part(body.inst(id).op(), value, path),
                    Site::Term(_) => true,
                }) {
                    return Err(CannotDrop::UsedAfter(site));
                }
                (value, extraction(self.module.types(), self.body_ref().ty_of(value), value, path))
            }
        };
        let (holder, op) = taken;
        let ty = parts::part_ty(self.module.types(), self.body_ref().ty_of(holder), path);
        self.edit(|rewriter| {
            let part = rewriter.insert(at.at(), op, ty, span);
            rewriter.insert_drop(at.at(), part, span);
        });
        Ok(())
    }

    /// The first use of `value`, in block order, that `counts` and that a
    /// path from `at` reaches without passing `value`'s definition again.
    /// Cost: O(blocks + edges) for the walk, and O(uses · log uses).
    fn first_use_after(&self, value: ValueId, at: Position, counts: impl Fn(Site) -> bool) -> Option<Site> {
        let body = self.body_ref();
        let redefined = match body.def_of(value) {
            Def::Inst(defined) => Some(BlockPoint::after(body, defined)),
            Def::BlockParam(defined, _) => Some(BlockPoint::first_of(body, defined)),
            Def::Param(_) | Def::Capture(_) | Def::OrderParam | Def::Slot { .. } | Def::UnplacedParam(_) => None,
        };
        let reached = Reached::walk(body, self.point(at), redefined, Direction::Forward);
        let follows = |site: Site| {
            let (block, key) = site_order(body, site);
            reached.holds(block, key)
        };
        let mut uses: Vec<Site> = body.users(value).collect();
        uses.sort_by_key(|site| site_order(body, *site));
        uses.into_iter().find(|site| follows(*site) && counts(*site))
    }
}

/// Whether `op`, a use of `value`, touches its part at `path`: an
/// extraction of an overlapping part, a `FieldSet` of a part that does not
/// hold it, and any other use, which uses the value whole.
fn touches_part(op: &Op, value: ValueId, path: &[PathSeg]) -> bool {
    match op {
        Op::FieldGet { object, field, rest } if *object == value => {
            let named: Vec<PathSeg> = std::iter::once(field).chain(rest).map(|name| PathSeg::Field(*name)).collect();
            parts::overlap(&named, path)
        }
        Op::ObjectGet { object, key } if *object == value => parts::overlap(&[PathSeg::Field(*key)], path),
        Op::TupleIndex { tuple: whole, index } | Op::ArrayIndex { array: whole, index } if *whole == value => {
            parts::overlap(&[PathSeg::Index(*index)], path)
        }
        Op::FieldSet { object, field, rest, .. } if *object == value => {
            let replaced: Vec<PathSeg> = std::iter::once(field).chain(rest).map(|name| PathSeg::Field(*name)).collect();
            !parts::covers(&replaced, path)
        }
        _ => true,
    }
}

/// The extraction of `value`'s part at `path`, of a value of type `ty`.
///
/// # Panics
/// `path` is neither all fields nor one index of a tuple or an array.
fn extraction(types: &Types<Concrete>, ty: Ty, value: ValueId, path: &[PathSeg]) -> Op {
    let fields: Option<Vec<acvus_utils::Astr>> = path
        .iter()
        .map(|seg| match seg {
            PathSeg::Field(name) => Some(*name),
            PathSeg::Index(_) | PathSeg::Payload => None,
        })
        .collect();
    match (fields.as_deref(), path, types.kind(ty)) {
        (Some([field, rest @ ..]), _, _) => Op::FieldGet { object: value, field: *field, rest: rest.to_vec() },
        (_, [PathSeg::Index(index)], TyKind::Tuple(.., _)) => Op::TupleIndex { tuple: value, index: *index },
        (_, [PathSeg::Index(index)], TyKind::Array(..)) => Op::ArrayIndex { array: value, index: *index },
        (_, path, kind) => panic!("a value's part is named by fields or by one index of a tuple or an array, not {path:?} of {kind:?}"),
    }
}


impl<S> Editor<'_, S>
where
    S: ClonesReads,
{
    /// S4: the `Take` at `take`, of a `String` place, becomes a shared
    /// `Ref` of that place and a `StringClone` of it, with the `Take`'s
    /// span; what used the `Take`'s result uses the clone, and the place
    /// keeps its content (ownership §3). `&String` is interned in the
    /// module's table. Refused as `NotAString` where the place is not a
    /// `String`.
    ///
    /// # Panics
    /// `take` is not a `Take`.
    pub fn clone_read(&mut self, take: InstRef) -> Result<(), CannotCloneRead> {
        let inst = self.inst(take);
        let Op::Take { target, path, .. } = inst.op() else {
            panic!("{take:?} is {:?}, not a Take", inst.op())
        };
        let (target, path, span) = (*target, path.clone(), inst.span());
        let read = inst.result().expect("a Take yields a value");
        let string = self.body_ref().ty_of(read);
        if !matches!(self.module.types().kind(string), TyKind::String) {
            return Err(CannotCloneRead::NotAString);
        }
        let at = At::Before(take.id);
        let shifts = self.edit(|rewriter| {
            let lent = rewriter.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(string)));
            let reference = rewriter.insert(at, Op::Ref { target, path, mutability: Mutability::Shared }, lent, span);
            let clone = rewriter.insert(at, Op::StringClone { src: reference }, string, span);
            rewriter.replace_all_uses(read, clone);
            rewriter.remove_inst(take.id).expect("a Take whose uses went to its clone has none")
        });
        self.shifted(shifts);
        Ok(())
    }
}
