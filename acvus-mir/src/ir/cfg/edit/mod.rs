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
mod keeps;
mod needs;
mod scopes;

pub use cursor::Cursor;
pub use keeps::{
    DefDominatesUses, Invariant, Invariants, OrderChainWhole, SplitTakenOnce, StructureKept, TypesAgree, Violation,
};
pub use needs::{DominanceFacts, Facts, LoanFacts, NoFacts, OrderFacts};
pub use scopes::{
    MotionFacts, Cfg, EditsCfg, EditsLoops, EditsOrder, Effect, Loop, Motion, MovesInsts, PromotesStorage, Remove, RemovesInsts,
    ReplacesValues, Sccp, Scope, Storage, Value,
};

use acvus_ast::Span;

use super::{BlockId, BodyId, Changes, Module, Op};
use crate::ir::ValueId;
use crate::ty::Ty;

/// An instruction of the body being edited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstRef {
    block: BlockId,
    index: usize,
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
    pub use_at: InstRef,
}

/// Why an instruction cannot be removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unremovable {
    StillUsed,
    /// Its effect is one the declarations do not allow to vanish.
    Effect,
}

/// Why an instruction cannot move to a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotMove {
    /// An operand is not defined where it would stand.
    OperandNotDominating(ValueId),
    /// A use would stand where the moved result is not defined.
    UseNotDominated(InstRef),
    /// It would cross an effect the declarations keep in order with it.
    CrossesOrderedEffect(InstRef),
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
}

impl<'m, S> Editor<'m, S>
where
    S: Scope,
{
    pub fn new(module: &'m mut Module, body: BodyId, needs: &'m mut S::Needs) -> Self {
        todo!("ir::cfg::edit::Editor::new")
    }

    /// A cursor over the body's instructions in block order; it stays valid
    /// across every edit made through it.
    pub fn cursor(&mut self) -> Cursor<'_, 'm, S> {
        todo!("ir::cfg::edit::Editor::cursor")
    }

    /// Every change the edits made. In debug and test builds, each edit has
    /// already been checked against `S::CHANGES` and `S::Keeps`.
    pub fn finish(self) -> Changes {
        todo!("ir::cfg::edit::Editor::finish")
    }
}

impl<S> Editor<'_, S>
where
    S: ReplacesValues,
    S::Needs: DominanceFacts,
{
    /// Every use of `of` takes `with`; refused where `with` does not
    /// dominate a use.
    pub fn replace_uses(&mut self, of: ValueId, with: ValueId) -> Result<(), DoesNotDominate> {
        todo!("ir::cfg::edit::Editor::replace_uses")
    }

    /// Inserts a pure instruction.
    pub fn insert(&mut self, at: Position, op: Op, ty: Ty, span: Span) -> ValueId {
        todo!("ir::cfg::edit::Editor::insert")
    }
}

impl<S> Editor<'_, S>
where
    S: RemovesInsts,
{
    /// Removes an instruction whose result nothing uses and whose effect,
    /// if any, the declarations allow to vanish.
    pub fn remove(&mut self, inst: InstRef) -> Result<(), Unremovable> {
        todo!("ir::cfg::edit::Editor::remove")
    }
}

impl<S> Editor<'_, S>
where
    S: EditsCfg,
{
    /// Every edge into `from` goes to `to`; structure follows (core rule 6).
    pub fn retarget(&mut self, from: BlockId, to: BlockId) {
        todo!("ir::cfg::edit::Editor::retarget")
    }

    /// A branch whose condition is known takes its one arm. A `Diamond`
    /// whose arm is known becomes a jump into that arm, and its join stays a
    /// block of the body; no structure is recovered afterwards (D45).
    pub fn take_arm(&mut self, block: BlockId, then: bool) {
        todo!("ir::cfg::edit::Editor::take_arm")
    }

    /// Removes blocks no edge reaches.
    pub fn prune(&mut self) {
        todo!("ir::cfg::edit::Editor::prune")
    }
}

impl<S> Editor<'_, S>
where
    S: MovesInsts,
    S::Needs: DominanceFacts + LoanFacts + OrderFacts,
{
    pub fn move_to(&mut self, inst: InstRef, at: Position) -> Result<(), CannotMove> {
        todo!("ir::cfg::edit::Editor::move_to")
    }
}

impl<S> Editor<'_, S>
where
    S: EditsOrder,
    S::Needs: OrderFacts,
{
    /// Splits a call into its start and its taking (effects §6).
    pub fn split_call(&mut self, call: InstRef) -> Result<(), CannotSplit> {
        todo!("ir::cfg::edit::Editor::split_call")
    }
}

impl<S> Editor<'_, S>
where
    S: PromotesStorage,
    S::Needs: LoanFacts,
{
    /// Promotes a slot to values (SSA construction); refused where a
    /// reference to it escapes.
    pub fn promote(&mut self, slot: ValueId) -> Result<(), CannotPromote> {
        todo!("ir::cfg::edit::Editor::promote")
    }
}

// `EditsLoops`' operations are defined with the loop design (compilation
// §4), after optimize's design; none exists yet.

