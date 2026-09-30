//! The loop edits (compilation §4): a loop is its head, its join, its body
//! and its exit, and each edit changes one of them, so every reshaping of a
//! loop is these edits composed with the body's value, motion and control-flow
//! edits. Promoting a `While` to a `For` is `set_head` and `remove_join` per
//! recurrence; strength reduction is `add_join`; an exit value computed from
//! the count is `remove_exit`.
//!
//! An edit checks form: the loop is one (a header ending in a loop
//! terminator), each operand is settled before the header runs (it dominates
//! the header) or dominates the uses it replaces, types agree, and SSA and
//! loop canonical form hold after it. The caller keeps the meaning
//! (compilation §1): a new head yields the iterations the old did, a join
//! parameter removed is replaced by a value equal to it on every iteration,
//! and so on. The editor cannot check a meaning; it keeps form.

use super::{Editor, EditsLoops};
use crate::ir::cfg::BlockId;
use crate::ir::{ForSource, ValueId};

/// A loop's head: how its iterations are yielded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Head {
    /// Each iteration runs where `cond`, computed in the header, is true.
    While { cond: ValueId },
    /// The iterations `source` yields; the body's leading parameters are the
    /// ones it supplies (`ForSource::supplied_params`).
    For { source: ForSource },
}

/// Why a loop edit is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotEditLoop {
    /// The block does not end in a loop terminator.
    NotALoop,
    /// An operand is not settled where the edit needs it: this value.
    NotSettled(ValueId),
    /// The body is entered by something other than the head.
    BodyEnteredElsewhere,
    /// The value is not a join parameter of this loop.
    NotAJoinParam(ValueId),
    /// The value is not an exit parameter of this loop.
    NotAnExitParam(ValueId),
    /// A value's type differs from the one its place has.
    TypeMismatch(ValueId),
    /// A back edge the arguments do not name, or one they name twice.
    BackEdges,
}

impl<S> Editor<'_, S>
where
    S: EditsLoops,
{
    /// Replaces the loop's head, keeping its join, body and exit. The body's
    /// leading supplied parameters become the new head's, returned in order;
    /// a supplied parameter the old head had and the new one lacks must have
    /// no use left.
    pub fn set_head(&mut self, header: BlockId, head: Head) -> Result<Vec<ValueId>, CannotEditLoop> {
        todo!()
    }

    /// Adds a value handed from one iteration to the next: a header parameter
    /// whose entry argument is `init` and whose argument on each back edge is
    /// the one `next` names for that edge's source block. Returns the
    /// parameter.
    pub fn add_join(
        &mut self,
        header: BlockId,
        init: ValueId,
        next: &[(BlockId, ValueId)],
    ) -> Result<ValueId, CannotEditLoop> {
        todo!()
    }

    /// Removes the join parameter `param`, every use of it replaced by `with`,
    /// which dominates each use; its entry and back-edge arguments go.
    pub fn remove_join(&mut self, header: BlockId, param: ValueId, with: ValueId) -> Result<(), CannotEditLoop> {
        todo!()
    }

    /// Adds a value the loop gives its exit: a parameter of the exit block
    /// filled by `value` on every edge into it. Returns the parameter.
    pub fn add_exit(&mut self, header: BlockId, value: ValueId) -> Result<ValueId, CannotEditLoop> {
        todo!()
    }

    /// Removes the exit parameter `param`, every use of it replaced by `with`,
    /// which dominates each use.
    pub fn remove_exit(&mut self, header: BlockId, param: ValueId, with: ValueId) -> Result<(), CannotEditLoop> {
        todo!()
    }
}
