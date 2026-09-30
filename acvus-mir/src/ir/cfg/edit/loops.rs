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

use super::dominators::{DefPoint, Dominators, def_points};
use super::{Editor, EditsLoops};
use crate::ir::cfg::{BlockId, Body, Site, Terminator};
use crate::ir::{ForSource, ValueId};
use crate::ty::{IntTy, Mutability, Ty, TyKind};

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
    ///
    /// Refused as `NotALoop`; `NotSettled` for a source operand that does
    /// not dominate the header, or a condition that does not reach the
    /// header's end; `TypeMismatch` for a source of no traversable type, a
    /// range whose ends differ in type, or a kept parameter whose type the
    /// new head does not supply there; `BodyEnteredElsewhere`. Cost:
    /// O(blocks + edges + values) for the dominators and definitions it
    /// reads, and the core's O(the terminator's operands and the body's
    /// parameters).
    ///
    /// # Panics
    /// A parameter the new head does not supply has a use.
    pub fn set_head(&mut self, header: BlockId, head: Head) -> Result<Vec<ValueId>, CannotEditLoop> {
        let found = self.a_loop(header)?;
        let form = Form::of(self.body_ref());
        match &head {
            Head::While { cond } => form.reaches_end_of(*cond, header)?,
            Head::For { source } => {
                for operand in source.uses() {
                    form.settled_before(operand, header)?;
                }
            }
        }
        if self.shape().predecessors[&found.body].iter().any(|pred| *pred != header) {
            return Err(CannotEditLoop::BodyEnteredElsewhere);
        }
        let supplied = self.supplied(&head)?;
        let body = self.body_ref();
        let old = body.block(found.body).params();
        for (param, ty) in old.iter().zip(&supplied) {
            if body.ty_of(*param) != *ty {
                return Err(CannotEditLoop::TypeMismatch(*param));
            }
        }
        if let Some(used) = old.iter().skip(supplied.len()).find(|param| body.use_count(**param) > 0) {
            panic!("{used:?}, which the new head does not supply, still has a use; the caller replaces it first");
        }
        Ok(self.edit(|rewriter| rewriter.set_loop_head(header, &head, &supplied)))
    }

    /// Adds a value handed from one iteration to the next: a header parameter
    /// whose entry argument is `init` and whose argument on each back edge is
    /// the one `next` names for that edge's source block. Returns the
    /// parameter.
    ///
    /// Refused as `NotALoop`; `NotSettled` for an `init` that does not
    /// dominate the header or a `next` value that does not reach its back
    /// edge's source's end; `TypeMismatch` for a `next` value of another
    /// type than `init`'s; `BackEdges` where `next` misses a back edge's
    /// source, names one twice, or names a block with no back edge. A back
    /// edge is an edge into the header from a block it dominates. Cost:
    /// O(blocks + edges + values), and the core's O(the edges into the
    /// header).
    pub fn add_join(&mut self, header: BlockId, init: ValueId, next: &[(BlockId, ValueId)]) -> Result<ValueId, CannotEditLoop> {
        self.a_loop(header)?;
        let form = Form::of(self.body_ref());
        form.settled_before(init, header)?;
        let preds = self.shape().predecessors[&header].clone();
        let (latches, entering): (Vec<BlockId>, Vec<BlockId>) = preds.iter().partition(|pred| form.dominators.dominates_block(header, **pred));
        let named: Vec<BlockId> = next.iter().map(|(latch, _)| *latch).collect();
        let each_once = named.iter().enumerate().all(|(at, latch)| !named[..at].contains(latch));
        if !each_once || named.len() != latches.len() || !named.iter().all(|latch| latches.contains(latch)) {
            return Err(CannotEditLoop::BackEdges);
        }
        let ty = self.body_ref().ty_of(init);
        for (latch, value) in next {
            if self.body_ref().ty_of(*value) != ty {
                return Err(CannotEditLoop::TypeMismatch(*value));
            }
            form.reaches_end_of(*value, *latch)?;
        }
        let from_outside = entering.iter().map(|pred| (*pred, init));
        let incoming = self.incoming(header, from_outside.chain(next.iter().copied()));
        Ok(self.edit(|rewriter| rewriter.add_param(header, ty, &incoming)))
    }

    /// Removes the join parameter `param`, every use of it replaced by `with`,
    /// which dominates each use; its entry and back-edge arguments go.
    ///
    /// Refused as `NotALoop`, `NotAJoinParam`, `TypeMismatch(with)`, or
    /// `NotSettled(with)` where `with` does not dominate a use. Cost:
    /// O(blocks + edges + values), and the core's O(the uses replaced and
    /// the edges into the header).
    pub fn remove_join(&mut self, header: BlockId, param: ValueId, with: ValueId) -> Result<(), CannotEditLoop> {
        self.a_loop(header)?;
        if !self.body_ref().block(header).params().contains(&param) {
            return Err(CannotEditLoop::NotAJoinParam(param));
        }
        self.replace_param(param, with)
    }

    /// Adds a value the loop gives its exit: a parameter of the exit block
    /// filled by `value` on every edge into it. Returns the parameter.
    ///
    /// Refused as `NotALoop`, or `NotSettled(value)` where `value` does not
    /// reach the end of a block with an edge into the exit. Cost: O(blocks +
    /// edges + values), and the core's O(the edges into the exit).
    pub fn add_exit(&mut self, header: BlockId, value: ValueId) -> Result<ValueId, CannotEditLoop> {
        let found = self.a_loop(header)?;
        let form = Form::of(self.body_ref());
        let preds = self.shape().predecessors[&found.exit].clone();
        for pred in &preds {
            form.reaches_end_of(value, *pred)?;
        }
        let ty = self.body_ref().ty_of(value);
        let incoming = self.incoming(found.exit, preds.iter().map(|pred| (*pred, value)));
        Ok(self.edit(|rewriter| rewriter.add_param(found.exit, ty, &incoming)))
    }

    /// Removes the exit parameter `param`, every use of it replaced by `with`,
    /// which dominates each use.
    ///
    /// Refused as `NotALoop`, `NotAnExitParam`, `TypeMismatch(with)`, or
    /// `NotSettled(with)` where `with` does not dominate a use. Cost: as
    /// `remove_join`'s, over the edges into the exit.
    pub fn remove_exit(&mut self, header: BlockId, param: ValueId, with: ValueId) -> Result<(), CannotEditLoop> {
        let found = self.a_loop(header)?;
        if !self.body_ref().block(found.exit).params().contains(&param) {
            return Err(CannotEditLoop::NotAnExitParam(param));
        }
        self.replace_param(param, with)
    }

    /// The body and exit of the loop `header` ends in.
    fn a_loop(&self, header: BlockId) -> Result<Found, CannotEditLoop> {
        match self.body_ref().block(header).terminator() {
            Terminator::For { body, exit, .. } | Terminator::While { body, exit, .. } => Ok(Found { body: *body, exit: *exit }),
            _ => Err(CannotEditLoop::NotALoop),
        }
    }

    /// `param` gives way to `with`, checked for type and dominance.
    fn replace_param(&mut self, param: ValueId, with: ValueId) -> Result<(), CannotEditLoop> {
        let body = self.body_ref();
        if body.ty_of(param) != body.ty_of(with) {
            return Err(CannotEditLoop::TypeMismatch(with));
        }
        let form = Form::of(body);
        let sites: Vec<Site> = body.users(param).collect();
        if !sites.iter().all(|site| form.dominators.dominates_site(body, form.points[&with], *site)) {
            return Err(CannotEditLoop::NotSettled(with));
        }
        self.edit(|rewriter| rewriter.replace_param(param, with));
        Ok(())
    }

    /// Each edge into `to` from the named blocks, by its position, with the
    /// value it passes.
    fn incoming(&self, to: BlockId, values: impl Iterator<Item = (BlockId, ValueId)>) -> Vec<(BlockId, usize, ValueId)> {
        let body = self.body_ref();
        let mut incoming = Vec::new();
        for (from, value) in values {
            let edges = body.block(from).terminator().edges();
            let positions = edges.iter().enumerate().filter(|(_, edge)| edge.to == to).map(|(at, _)| at);
            incoming.extend(positions.map(|at| (from, at, value)));
        }
        incoming
    }

    /// The types of the body parameters `head` supplies, in order.
    fn supplied(&mut self, head: &Head) -> Result<Vec<Ty>, CannotEditLoop> {
        let Head::For { source } = head else {
            return Ok(Vec::new());
        };
        let body = self.module.body(self.body);
        let kind = |value: ValueId| self.module.types().kind(body.ty_of(value)).clone();
        let element = match *source {
            ForSource::Range { at, hi } => {
                let at_ty = body.ty_of(at);
                if !matches!(kind(at), TyKind::Int(_)) {
                    return Err(CannotEditLoop::TypeMismatch(at));
                }
                if body.ty_of(hi) != at_ty {
                    return Err(CannotEditLoop::TypeMismatch(hi));
                }
                return Ok(vec![at_ty]);
            }
            ForSource::Slice(slice) | ForSource::SliceMut(slice) => {
                let wanted = match source {
                    ForSource::SliceMut(_) => Mutability::Mut,
                    _ => Mutability::Shared,
                };
                let TyKind::Ref(held, lent) = kind(slice) else {
                    return Err(CannotEditLoop::TypeMismatch(slice));
                };
                let TyKind::Slice(element) = self.module.types().kind(lent).clone() else {
                    return Err(CannotEditLoop::TypeMismatch(slice));
                };
                if held != wanted {
                    return Err(CannotEditLoop::TypeMismatch(slice));
                }
                self.module.types_mut().intern(TyKind::Ref(wanted, element))
            }
            ForSource::Array(array) => match kind(array) {
                TyKind::Array(element, _, _) => element,
                _ => return Err(CannotEditLoop::TypeMismatch(array)),
            },
        };
        let counter = self.module.types_mut().intern(TyKind::Int(IntTy::U64));
        Ok(vec![element, counter])
    }
}

/// A loop's body and exit.
struct Found {
    body: BlockId,
    exit: BlockId,
}

/// What the loop edits check form against: dominance and where each value
/// is defined, of the body as it stands.
struct Form {
    dominators: Dominators,
    points: rustc_hash::FxHashMap<ValueId, DefPoint>,
}

impl Form {
    /// Cost: O(blocks + edges + values).
    fn of(body: &Body) -> Form {
        Form { dominators: Dominators::of(body), points: def_points(body) }
    }

    /// Whether `value` is defined before `header` runs: before the entry,
    /// or in a block other than `header` that dominates it.
    fn settled_before(&self, value: ValueId, header: BlockId) -> Result<(), CannotEditLoop> {
        let settled = match self.points[&value] {
            DefPoint::Entry => true,
            DefPoint::BlockStart(block) | DefPoint::Inst(block, _) => block != header && self.dominators.dominates_block(block, header),
        };
        settled.then_some(()).ok_or(CannotEditLoop::NotSettled(value))
    }

    /// Whether `value` is defined wherever the end of `block` is reached.
    fn reaches_end_of(&self, value: ValueId, block: BlockId) -> Result<(), CannotEditLoop> {
        let reaches = match self.points[&value] {
            DefPoint::Entry => true,
            DefPoint::BlockStart(defined) | DefPoint::Inst(defined, _) => self.dominators.dominates_block(defined, block),
        };
        reaches.then_some(()).ok_or(CannotEditLoop::NotSettled(value))
    }
}
