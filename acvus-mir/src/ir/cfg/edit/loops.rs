//! The loop edits (compilation §4): a loop is its head, its join, its body
//! and its exit, and each edit changes one of them, so every reshaping of a
//! loop is these edits composed with the body's value, motion and control-flow
//! edits. Promoting a `While` to a `For` is `set_head` and `remove_join` per
//! recurrence; strength reduction is `add_join`; an exit value computed from
//! the count is `remove_exit`.
//!
//! Block parameters are values: changing what fills one replaces its uses
//! with a value the caller names and then removes it, and the editor never
//! guesses which old value equals which new one. An edit checks form: the
//! loop is one (a header ending in a loop terminator), each operand is
//! defined where the edit reads it, types agree, and SSA and loop form hold
//! after it. The caller keeps the meaning (compilation §1): a new head
//! yields the iterations the old did, a join parameter removed is replaced
//! by a value equal to it on every iteration, and so on. The editor cannot
//! check a meaning; it keeps form.

use smallvec::SmallVec;

use super::{
    Asked, CannotRemoveParam, DoesNotDominate, DominanceFacts, Editor, EditsCfg, EditsLoops, LoopEditFacts, LoopFacts, Position,
    ReplacesValues, site_order,
    site_position, supplied_by_an_edge,
};
use crate::ir::ValueId;
use crate::ir::cfg::body::Def;
use crate::ir::cfg::{BlockId, Definition, Head, Rebind, Site};

/// Why a loop edit is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotEditLoop {
    /// The block does not end in a loop terminator.
    NotALoop,
    /// The loop has no preheader, so it is not in F2: the one predecessor
    /// of its header outside it, with no other successor. A header that is
    /// the body's entry has none, since the body's start enters it.
    NoPreheader,
    /// A value is not defined where the edit reads it: a `For`'s operand
    /// defined in or after the header, a `While`'s condition not defined
    /// where the header ends, an argument not defined at the end of its
    /// edge's source, or a parameter the old head supplied, which the edit
    /// removes, named as a replacement.
    NotSettled(ValueId),
    /// The body is entered otherwise than by the head's own edge: by
    /// another edge, the loop's exit edge included, or, as the body's
    /// entry, by the body's start.
    BodyEnteredElsewhere,
    /// The rebinding names `named` parameters where the old head supplied
    /// `supplied`.
    RebindCount { supplied: usize, named: usize },
    /// The rebinding names a supplied parameter past the new head's.
    NoSuchSupplied(usize),
    /// The value is not a join parameter of this loop.
    NotAJoinParam(ValueId),
    /// The value is not an exit parameter of this loop.
    NotAnExitParam(ValueId),
    /// A value's type differs from the one its place has.
    TypeMismatch(ValueId),
    /// A back edge the arguments do not name, or one they name twice, or a
    /// named block with no back edge.
    BackEdges,
    /// A replacement does not dominate a use it would take over.
    DoesNotDominate(DoesNotDominate),
    /// The parameter cannot be removed.
    RemoveParam(CannotRemoveParam),
}

impl<'m, S> Editor<'m, S>
where
    S: EditsLoops + ReplacesValues + EditsCfg,
    S::Needs<'m>: LoopEditFacts,
{
    /// Replaces the loop's head, keeping its join, body, exit and their
    /// arguments; the block-argument form of MLIR's `mergeBlocks`, with
    /// argument values restricted to what exists at the edit. `rebind`
    /// names, for each body parameter the old head supplied, in order, what
    /// takes its uses: the new head's supplied parameter at an index, or an
    /// existing value that dominates every use. The old supplied parameters
    /// then go, and the new head's stand first in the body; they are
    /// returned in order. A rebinding computed from the new parameters,
    /// such as the old counter as the new one plus the range's start, is
    /// this edit with `Rebind::Supplied` followed by inserting the
    /// computation and moving the uses to it with `replace_use`.
    ///
    /// Refused as `NotALoop`; `NoPreheader`; `RebindCount`; `TypeMismatch`
    /// at an operand of a type the new head does not take
    /// (`Head::supplied_kinds`), at an old parameter whose named supplied
    /// parameter differs in type, or at a value of another type than the
    /// parameter it replaces; `NotSettled` at a `For` operand not defined
    /// before the header runs, a `While` condition not defined where the
    /// header ends, or a value the old head supplied;
    /// `BodyEnteredElsewhere`; `NoSuchSupplied`; and
    /// `DoesNotDominate` at the first use, in block order, a value would
    /// take over without dominating it. Cost: O(the head's operands, the
    /// edges into the body and the uses of the old parameters) queries of
    /// the needs, and `Body::set_loop_head`'s.
    pub fn set_head(&mut self, header: BlockId, head: Head, rebind: &[Rebind]) -> Result<Vec<ValueId>, CannotEditLoop> {
        let edited = self.loop_in_f2(header)?;
        let old_head = edited.head;
        let loop_body = edited.body;
        let body = self.body_ref();
        let old_supplied: SmallVec<[ValueId; 2]> =
            body.block(loop_body).params()[..old_head.supplied_params()].iter().copied().collect();
        if rebind.len() != old_supplied.len() {
            return Err(CannotEditLoop::RebindCount { supplied: old_supplied.len(), named: rebind.len() });
        }
        let kinds = head
            .supplied_kinds(|value| body.ty_of(value), self.types())
            .map_err(|unfit| CannotEditLoop::TypeMismatch(unfit.operand))?;
        for operand in head.operands() {
            let settled = match head {
                Head::For { .. } => !self.defined_in(operand, header) && self.dominates(operand, Position::End(header)),
                Head::While { .. } => self.dominates(operand, Position::End(header)),
            };
            if !settled {
                return Err(CannotEditLoop::NotSettled(operand));
            }
        }
        if !self.entered_by_its_head_alone(header, loop_body) {
            return Err(CannotEditLoop::BodyEnteredElsewhere);
        }
        for (param, rebind) in old_supplied.iter().zip(rebind) {
            let body = self.body_ref();
            let param_ty = body.ty_of(*param);
            match *rebind {
                Rebind::Supplied(index) => {
                    let Some(kind) = kinds.get(index) else {
                        return Err(CannotEditLoop::NoSuchSupplied(index));
                    };
                    if self.types().kind(param_ty) != kind {
                        return Err(CannotEditLoop::TypeMismatch(*param));
                    }
                }
                Rebind::Value(value) => {
                    if body.ty_of(value) != param_ty {
                        return Err(CannotEditLoop::TypeMismatch(value));
                    }
                    if old_supplied.contains(&value) {
                        return Err(CannotEditLoop::NotSettled(value));
                    }
                    let mut sites: Vec<Site> = body.users(*param).collect();
                    sites.sort_by_key(|site| site_order(body, *site));
                    if let Some(site) = sites.into_iter().find(|site| !self.dominates(value, site_position(*site))) {
                        return Err(CannotEditLoop::DoesNotDominate(DoesNotDominate { use_at: site }));
                    }
                }
            }
        }
        Ok(self.edit(|rewriter| rewriter.set_loop_head(header, head, rebind)))
    }

    /// Adds a value handed from one iteration to the next: a header parameter
    /// whose argument is `init` on every edge entering the loop from outside
    /// and, on each back edge, the value `next` names for that edge's source
    /// block. A back edge is an edge into the header from a block the header
    /// dominates. Returns the parameter.
    ///
    /// Refused as `NotALoop`; `NoPreheader`; `BackEdges` where
    /// `next` does not name each back edge's source exactly once;
    /// `NotSettled` at an argument not defined at the end of its edge's
    /// source; and `TypeMismatch` at a `next` value of another type than
    /// `init`'s. Cost: O(the edges into the header) queries of the needs,
    /// and `Body::add_block_param`'s.
    ///
    /// # Panics
    /// `init` is an order token, which only the Builder makes.
    pub fn add_join(&mut self, header: BlockId, init: ValueId, next: &[(BlockId, ValueId)]) -> Result<ValueId, CannotEditLoop> {
        self.loop_in_f2(header)?;
        let preds = self.shape().predecessors[&header].clone();
        let start = self.block_start(header);
        let (back, entering): (Vec<BlockId>, Vec<BlockId>) =
            preds.into_iter().partition(|pred| self.position_dominates(start, Site::Term(*pred)));
        let named_once = next.iter().enumerate().all(|(at, (from, _))| !next[..at].iter().any(|(earlier, _)| earlier == from));
        if !named_once || next.len() != back.len() || !next.iter().all(|(from, _)| back.contains(from)) {
            return Err(CannotEditLoop::BackEdges);
        }
        if entering.iter().any(|pred| !self.dominates(init, Position::End(*pred))) {
            return Err(CannotEditLoop::NotSettled(init));
        }
        let ty = self.body_ref().ty_of(init);
        for (latch, value) in next {
            if self.body_ref().ty_of(*value) != ty {
                return Err(CannotEditLoop::TypeMismatch(*value));
            }
            if !self.dominates(*value, Position::End(*latch)) {
                return Err(CannotEditLoop::NotSettled(*value));
            }
        }
        let arguments = entering.iter().map(|pred| (*pred, init)).chain(next.iter().copied());
        let incoming = self.incoming(header, arguments);
        Ok(self.edit(|rewriter| rewriter.add_block_param(header, ty, &incoming)))
    }

    /// Removes the join parameter `param`: its uses take `with`, then it
    /// goes with its entry and back-edge arguments.
    ///
    /// Refused as `NotALoop`; `NoPreheader`; `NotAJoinParam`; `TypeMismatch` at `with`;
    /// `DoesNotDominate` at a use `with` does not dominate; and
    /// `RemoveParam` where the parameter cannot go. The body is unchanged
    /// on a refusal. Cost: `replace_uses`' and `remove_param`'s.
    pub fn remove_join(&mut self, header: BlockId, param: ValueId, with: ValueId) -> Result<(), CannotEditLoop> {
        self.loop_in_f2(header)?;
        if !self.body_ref().block(header).params().contains(&param) {
            return Err(CannotEditLoop::NotAJoinParam(param));
        }
        self.replace_and_remove(param, with)
    }

    /// Adds a value the loop gives its exit: a parameter of the exit block
    /// filled by `value` on every edge into it. Returns the parameter.
    ///
    /// Refused as `NotALoop`; `NoPreheader`; and `NotSettled`
    /// where `value` is not defined at the end of a block with an edge into
    /// the exit. Cost: O(the edges into the exit) queries of the needs, and
    /// `Body::add_block_param`'s.
    ///
    /// # Panics
    /// `value` is an order token, which only the Builder makes.
    pub fn add_exit(&mut self, header: BlockId, value: ValueId) -> Result<ValueId, CannotEditLoop> {
        let exit = self.loop_in_f2(header)?.exit;
        let preds = self.shape().predecessors[&exit].clone();
        if preds.iter().any(|pred| !self.dominates(value, Position::End(*pred))) {
            return Err(CannotEditLoop::NotSettled(value));
        }
        let ty = self.body_ref().ty_of(value);
        let incoming = self.incoming(exit, preds.iter().map(|pred| (*pred, value)));
        Ok(self.edit(|rewriter| rewriter.add_block_param(exit, ty, &incoming)))
    }

    /// Removes the exit parameter `param`: its uses take `with`, then it
    /// goes with its argument on every edge into the exit.
    ///
    /// Refused as `NotALoop`; `NoPreheader`; `NotAnExitParam`; and as `remove_join` is
    /// for `with` and the parameter. The body is unchanged on a refusal.
    /// Cost: `replace_uses`' and `remove_param`'s.
    pub fn remove_exit(&mut self, header: BlockId, param: ValueId, with: ValueId) -> Result<(), CannotEditLoop> {
        let exit = self.loop_in_f2(header)?.exit;
        if !self.body_ref().block(exit).params().contains(&param) {
            return Err(CannotEditLoop::NotAnExitParam(param));
        }
        self.replace_and_remove(param, with)
    }

    /// The loop `header` ends in, which the loop edits require in F2.
    fn loop_in_f2(&mut self, header: BlockId) -> Result<EditedLoop, CannotEditLoop> {
        let parts = self.body_ref().block(header).terminator().as_loop().ok_or(CannotEditLoop::NotALoop)?;
        let edited = EditedLoop { head: parts.head, body: parts.body, exit: parts.exit };
        let asked = Asked::new(self.module, self.body);
        if self.needs.preheader(asked, header).is_none() {
            return Err(CannotEditLoop::NoPreheader);
        }
        Ok(edited)
    }

    /// `replace_uses` and then `remove_param`, refused before either where
    /// the types differ or the parameter is one a loop supplies, so a
    /// refusal leaves the body as it was.
    fn replace_and_remove(&mut self, param: ValueId, with: ValueId) -> Result<(), CannotEditLoop> {
        let body = self.body_ref();
        if body.ty_of(param) != body.ty_of(with) {
            return Err(CannotEditLoop::TypeMismatch(with));
        }
        let Def::BlockParam(block, index) = body.def_of(param) else {
            panic!("{param:?} is a join or exit parameter, so a block parameter")
        };
        if supplied_by_an_edge(body, block, index) {
            return Err(CannotEditLoop::RemoveParam(CannotRemoveParam::Supplied));
        }
        self.replace_uses(param, with).map_err(CannotEditLoop::DoesNotDominate)?;
        self.remove_param(param).map_err(CannotEditLoop::RemoveParam)
    }

    /// Whether every edge into `loop_body` is `header`'s edge into its body,
    /// and `loop_body` is not the entry, which the body's start enters.
    fn entered_by_its_head_alone(&mut self, header: BlockId, loop_body: BlockId) -> bool {
        let preds = self.shape().predecessors[&loop_body].clone();
        let body = self.body_ref();
        let edges_in: usize = preds
            .iter()
            .map(|from| body.block(*from).terminator().edges().iter().filter(|edge| edge.to == loop_body).count())
            .sum();
        loop_body != body.entry() && preds[..] == [header] && edges_in == 1
    }

    /// Whether `value` is defined in `block`: a parameter of it or the
    /// result of one of its instructions.
    fn defined_in(&self, value: ValueId, block: BlockId) -> bool {
        let body = self.body_ref();
        match body.definition(value) {
            Definition::BlockParam(defined, _) => defined == block,
            Definition::Inst(inst) => body.block_of(inst) == block,
            Definition::Param(_) | Definition::Capture(_) | Definition::OrderParam | Definition::Slot(_) => false,
        }
    }

    /// The first position of `block`: before its first instruction, or at
    /// its end where it has none.
    fn block_start(&self, block: BlockId) -> Position {
        self.first_inst(block).map_or(Position::End(block), Position::Before)
    }

    fn dominates(&mut self, value: ValueId, at: Position) -> bool {
        let asked = Asked::new(self.module, self.body);
        self.needs.dominates(asked, value, at)
    }

    fn position_dominates(&mut self, at: Position, site: Site) -> bool {
        let asked = Asked::new(self.module, self.body);
        self.needs.position_dominates(asked, at, site)
    }

    /// Each edge into `to` from the named blocks, by its position among its
    /// source's edges, with the value it passes.
    fn incoming(&self, to: BlockId, arguments: impl Iterator<Item = (BlockId, ValueId)>) -> Vec<(BlockId, usize, ValueId)> {
        let body = self.body_ref();
        let mut incoming = Vec::new();
        for (from, value) in arguments {
            let edges = body.block(from).terminator().edges();
            let positions = edges.iter().enumerate().filter(|(_, edge)| edge.to == to).map(|(position, _)| position);
            incoming.extend(positions.map(|position| (from, position, value)));
        }
        incoming
    }
}

/// What the loop edits read of the loop they edit.
struct EditedLoop {
    head: Head,
    body: BlockId,
    exit: BlockId,
}
