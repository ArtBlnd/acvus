//! passes' face (ir-core-structure §4): edits over a body that is whole
//! between any two operations, recording what each changed.

use acvus_ast::Span;
use smallvec::SmallVec;

use super::aspect::{Aspect, Changes};
use super::body::{BlockId, Body, BodyId, Def, Inst, InstId, Module, Site};
use super::core::{InstShift, StillUsed};
use super::op::Op;
use super::order::At;
use crate::ir::{SwitchKey, ValueId};
use crate::ty::{Concrete, Ty, TyKind, Types};

/// Made by `Module::rewrite`, over one body of the module.
pub(in crate::ir::cfg) struct Rewriter<'m> {
    module: &'m mut Module,
    body: super::BodyId,
    changes: Changes,
}

impl<'m> Rewriter<'m> {
    pub(super) fn new(module: &'m mut Module, body: super::BodyId) -> Rewriter<'m> {
        Rewriter { module, body, changes: Changes::default() }
    }

    /// The body as the edits so far left it.
    pub fn body(&self) -> &Body {
        self.module.body(self.body)
    }

    pub fn types(&self) -> &Types<Concrete> {
        &self.module.types
    }

    fn body_mut(&mut self) -> &mut Body {
        &mut self.module.bodies[self.body.0 as usize]
    }

    fn is_order(&self, ty: Ty) -> bool {
        matches!(self.module.types.kind(ty), TyKind::Order)
    }

    /// Every use of `of` becomes a use of `with`, which has its type. Cost:
    /// `Body::replace_all_uses`'s, amortized O(1) per use changed in the
    /// def-use record.
    ///
    /// # Panics
    /// Either is not a value of the body, or their types differ.
    pub fn replace_all_uses(&mut self, of: ValueId, with: ValueId) {
        self.body_mut().replace_all_uses(of, with);
        self.changes.record(Aspect::Instructions);
        if self.is_order(self.body().ty_of(of)) {
            self.changes.record(Aspect::OrderTokens);
        }
    }

    /// Removes `value` and what defines it; refused while it has uses. An
    /// effectful instruction's order token is rethreaded around it, and a
    /// split call's taking goes with its start. Cost: `Body::remove_value`'s.
    ///
    /// # Panics
    /// `value` is not a value of the body, is one of its parameters,
    /// captures or order parameter, or is a block parameter a loop's
    /// terminator supplies.
    pub fn remove(&mut self, value: ValueId) -> Result<(), StillUsed> {
        let body = self.body();
        let (ty, aspects) = (body.ty_of(value), value_removal_aspects(body, value));
        let is_order = self.is_order(ty);
        self.body_mut().remove_value(value)?;
        for aspect in aspects {
            self.changes.record(aspect);
        }
        if is_order {
            self.changes.record(Aspect::OrderTokens);
        }
        Ok(())
    }

    /// Splits `block` before its instruction `at`; the new block holds the
    /// rest and the terminator, and `block` keeps its id, parameters and
    /// every block naming it. Cost: `Body::split_block`'s.
    ///
    /// # Panics
    /// `at` is past the block's end.
    pub fn split_block(&mut self, block: BlockId, at: Option<InstId>) -> BlockId {
        let tail = self.body_mut().split_block(block, at);
        self.changes.record(Aspect::CfgShape);
        tail
    }

    /// Makes the call whose result is `call` a `Spawn` and its `TakeEffect`
    /// (effects §6). Cost: `Body::split_call`'s.
    ///
    /// # Panics
    /// `call` is not the result of a `FunctionCall` whose callee's type
    /// states that it may be split.
    pub fn split_call(&mut self, call: ValueId) {
        let module = &mut *self.module;
        module.bodies[self.body.0 as usize].split_call(&mut module.types, call);
        self.changes.record(Aspect::Instructions);
        if self.body().order_param().is_some() {
            self.changes.record(Aspect::OrderTokens);
        }
    }

    /// Every edge into `from`, every `Diamond` join and loop stage and exit
    /// naming it, and the entry if it is `from`, name `to` instead; `from`'s
    /// parameters agree with `to`'s, so every edge still agrees with its
    /// target. Cost: O(the body's terminators).
    ///
    /// # Panics
    /// Their parameters differ in count or type.
    pub fn retarget(&mut self, from: BlockId, to: BlockId) {
        self.body_mut().retarget(from, to);
        self.changes.record(Aspect::CfgShape);
    }

    /// Inserts an instruction of `op` that takes no order token at `index`
    /// of `block`, and returns its result, typed `ty`. Cost:
    /// `Body::insert_pure`'s.
    ///
    /// # Panics
    /// As `Body::insert_pure`.
    pub fn insert(&mut self, at: At, op: Op, ty: Ty, span: Span) -> ValueId {
        let module = &mut *self.module;
        let result = module.bodies[self.body.0 as usize].insert_pure(&module.types, at, op, ty, span);
        let Def::Inst(inserted) = self.body().def_of(result) else {
            panic!("an inserted instruction defines its result")
        };
        for aspect in inst_aspects(self.body().inst(inserted)) {
            self.changes.record(aspect);
        }
        result
    }

    /// Removes the instruction at `index` of `block`, refused while its
    /// result has uses; the token is rethreaded around an effectful call,
    /// and a split call's taking goes with its start. It returns the
    /// positions taken out. Cost: `Body::remove_inst_at`'s.
    ///
    /// # Panics
    /// As `Body::remove_inst_at`.
    pub fn remove_inst(&mut self, id: InstId) -> Result<SmallVec<[InstShift; 2]>, StillUsed> {
        let aspects = removal_aspects(self.body(), id);
        let shifts = self.body_mut().remove_inst_at(id)?;
        for aspect in aspects {
            self.changes.record(aspect);
        }
        Ok(shifts)
    }

    /// Moves an instruction with its operands and results, order tokens
    /// included. Cost: `Body::move_inst`'s.
    ///
    /// # Panics
    /// As `Body::move_inst`.
    pub fn move_inst(&mut self, id: InstId, at: At) -> InstShift {
        let aspects = inst_aspects(self.body().inst(id));
        let shifts = self.body_mut().move_inst(id, at);
        for aspect in aspects {
            self.changes.record(aspect);
        }
        shifts
    }

    /// `block`'s two-way branch becomes a jump into one arm. Cost:
    /// `Body::take_arm`'s.
    ///
    /// # Panics
    /// As `Body::take_arm`.
    pub fn take_arm(&mut self, block: BlockId, then: bool) {
        self.body_mut().take_arm(block, then);
        self.changes.record(Aspect::CfgShape);
    }

    /// Removes the blocks `keep` does not keep, and renumbers the rest.
    /// Cost: `Body::remove_blocks`'s.
    ///
    /// # Panics
    /// As `Body::remove_blocks`.
    pub fn remove_blocks(&mut self, keep: &[bool]) -> Vec<Option<BlockId>> {
        let body = self.body();
        let mut aspects = vec![Aspect::CfgShape];
        for (block, _) in body.blocks().filter(|(block, _)| !keep[block.index()]) {
            let block = body.block(block);
            aspects.extend(block.insts().flat_map(inst_aspects));
            if !block.params().is_empty() {
                aspects.push(Aspect::Instructions);
            }
            if block.params().iter().any(|param| body.is_token(*param)) {
                aspects.push(Aspect::OrderTokens);
            }
        }
        let renamed = self.body_mut().remove_blocks(keep);
        for aspect in aspects {
            self.changes.record(aspect);
        }
        renamed
    }

    /// Replaces `slot`'s whole-slot reads and writes with values. Cost:
    /// `Body::promote_slot`'s.
    ///
    /// # Panics
    /// As `Body::promote_slot`.
    pub fn promote(&mut self, slot: ValueId) -> Vec<InstShift> {
        let module = &mut *self.module;
        let shifts = module.bodies[self.body.0 as usize].promote_slot(&module.types, slot);
        self.changes.record(Aspect::Instructions);
        self.changes.record(Aspect::Storage);
        shifts
    }

    /// `block`'s `Switch` becomes a jump into the arm `key` selects. Cost:
    /// `Body::take_case`'s.
    ///
    /// # Panics
    /// As `Body::take_case`.
    pub fn take_case(&mut self, block: BlockId, key: &SwitchKey) {
        self.body_mut().take_case(block, key);
        self.changes.record(Aspect::CfgShape);
    }

    /// A new block on the edge at `position` of `from`. Cost:
    /// `Body::split_edges`' for one edge.
    ///
    /// # Panics
    /// As `Body::split_edges`.
    pub fn split_edge(&mut self, from: BlockId, position: usize) -> BlockId {
        self.split_edges(&[(from, position)])
    }

    /// A new block on every edge of `edges`, all into one block. Cost:
    /// `Body::split_edges`'.
    ///
    /// # Panics
    /// As `Body::split_edges`.
    pub fn split_edges(&mut self, edges: &[(BlockId, usize)]) -> BlockId {
        let middle = self.body_mut().split_edges(edges);
        self.changes.record(Aspect::CfgShape);
        let body = self.body();
        let params = body.block(middle).params();
        let (any, tokens) = (!params.is_empty(), params.iter().any(|param| body.is_token(*param)));
        if any {
            self.changes.record(Aspect::Instructions);
        }
        if tokens {
            self.changes.record(Aspect::OrderTokens);
        }
        middle
    }

    /// `block` joins its one predecessor `pred`. The moved instructions keep
    /// what they read and write, and stand in another block: the shape and
    /// the instructions' places change, and the order tokens where a
    /// parameter that goes is one. Cost: `Body::merge_into_predecessor`'s.
    ///
    /// # Panics
    /// As `Body::merge_into_predecessor`.
    pub fn merge_into_predecessor(&mut self, block: BlockId, pred: BlockId) -> Vec<Option<BlockId>> {
        let body = self.body();
        let tokens = body.block(block).params().iter().any(|param| body.is_token(*param));
        let renamed = self.body_mut().merge_into_predecessor(block, pred);
        self.changes.record(Aspect::CfgShape);
        self.changes.record(Aspect::Instructions);
        if tokens {
            self.changes.record(Aspect::OrderTokens);
        }
        renamed
    }

    /// The operands at `site` that are `of` become `with`. Cost:
    /// `Body::replace_uses_at`'s.
    ///
    /// # Panics
    /// As `Body::replace_uses_at`.
    pub fn replace_uses_at(&mut self, site: Site, of: ValueId, with: ValueId) {
        self.body_mut().replace_uses_at(site, of, with);
        self.changes.record(Aspect::Instructions);
        if self.is_order(self.body().ty_of(of)) {
            self.changes.record(Aspect::OrderTokens);
        }
    }

    /// The operand at `index` of `site`, which is `of`, becomes `with`.
    /// Cost: `Body::replace_operand_at`'s.
    ///
    /// # Panics
    /// As `Body::replace_operand_at`.
    pub fn replace_operand_at(&mut self, site: Site, index: usize, of: ValueId, with: ValueId) {
        self.body_mut().replace_operand_at(site, index, of, with);
        self.changes.record(Aspect::Instructions);
        if self.is_order(self.body().ty_of(of)) {
            self.changes.record(Aspect::OrderTokens);
        }
    }

    /// The id of `kind` in the module's one table. Cost: O(kind).
    ///
    /// # Panics
    /// As `Types::intern`.
    pub fn intern(&mut self, kind: TyKind<Concrete>) -> Ty {
        self.changes.record(Aspect::Types);
        self.module.types.intern(kind)
    }

    /// Two adjacent effects on one chain trade places. Cost:
    /// `Body::swap_effects`'s.
    ///
    /// # Panics
    /// As `Body::swap_effects`.
    pub fn swap_effects(&mut self, first: InstId, second: InstId) -> InstShift {
        let shifts = self.body_mut().swap_effects(first, second);
        self.changes.record(Aspect::Instructions);
        self.changes.record(Aspect::OrderTokens);
        shifts
    }

    /// `slot` becomes one slot per field. Cost: `Body::split_slot`'s.
    ///
    /// # Panics
    /// As `Body::split_slot`.
    pub fn split_slot(&mut self, slot: ValueId) -> Vec<ValueId> {
        let module = &mut *self.module;
        let parts = module.bodies[self.body.0 as usize].split_slot(&module.types, slot);
        self.changes.record(Aspect::Instructions);
        self.changes.record(Aspect::Storage);
        parts
    }

    /// Inserts a `Drop` of `value`. Cost: `Body::insert_drop`'s.
    ///
    /// # Panics
    /// As `Body::insert_drop`.
    pub fn insert_drop(&mut self, at: At, value: ValueId, span: Span) {
        self.body_mut().insert_drop(at, value, span);
        self.changes.record(Aspect::Instructions);
    }

    /// The call at `index` of `block` becomes a copy of `callee`'s body,
    /// with `captures` for its captures; it returns the continuation. Cost:
    /// `Body::inline_call`'s.
    ///
    /// # Panics
    /// `callee` is the body being rewritten, or as `Body::inline_call`.
    pub fn inline_call(&mut self, call: InstId, callee: BodyId, captures: &[ValueId]) -> BlockId {
        self.module.assert_body(callee);
        let (caller, callee) = (self.body.0 as usize, callee.0 as usize);
        assert_ne!(caller, callee, "a body is not inlined into itself");
        let (caller_body, callee_body) = match caller < callee {
            true => {
                let (low, high) = self.module.bodies.split_at_mut(callee);
                (&mut low[caller], &high[0])
            }
            false => {
                let (low, high) = self.module.bodies.split_at_mut(caller);
                (&mut high[0], &low[callee])
            }
        };
        let ordered = caller_body.inst(call).order().is_some();
        let continuation = caller_body.inline_call(call, callee_body, captures);
        let body = self.body();
        let mut aspects = vec![Aspect::CfgShape, Aspect::Instructions];
        let copied = (continuation.index()..body.blocks().count()).map(BlockId::of_index);
        aspects.extend(copied.flat_map(|block| body.block(block).insts().flat_map(inst_aspects).collect::<Vec<_>>()));
        if ordered {
            aspects.push(Aspect::OrderTokens);
        }
        for aspect in aspects {
            self.changes.record(aspect);
        }
        continuation
    }

    /// The checked bound of the `Index` or `IndexSet` at `index` of `block`
    /// becomes proven. Cost: O(1).
    ///
    /// # Panics
    /// As `Body::prove_bound`.
    pub fn prove_bound(&mut self, id: InstId) {
        self.body_mut().prove_bound(id);
        self.changes.record(Aspect::Instructions);
    }

    pub fn finish(self) -> Changes {
        self.changes
    }
}

/// What an instruction's insertion, removal or move changes.
fn inst_aspects(inst: &Inst) -> SmallVec<[Aspect; 3]> {
    let mut aspects = SmallVec::new();
    aspects.push(Aspect::Instructions);
    if inst.op().touches_storage() {
        aspects.push(Aspect::Storage);
    }
    if inst.is_on_order_chain() {
        aspects.push(Aspect::OrderTokens);
    }
    aspects
}

/// What removing the instruction at `index` of `block` changes: its own
/// aspects, and a split call's start's where it is a taking.
fn removal_aspects(body: &Body, id: InstId) -> SmallVec<[Aspect; 3]> {
    let inst = body.inst(id);
    let mut aspects = inst_aspects(inst);
    if let Op::TakeEffect { src } = inst.op()
        && let Def::Inst(start) = body.def_of(*src)
    {
        aspects.extend(inst_aspects(body.inst(start)));
    }
    aspects
}

/// What removing `value` changes, beyond an order token's rethreading.
fn value_removal_aspects(body: &Body, value: ValueId) -> Vec<Aspect> {
    match body.def_of(value) {
        Def::Inst(id) => removal_aspects(body, id).into_vec(),
        Def::BlockParam(..) => vec![Aspect::Instructions],
        Def::Slot { .. } => vec![Aspect::Storage],
        Def::Param(_) | Def::Capture(_) | Def::OrderParam | Def::UnplacedParam(_) => Vec::new(),
    }
}
