//! passes' face (ir-core-structure §4): edits over a body that is whole
//! between any two operations, recording what each changed.

use acvus_ast::Span;
use smallvec::SmallVec;

use super::aspect::{Aspect, Changes};
use super::body::{BlockId, Body, Def, Inst, Module};
use super::core::{InstShift, StillUsed};
use super::op::Op;
use crate::ir::ValueId;
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
    pub fn split_block(&mut self, block: BlockId, at: usize) -> BlockId {
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
    pub fn insert(&mut self, block: BlockId, index: usize, op: Op, ty: Ty, span: Span) -> ValueId {
        let module = &mut *self.module;
        let result = module.bodies[self.body.0 as usize].insert_pure(&module.types, block, index, op, ty, span);
        let inserted = &self.body().block(block).insts()[index];
        for aspect in inst_aspects(inserted) {
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
    pub fn remove_inst(&mut self, block: BlockId, index: usize) -> Result<SmallVec<[InstShift; 2]>, StillUsed> {
        let aspects = removal_aspects(self.body(), block, index);
        let shifts = self.body_mut().remove_inst_at(block, index)?;
        for aspect in aspects {
            self.changes.record(aspect);
        }
        Ok(shifts)
    }

    /// Moves an instruction, rethreading an effectful call's token. Cost:
    /// `Body::move_inst`'s.
    ///
    /// # Panics
    /// As `Body::move_inst`.
    pub fn move_inst(&mut self, from: BlockId, from_index: usize, to: BlockId, to_index: usize) -> [InstShift; 2] {
        let aspects = inst_aspects(&self.body().block(from).insts()[from_index]);
        let module = &mut *self.module;
        let shifts = module.bodies[self.body.0 as usize].move_inst(&module.types, from, from_index, to, to_index);
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
            aspects.extend(block.insts().iter().flat_map(inst_aspects));
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

    /// Replaces `slot`'s whole-slot `Take`s and `Assign`s with values. Cost:
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
fn removal_aspects(body: &Body, block: BlockId, index: usize) -> SmallVec<[Aspect; 3]> {
    let inst = &body.block(block).insts()[index];
    let mut aspects = inst_aspects(inst);
    if let Op::TakeEffect { src } = inst.op()
        && let Def::Inst(start_block, start_index) = body.def_of(*src)
    {
        aspects.extend(inst_aspects(&body.block(start_block).insts()[start_index]));
    }
    aspects
}

/// What removing `value` changes, beyond an order token's rethreading.
fn value_removal_aspects(body: &Body, value: ValueId) -> Vec<Aspect> {
    match body.def_of(value) {
        Def::Inst(block, index) => removal_aspects(body, block, index).into_vec(),
        Def::BlockParam(..) => vec![Aspect::Instructions],
        Def::Slot(_) => vec![Aspect::Storage],
        Def::Param(_) | Def::Capture(_) | Def::OrderParam | Def::UnplacedParam(_) => Vec::new(),
    }
}
