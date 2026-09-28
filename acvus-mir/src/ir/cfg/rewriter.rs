//! passes' face (ir-core-structure §4): edits over a body that is whole
//! between any two operations, recording what each changed.

use super::aspect::{Aspect, Changes};
use super::body::{BlockId, Body, Def, Module};
use super::core::StillUsed;
use crate::ir::ValueId;
use crate::ty::{Concrete, Ty, TyKind, Types};

/// Made by `Module::rewrite`, over one body of the module.
pub struct Rewriter<'m> {
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
        let (ty, aspects) = (body.ty_of(value), removal_aspects(body, value));
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

    pub fn finish(self) -> Changes {
        self.changes
    }
}

/// What removing `value` changes, beyond an order token's rethreading.
fn removal_aspects(body: &Body, value: ValueId) -> Vec<Aspect> {
    match body.def_of(value) {
        Def::Inst(block, index) => {
            let op = &body.blocks[block.index()].insts[index].0;
            let mut aspects = vec![Aspect::Instructions];
            if op.touches_storage() {
                aspects.push(Aspect::Storage);
            }
            if op.is_on_order_chain() {
                aspects.push(Aspect::OrderTokens);
            }
            aspects
        }
        Def::BlockParam(..) => vec![Aspect::Instructions],
        Def::Param(_) | Def::Capture(_) | Def::OrderParam => Vec::new(),
    }
}
