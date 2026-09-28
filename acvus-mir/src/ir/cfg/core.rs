//! The invariants (ir-core-structure §2), in one place. Both faces call
//! these and nothing else writes a body. Each operation states its cost.
//!
//! - A value is made with its type, interned in the module's one table; a
//!   reference to a reference is refused (types §4).
//! - Def-use is kept for every operand; a value with uses is removed only
//!   with its replacement. Amortized O(1) per use changed.
//! - A block's parameters agree with every edge's arguments.
//! - The order token chain (effects §2-3) is rethreaded by every insertion,
//!   removal or move of an effectful instruction, O(length rethreaded); an
//!   `anyorder` scope gives its entry token inside and merges on close and
//!   on every edge that leaves it.
//! - `Spawn` and `TakeEffect` are made only together (effects §6).
//! - A `Diamond`'s join and a `For`'s or `While`'s stages and exit follow
//!   every retarget and block split.

use acvus_ast::Span;
use acvus_utils::LocalFactory;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::body::{Block, BlockId, Body, Def, Edge, Site, Terminator, UseSites, ValueDef};
use super::op::Op;
use crate::ir::{DebugInfo, ValueId, fn_effect};
use crate::ty::{Concrete, SplitCost, Ty, TyKind, Types};

/// A value that still has uses, removed without a replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StillUsed {
    pub value: ValueId,
}

impl Body {
    /// A body with no block and no value yet; its entry is the first block
    /// added to it.
    pub(super) fn empty() -> Body {
        Body {
            blocks: Vec::new(),
            entry: BlockId::of_index(0),
            params: Vec::new(),
            captures: Vec::new(),
            order_param: None,
            values: FxHashMap::default(),
            uses: FxHashMap::default(),
            value_ids: LocalFactory::new(),
            debug: DebugInfo::new(),
        }
    }

    /// A new value of type `ty`, defined at `def`. Cost: O(1).
    ///
    /// # Panics
    /// `ty` is an id of another table than `types`, or a reference whose
    /// target is a reference (types §4).
    pub(super) fn make_value(&mut self, types: &Types<Concrete>, ty: Ty, def: Def) -> ValueId {
        if let TyKind::Ref(_, target) = types.kind(ty) {
            assert!(
                !matches!(types.kind(target.ty()), TyKind::Ref(..)),
                "a reference's target is never a reference (types §4): {ty:?}"
            );
        }
        let value = self.value_ids.next();
        self.values.insert(value, ValueDef { ty, def });
        self.uses.insert(value, UseSites::default());
        value
    }

    fn value_def_mut(&mut self, value: ValueId) -> &mut ValueDef {
        self.values
            .get_mut(&value)
            .unwrap_or_else(|| panic!("{value:?} is not a value of this body"))
    }

    fn use_sites_mut(&mut self, value: ValueId) -> &mut UseSites {
        self.uses
            .get_mut(&value)
            .unwrap_or_else(|| panic!("{value:?} is not a value of this body"))
    }

    /// # Panics
    /// `value` is not a value of this body.
    pub(super) fn def_of(&self, value: ValueId) -> Def {
        self.values
            .get(&value)
            .unwrap_or_else(|| panic!("{value:?} is not a value of this body"))
            .def
    }

    /// Records `op` as the instruction at `index` of `block`: it defines
    /// its results there and uses its operands there. Cost: O(operands and
    /// results of `op`).
    ///
    /// # Panics
    /// A result or an operand of `op` is not a value of this body.
    pub(super) fn record_inst(&mut self, block: BlockId, index: usize, op: &Op) {
        for value in op.defs() {
            self.value_def_mut(value).def = Def::Inst(block, index);
        }
        let site = Site::Inst(block, index);
        op.for_each_operand(|value| self.use_sites_mut(value).add(site, 1));
    }


    /// Records the uses of `block`'s terminator. Cost: O(its operands).
    ///
    /// # Panics
    /// An operand is not a value of this body.
    pub(super) fn record_terminator(&mut self, block: BlockId, terminator: &Terminator) {
        let site = Site::Term(block);
        terminator.for_each_operand(|value| self.use_sites_mut(value).add(site, 1));
    }

    /// Moves the record of every instruction of `block` from `from` on by
    /// one position, up or down, ahead of the insertion or removal that
    /// moves the instructions themselves. Cost: O(operands and results of
    /// those instructions).
    fn shift_insts(&mut self, block: BlockId, from: usize, shift: Shift) {
        let count = self.blocks[block.index()].insts.len();
        let indices: Vec<usize> = match shift {
            Shift::Up => (from..count).rev().collect(),
            Shift::Down => (from..count).collect(),
        };
        for index in indices {
            let to = match shift {
                Shift::Up => index + 1,
                Shift::Down => index - 1,
            };
            let op = &self.blocks[block.index()].insts[index].0;
            let (defs, operands) = (op.defs(), operands_of(op));
            for value in defs {
                self.value_def_mut(value).def = Def::Inst(block, to);
            }
            for value in operands {
                self.use_sites_mut(value)
                    .move_site(Site::Inst(block, index), Site::Inst(block, to));
            }
        }
    }

    /// Inserts `op` at `index` of `block`; the instructions from `index` on
    /// move down by one. Every result of `op` is already a value of this
    /// body and is defined here from now on. Cost: O(operands and results
    /// of the instructions from `index` on).
    pub(super) fn insert_inst(&mut self, block: BlockId, index: usize, op: Op, span: Span) {
        self.shift_insts(block, index, Shift::Up);
        self.record_inst(block, index, &op);
        self.blocks[block.index()].insts.insert(index, (op, span));
    }

    /// Inserts at `index` of `block` the instruction `build` makes from new
    /// values of `result_tys`, which it defines, and returns those values.
    /// Cost: O(|result_tys|) and `insert_inst`'s.
    ///
    /// # Panics
    /// The instruction does not define exactly the values made for it.
    pub(super) fn insert_new_inst(
        &mut self,
        types: &Types<Concrete>,
        block: BlockId,
        index: usize,
        result_tys: &[Ty],
        build: impl FnOnce(&[ValueId]) -> Op,
        span: Span,
    ) -> SmallVec<[ValueId; 2]> {
        let results: SmallVec<[ValueId; 2]> = result_tys
            .iter()
            .map(|ty| self.make_value(types, *ty, Def::Inst(block, index)))
            .collect();
        let op = build(&results);
        assert_eq!(
            op.defs(),
            results,
            "an instruction defines exactly the values made for it"
        );
        self.insert_inst(block, index, op, span);
        results
    }

    /// Takes the instruction at `index` of `block` out, with its uses; the
    /// instructions after it move up by one. The values it defined keep a
    /// stale definition: the caller defines them again or drops them.
    /// Cost: O(operands and results of the instructions from `index` on).
    fn take_inst(&mut self, block: BlockId, index: usize) -> (Op, Span) {
        for value in operands_of(&self.blocks[block.index()].insts[index].0) {
            self.use_sites_mut(value).remove_one(Site::Inst(block, index));
        }
        self.shift_insts(block, index + 1, Shift::Down);
        self.blocks[block.index()].insts.remove(index)
    }

    fn drop_value(&mut self, value: ValueId) {
        self.values.remove(&value);
        self.uses.remove(&value);
        self.debug.val_origins.remove(&value);
    }

    /// Every use of `of` becomes a use of `with`. Cost: amortized O(1) per
    /// use changed in the def-use record, plus a walk of the operands of
    /// every instruction and terminator that used `of`.
    ///
    /// # Panics
    /// Either is not a value of this body, or their types differ.
    pub(super) fn replace_all_uses(&mut self, of: ValueId, with: ValueId) {
        let (of_ty, with_ty) = (self.ty_of(of), self.ty_of(with));
        assert!(
            of_ty == with_ty,
            "{of:?} of {of_ty:?} is replaced only by a value of its type, not {with:?} of {with_ty:?}"
        );
        if of == with {
            return;
        }
        let sites = std::mem::take(self.use_sites_mut(of));
        for (site, count) in sites.into_sites() {
            let swap = |value: &mut ValueId| {
                if *value == of {
                    *value = with;
                }
            };
            match site {
                Site::Inst(block, index) => {
                    self.blocks[block.index()].insts[index].0.for_each_operand_mut(swap)
                }
                Site::Term(block) => self.blocks[block.index()].terminator.for_each_operand_mut(swap),
            }
            self.use_sites_mut(with).add(site, count);
        }
    }

    /// Removes `value` and what defines it, refused while it has uses.
    ///
    /// - A value an instruction defines: the instruction goes, and every
    ///   value it defines must be unused but the order token it yields,
    ///   whose uses are handed the token it consumed (effects §3). The
    ///   taking of a split call goes with its start (effects §6).
    /// - A block parameter: it goes with its argument on every edge.
    ///
    /// Cost: an instruction, O(the block's instructions after it); a block
    /// parameter, O(the body's edges).
    ///
    /// # Panics
    /// `value` is not a value of this body; it is a parameter, a capture or
    /// the order parameter, which are the body's interface; or it is a
    /// block parameter a loop's terminator supplies.
    pub(super) fn remove_value(&mut self, value: ValueId) -> Result<(), StillUsed> {
        match self.def_of(value) {
            Def::Inst(block, index) => self.remove_inst(block, index, value),
            Def::BlockParam(block, index) => {
                self.refuse_used(value)?;
                self.remove_block_param(block, index);
                Ok(())
            }
            Def::Param(_) | Def::Capture(_) | Def::OrderParam => panic!(
                "{value:?} is part of the body's interface; the core does not remove it"
            ),
        }
    }

    fn refuse_used(&self, value: ValueId) -> Result<(), StillUsed> {
        match self.use_count(value) {
            0 => Ok(()),
            _ => Err(StillUsed { value }),
        }
    }

    fn remove_inst(&mut self, block: BlockId, index: usize, named: ValueId) -> Result<(), StillUsed> {
        self.refuse_used(named)?;
        let op = &self.blocks[block.index()].insts[index].0;
        match *op {
            Op::TakeEffect { dst, src, order } => {
                self.refuse_used(dst)?;
                let Def::Inst(spawn_block, spawn_index) = self.def_of(src) else {
                    panic!("a taking's handle is a spawn's result (effects §6)")
                };
                let Op::Spawn { order: spawn_order, .. } = self.blocks[spawn_block.index()].insts[spawn_index].0
                else {
                    panic!("a taking's handle is a spawn's result (effects §6)")
                };
                match (spawn_order, order) {
                    (Some(before), Some(after)) => self.replace_all_uses(after, before),
                    (None, None) => {}
                    (Some(_), None) | (None, Some(_)) => {
                        panic!("a split call's start and taking are ordered together (effects §6)")
                    }
                }
                self.take_inst(block, index);
                let spawn_index =
                    if spawn_block == block && spawn_index > index { spawn_index - 1 } else { spawn_index };
                let (spawn, _) = self.take_inst(spawn_block, spawn_index);
                for defined in spawn.defs().into_iter().chain([dst, src]).chain(order) {
                    self.drop_value(defined);
                }
                Ok(())
            }
            Op::FunctionCall { dst, order: Some(edge), .. } => {
                self.refuse_used(dst)?;
                self.replace_all_uses(edge.after, edge.before);
                self.take_inst(block, index);
                self.drop_value(dst);
                self.drop_value(edge.after);
                Ok(())
            }
            _ => {
                let defs = op.defs();
                for defined in &defs {
                    self.refuse_used(*defined)?;
                }
                self.take_inst(block, index);
                for defined in defs {
                    self.drop_value(defined);
                }
                Ok(())
            }
        }
    }

    fn remove_block_param(&mut self, block: BlockId, index: usize) {
        for pred in 0..self.blocks.len() {
            let pred = BlockId::of_index(pred);
            let mut dropped: SmallVec<[ValueId; 2]> = SmallVec::new();
            self.blocks[pred.index()]
                .terminator
                .for_each_edge_mut(|to, supplied, args| {
                    if to != block {
                        return;
                    }
                    let Some(args) = args.filter(|_| index >= supplied) else {
                        panic!("{pred:?}'s terminator supplies parameter {index} of {block:?} itself")
                    };
                    dropped.push(args.remove(index - supplied));
                });
            for arg in dropped {
                self.use_sites_mut(arg).remove_one(Site::Term(pred));
            }
        }
        let params = &mut self.blocks[block.index()].params;
        let removed = params.remove(index);
        let later: SmallVec<[ValueId; 4]> = params[index..].iter().copied().collect();
        for (offset, param) in later.into_iter().enumerate() {
            self.value_def_mut(param).def = Def::BlockParam(block, index + offset);
        }
        self.drop_value(removed);
    }

    /// Appends `block` to the body. Cost: O(1).
    pub(super) fn new_block(&mut self, block: Block) -> BlockId {
        let id = BlockId::of_index(self.blocks.len());
        self.blocks.push(block);
        id
    }

    /// Checks that `params`, the parameters of `edge`'s target, agree in
    /// count with what the edge supplies and passes, and in type with its
    /// arguments. Cost: O(parameters).
    ///
    /// # Panics
    /// They disagree.
    pub(super) fn assert_edge_agrees(&self, from: BlockId, params: &[ValueId], edge: &Edge<'_>) {
        assert_eq!(
            params.len(),
            edge.supplied + edge.args.len(),
            "the edge {from:?} -> {:?} supplies {} and passes {} arguments to {} parameters",
            edge.to,
            edge.supplied,
            edge.args.len(),
            params.len()
        );
        for (param, arg) in params[edge.supplied..].iter().zip(edge.args) {
            let (param_ty, arg_ty) = (self.ty_of(*param), self.ty_of(*arg));
            assert!(
                param_ty == arg_ty,
                "the edge {from:?} -> {:?} passes {arg:?} of {arg_ty:?} to {param:?} of {param_ty:?}",
                edge.to
            );
        }
    }

    /// The instructions of `block` from `at` on, and its terminator, move
    /// to a new block with no parameters, and `block` jumps to it. `block`
    /// keeps its id, so every edge into it, a `Diamond`'s join and a loop's
    /// stages and exit that name it still do. Cost: O(operands and results
    /// of what moves).
    ///
    /// # Panics
    /// `at` is past the block's end.
    pub(super) fn split_block(&mut self, block: BlockId, at: usize) -> BlockId {
        let tail_id = BlockId::of_index(self.blocks.len());
        let head = &mut self.blocks[block.index()];
        assert!(at <= head.insts.len(), "{block:?} has no position {at}");
        let insts = head.insts.split_off(at);
        let terminator = std::mem::replace(&mut head.terminator, Terminator::Jump { to: tail_id, args: Vec::new() });
        for (offset, (op, _)) in insts.iter().enumerate() {
            for value in op.defs() {
                self.value_def_mut(value).def = Def::Inst(tail_id, offset);
            }
            for value in operands_of(op) {
                self.use_sites_mut(value)
                    .move_site(Site::Inst(block, at + offset), Site::Inst(tail_id, offset));
            }
        }
        let mut operands: SmallVec<[ValueId; 4]> = SmallVec::new();
        terminator.for_each_operand(|value| operands.push(value));
        for value in operands {
            self.use_sites_mut(value).move_site(Site::Term(block), Site::Term(tail_id));
        }
        self.new_block(Block { params: Vec::new(), insts, terminator })
    }

    /// Every edge into `from`, every `Diamond` join and loop stage and exit
    /// that names it, and the entry if it is `from`, name `to` instead.
    /// `to`'s own terminator is included, so an edge `to -> from` becomes a
    /// loop on `to`. Cost: O(the body's terminators).
    ///
    /// # Panics
    /// The parameters of `from` and `to` differ in count or type, so some
    /// edge would not agree with its new target.
    pub(super) fn retarget(&mut self, from: BlockId, to: BlockId) {
        let param_tys = |body: &Body, block: BlockId| -> Vec<Ty> {
            body.blocks[block.index()].params.iter().map(|param| body.ty_of(*param)).collect()
        };
        let (from_tys, to_tys) = (param_tys(self, from), param_tys(self, to));
        assert!(
            from_tys == to_tys,
            "edges into {from:?} of parameters {from_tys:?} cannot go to {to:?} of parameters {to_tys:?}"
        );
        for block in &mut self.blocks {
            block.terminator.for_each_block_mut(|named| {
                if *named == from {
                    *named = to;
                }
            });
        }
        if self.entry == from {
            self.entry = to;
        }
    }

    /// The call `call` is the result of becomes a `Spawn` and its
    /// `TakeEffect` at its place (effects §6): the start takes the call's
    /// operands and its order token, and the taking defines `call` and the
    /// token the call yielded. Cost: O(the block's instructions after it).
    ///
    /// # Panics
    /// `call` is not the result of a `FunctionCall`, or the callee's type
    /// does not state that the call may be split.
    pub(super) fn split_call(&mut self, types: &mut Types<Concrete>, call: ValueId) {
        let Def::Inst(block, index) = self.def_of(call) else {
            panic!("{call:?} is not the result of a call")
        };
        let Op::FunctionCall { dst, callee_ty, .. } = &self.blocks[block.index()].insts[index].0 else {
            panic!("{call:?} is not the result of a call")
        };
        assert_eq!(*dst, call, "{call:?} is not the result of a call");
        let TyKind::Fn { ret, .. } = types.kind(*callee_ty) else {
            panic!("a call's callee type is a function type")
        };
        let ret = *ret;
        let splits = fn_effect(types, *callee_ty).map(|effect| effect.split());
        assert!(
            matches!(splits, Some(SplitCost::WithCost(_))),
            "{call:?}'s callee does not state that it may be split (effects §6)"
        );
        let handle_ty = types.intern(TyKind::Handle(ret));
        let (op, span) = self.take_inst(block, index);
        let Op::FunctionCall { dst, callee, callee_ty, args, order } = op else {
            panic!("the instruction read above is a call")
        };
        let handle = self.insert_new_inst(
            types,
            block,
            index,
            &[handle_ty],
            |made| Op::Spawn { dst: made[0], callee, callee_ty, args, order: order.map(|edge| edge.before) },
            span,
        )[0];
        let taking = Op::TakeEffect { dst, src: handle, order: order.map(|edge| edge.after) };
        self.insert_inst(block, index + 1, taking, span);
    }
}

fn operands_of(op: &Op) -> SmallVec<[ValueId; 4]> {
    let mut operands = SmallVec::new();
    op.for_each_operand(|value| operands.push(value));
    operands
}

#[derive(Clone, Copy)]
enum Shift {
    Up,
    Down,
}
