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
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::body::{Block, BlockId, Body, Def, Edge, Inst, OrderUse, Site, Terminator, UseSites, ValueDef};
use super::op::Op;
use crate::ir::{DebugInfo, OrderEdge, RefTarget, ValueId, fn_effect};
use crate::ty::{Concrete, Reissue, SplitCost, Ty, TyKind, Types};

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
            slots: Vec::new(),
            values: FxHashMap::default(),
            uses: FxHashMap::default(),
            value_ids: LocalFactory::new(),
            debug: DebugInfo::new(),
        }
    }

    /// A new value of type `ty`, defined at `def`, which is not an order
    /// token. Cost: O(1).
    ///
    /// # Panics
    /// `ty` is an id of another table than `types`; a reference whose
    /// target is a reference (types §4); or `Order`, since an order token
    /// is made only by `make_token`, on the Builder's order thread
    /// (ir-core-structure §2).
    pub(super) fn make_value(&mut self, types: &Types<Concrete>, ty: Ty, def: Def) -> ValueId {
        match types.kind(ty) {
            TyKind::Ref(_, target) => assert!(
                !matches!(types.kind(target.ty()), TyKind::Ref(..)),
                "a reference's target is never a reference (types §4): {ty:?}"
            ),
            TyKind::Order => panic!(
                "a value of Order type is an order token, which only the core makes on the Builder's order \
                 thread (ir-core-structure §2); no caller makes one"
            ),
            _ => {}
        }
        self.insert_value(ValueDef { ty, def, token: false })
    }

    /// A new order token of `order_ty`, defined at `def`. Only the
    /// Builder's order thread calls it. Cost: O(1).
    ///
    /// # Panics
    /// `order_ty` is not `Order`.
    pub(super) fn make_token(&mut self, types: &Types<Concrete>, order_ty: Ty, def: Def) -> ValueId {
        assert!(matches!(types.kind(order_ty), TyKind::Order), "an order token's type is Order, not {order_ty:?}");
        self.insert_value(ValueDef { ty: order_ty, def, token: true })
    }

    fn insert_value(&mut self, value_def: ValueDef) -> ValueId {
        let value = self.value_ids.next();
        self.values.insert(value, value_def);
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

    /// Records `inst` as the instruction at `index` of `block`: it defines
    /// its result and the token it yields there, and uses its operands
    /// there. Cost: O(its operands).
    ///
    /// # Panics
    /// A value it defines or reads is not a value of this body.
    pub(super) fn record_inst(&mut self, block: BlockId, index: usize, inst: &Inst) {
        for value in inst.defs() {
            self.value_def_mut(value).def = Def::Inst(block, index);
        }
        let site = Site::Inst(block, index);
        for value in inst.operands() {
            self.use_sites_mut(value).add(site, 1);
        }
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
            let inst = &self.blocks[block.index()].insts[index];
            let (defs, operands) = (inst.defs(), inst.operands());
            for value in defs {
                self.value_def_mut(value).def = Def::Inst(block, to);
            }
            for value in operands {
                self.use_sites_mut(value)
                    .move_site(Site::Inst(block, index), Site::Inst(block, to));
            }
        }
    }

    /// Inserts `inst` at `index` of `block`; the instructions from `index`
    /// on move down by one. Every value it defines is already a value of
    /// this body and is defined here from now on. Cost: O(operands of the
    /// instructions from `index` on).
    pub(super) fn insert_inst(&mut self, block: BlockId, index: usize, inst: Inst) {
        self.shift_insts(block, index, Shift::Up);
        self.record_inst(block, index, &inst);
        self.blocks[block.index()].insts.insert(index, inst);
    }

    /// Takes the instruction at `index` of `block` out, with its uses; the
    /// instructions after it move up by one. The values it defined keep a
    /// stale definition: the caller defines them again or drops them.
    /// Cost: O(operands and results of the instructions from `index` on).
    fn take_inst(&mut self, block: BlockId, index: usize) -> Inst {
        for value in self.blocks[block.index()].insts[index].operands() {
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
                Site::Inst(block, index) => self.blocks[block.index()].insts[index].for_each_operand_mut(swap),
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
            Def::Slot(index) => {
                self.refuse_used(value)?;
                self.slots.remove(index);
                let later: SmallVec<[ValueId; 4]> = self.slots[index..].iter().copied().collect();
                for (offset, slot) in later.into_iter().enumerate() {
                    self.value_def_mut(slot).def = Def::Slot(index + offset);
                }
                self.drop_value(value);
                Ok(())
            }
            Def::Param(_) | Def::Capture(_) | Def::OrderParam => panic!(
                "{value:?} is part of the body's interface; the core does not remove it"
            ),
            Def::UnplacedParam(block) => {
                panic!("{value:?} is a parameter the Builder has not placed on {block:?} yet")
            }
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
        self.remove_inst_at(block, index).map(drop)
    }

    /// Removes the instruction at `index` of `block`, refused while its
    /// result has uses, as `remove_value` removes a value an instruction
    /// defines; an instruction without a result is removed the same way.
    /// It returns the positions taken out, in the order they were taken,
    /// each in the coordinates left by the ones before it. Cost:
    /// `remove_value`'s for an instruction.
    ///
    /// # Panics
    /// The instruction is a split call's start, which goes only with its
    /// taking.
    pub(super) fn remove_inst_at(&mut self, block: BlockId, index: usize) -> Result<SmallVec<[InstShift; 2]>, StillUsed> {
        let inst = &self.blocks[block.index()].insts[index];
        if let Some(result) = inst.result {
            self.refuse_used(result)?;
        }
        let mut shifts = SmallVec::new();
        shifts.push(InstShift::Removed { block, index });
        match (&inst.op, inst.order) {
            (Op::TakeEffect { src }, order) => {
                let (src, yielded) = (*src, order.and_then(OrderUse::yielded));
                let Def::Inst(start_block, start_index) = self.def_of(src) else {
                    panic!("a taking's handle is a start's result (effects §6)")
                };
                let start = &self.blocks[start_block.index()].insts[start_index];
                assert!(
                    matches!(start.op, Op::Spawn { .. }),
                    "a taking's handle is a start's result (effects §6)"
                );
                match (start.order.and_then(OrderUse::consumed), yielded) {
                    (Some(before), Some(after)) => self.replace_all_uses(after, before),
                    (None, None) => {}
                    (Some(_), None) | (None, Some(_)) => {
                        panic!("a split call's start and taking are ordered together (effects §6)")
                    }
                }
                let taking = self.take_inst(block, index);
                let start_index =
                    if start_block == block && start_index > index { start_index - 1 } else { start_index };
                let start = self.take_inst(start_block, start_index);
                shifts.push(InstShift::Removed { block: start_block, index: start_index });
                for defined in start.defs().into_iter().chain(taking.defs()) {
                    self.drop_value(defined);
                }
            }
            (_, Some(OrderUse::Edge(edge))) => {
                self.replace_all_uses(edge.after, edge.before);
                let removed = self.take_inst(block, index);
                for defined in removed.defs() {
                    self.drop_value(defined);
                }
            }
            (_, Some(OrderUse::Start(_) | OrderUse::Take(_))) => {
                panic!("a split call's start goes only with its taking (effects §6)")
            }
            (_, None) => {
                let removed = self.take_inst(block, index);
                for defined in removed.defs() {
                    self.drop_value(defined);
                }
            }
        }
        Ok(shifts)
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

    /// Places the Builder's parameter `param` last on `block`, and passes
    /// each value of `incoming` on the edge it names by source block and
    /// edge position. Cost: O(|incoming| and the named terminators' edges).
    ///
    /// # Panics
    /// A named edge is a loop's edge into its body, which carries no
    /// arguments.
    pub(super) fn place_param(&mut self, block: BlockId, param: ValueId, incoming: &[(BlockId, usize, ValueId)]) {
        let index = self.blocks[block.index()].params.len();
        self.blocks[block.index()].params.push(param);
        self.value_def_mut(param).def = Def::BlockParam(block, index);
        for (from, position, value) in incoming {
            let mut edge_at = 0;
            self.blocks[from.index()].terminator.for_each_edge_mut(|_, _, args| {
                if edge_at == *position {
                    let Some(args) = args else {
                        panic!("{block:?} is a loop body entered from {from:?}; it takes no parameter of the Builder's")
                    };
                    args.push(*value);
                }
                edge_at += 1;
            });
            self.use_sites_mut(*value).add(Site::Term(*from), 1);
        }
    }

    /// Removes every parameter of `params` whose arguments, other than the
    /// parameter itself, are one value: its uses become that value, until
    /// no such parameter is left (Braun et al. 2013, trivial phis). Cost:
    /// O(|params| · (the body's edges + uses)) per round, at most |params|
    /// rounds.
    pub(super) fn remove_trivial_params(&mut self, params: &[ValueId]) {
        loop {
            let mut removed_any = false;
            for param in params {
                let Some(ValueDef { def: Def::BlockParam(block, index), .. }) = self.values.get(param).copied() else {
                    continue;
                };
                let mut incoming: SmallVec<[ValueId; 2]> = SmallVec::new();
                for pred in &self.blocks {
                    for edge in pred.terminator.edges().iter().filter(|edge| edge.to == block) {
                        let arg = edge.args[index - edge.supplied];
                        if arg != *param && !incoming.contains(&arg) {
                            incoming.push(arg);
                        }
                    }
                }
                let [only] = incoming[..] else {
                    continue;
                };
                self.replace_all_uses(*param, only);
                self.remove_value(*param)
                    .expect("a parameter whose uses were all replaced has none");
                removed_any = true;
            }
            if !removed_any {
                return;
            }
        }
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
        for (offset, inst) in insts.iter().enumerate() {
            for value in inst.defs() {
                self.value_def_mut(value).def = Def::Inst(tail_id, offset);
            }
            for value in inst.operands() {
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
    /// `TakeEffect` at its place (effects §6): the start reads the call's
    /// operands and consumes its order token, and the taking defines `call`
    /// and yields the token the call yielded. Splitting is a property of the
    /// callee type's effect, so an indirect callee splits as a named one
    /// does. Cost: O(the block's instructions after it).
    ///
    /// # Panics
    /// `call` is not the result of a `FunctionCall`, or the callee's type
    /// does not state that the call may be split.
    pub(super) fn split_call(&mut self, types: &mut Types<Concrete>, call: ValueId) {
        let Def::Inst(block, index) = self.def_of(call) else {
            panic!("{call:?} is not the result of a call")
        };
        let Op::FunctionCall { callee_ty, .. } = &self.blocks[block.index()].insts[index].op else {
            panic!("{call:?} is not the result of a call")
        };
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
        let Inst { op, order, span, .. } = self.take_inst(block, index);
        let Op::FunctionCall { callee, callee_ty, args } = op else {
            panic!("the instruction read above is a call")
        };
        let edge = order.map(|order| match order {
            OrderUse::Edge(edge) => edge,
            OrderUse::Start(_) | OrderUse::Take(_) => panic!("a call's order is an edge"),
        });
        let handle = self.make_value(types, handle_ty, Def::Inst(block, index));
        let start = Inst {
            op: Op::Spawn { callee, callee_ty, args },
            result: Some(handle),
            order: edge.map(|OrderEdge { before, .. }| OrderUse::Start(before)),
            span,
        };
        let taking = Inst {
            op: Op::TakeEffect { src: handle },
            result: Some(call),
            order: edge.map(|OrderEdge { after, .. }| OrderUse::Take(after)),
            span,
        };
        self.insert_inst(block, index, start);
        self.insert_inst(block, index + 1, taking);
    }

    /// Inserts an instruction of `op` at `index` of `block` and returns its
    /// result, a new value of `ty`. The instruction takes no order token.
    /// Cost: O(operands of the instructions from `index` on).
    ///
    /// # Panics
    /// `op` yields no value; it is a `Spawn`, a `TakeEffect` or a `Merge`,
    /// which only the core writes; it is a call whose effect is not pure,
    /// which takes an order token (effects §3); `index` is past the block's
    /// end; or `ty` is refused by `make_value`.
    pub(super) fn insert_pure(
        &mut self,
        types: &Types<Concrete>,
        block: BlockId,
        index: usize,
        op: Op,
        ty: Ty,
        span: Span,
    ) -> ValueId {
        assert!(op.has_result(), "{op:?} yields no value; an inserted instruction yields one");
        assert!(
            !matches!(op, Op::Spawn { .. } | Op::TakeEffect { .. } | Op::Merge { .. }),
            "{op:?} is written only by the core (effects §3, §6)"
        );
        if let Op::FunctionCall { callee_ty, .. } = &op {
            let Some(effect) = fn_effect(types, *callee_ty) else {
                panic!("a call's callee type is a function type")
            };
            assert!(
                effect.reissue() == Reissue::Pure,
                "a call whose effect is not pure takes an order token (effects §3); it is not inserted as pure"
            );
        }
        assert!(index <= self.blocks[block.index()].insts.len(), "{block:?} has no position {index}");
        let result = self.make_value(types, ty, Def::Inst(block, index));
        self.insert_inst(block, index, Inst { op, result: Some(result), order: None, span });
        result
    }

    /// Every operand `of` at `site` becomes `with`. Cost: O(the site's
    /// operands).
    ///
    /// # Panics
    /// `site` does not use `of`.
    fn replace_use_at(&mut self, site: Site, of: ValueId, with: ValueId) {
        let count = self.use_sites_mut(of).take_site(site);
        let swap = |value: &mut ValueId| {
            if *value == of {
                *value = with;
            }
        };
        match site {
            Site::Inst(block, index) => self.blocks[block.index()].insts[index].for_each_operand_mut(swap),
            Site::Term(block) => self.blocks[block.index()].terminator.for_each_operand_mut(swap),
        }
        self.use_sites_mut(with).add(site, count);
    }

    /// `block`'s two-way branch becomes a jump into the arm `then` names,
    /// passing that arm's arguments; the condition and the other arm's
    /// arguments lose their use here. A `Diamond`'s join stays a block of
    /// the body (D45). Cost: O(the terminator's operands).
    ///
    /// # Panics
    /// `block` does not end in a `JumpIf` or a `Diamond`.
    pub(super) fn take_arm(&mut self, block: BlockId, then: bool) {
        let terminator = &mut self.blocks[block.index()].terminator;
        assert!(
            matches!(terminator, Terminator::JumpIf { .. } | Terminator::Diamond { .. }),
            "{block:?} ends in {terminator:?}, not in a two-way branch"
        );
        let (cond, to, args, dropped) = match std::mem::replace(terminator, Terminator::Diverge) {
            Terminator::JumpIf { cond, then_to, then_args, else_to, else_args }
            | Terminator::Diamond { cond, then_to, then_args, else_to, else_args, .. } => match then {
                true => (cond, then_to, then_args, else_args),
                false => (cond, else_to, else_args, then_args),
            },
            other => panic!("{block:?} ended in a two-way branch a moment ago, not in {other:?}"),
        };
        *terminator = Terminator::Jump { to, args };
        let site = Site::Term(block);
        for value in std::iter::once(cond).chain(dropped) {
            self.use_sites_mut(value).remove_one(site);
        }
    }

    /// Removes every block `keep` does not keep, with its instructions, its
    /// terminator and every value it defines; the kept blocks are renumbered
    /// in their order, and the result names each old block's new id. Cost:
    /// O(the body's instructions, values and uses).
    ///
    /// # Panics
    /// `keep` is not one flag per block; the entry is removed; a kept block
    /// names a removed one; or a kept block uses a value a removed block
    /// defines.
    pub(super) fn remove_blocks(&mut self, keep: &[bool]) -> Vec<Option<BlockId>> {
        assert_eq!(keep.len(), self.blocks.len(), "one flag per block says whether it is kept");
        let mut kept = 0;
        let renamed: Vec<Option<BlockId>> = keep
            .iter()
            .map(|keep| {
                keep.then(|| {
                    kept += 1;
                    BlockId::of_index(kept - 1)
                })
            })
            .collect();
        let mut released: Vec<(Site, ValueId)> = Vec::new();
        let mut defined: Vec<ValueId> = Vec::new();
        for (index, block) in self.blocks.iter().enumerate().filter(|(index, _)| !keep[*index]) {
            let id = BlockId::of_index(index);
            defined.extend(block.params.iter().copied());
            for (at, inst) in block.insts.iter().enumerate() {
                defined.extend(inst.defs());
                released.extend(inst.operands().into_iter().map(|value| (Site::Inst(id, at), value)));
            }
            block.terminator.for_each_operand(|value| released.push((Site::Term(id), value)));
        }
        for (site, value) in released {
            self.use_sites_mut(value).remove_one(site);
        }
        for value in defined {
            assert_eq!(self.use_count(value), 0, "{value:?} is defined in a removed block and used in a kept one");
            self.drop_value(value);
        }
        let rename = |block: BlockId| {
            renamed[block.index()].unwrap_or_else(|| panic!("a kept block names the removed {block:?}"))
        };
        let blocks = std::mem::take(&mut self.blocks);
        self.blocks = blocks.into_iter().zip(keep).filter(|(_, keep)| **keep).map(|(block, _)| block).collect();
        for block in &mut self.blocks {
            block.terminator.for_each_block_mut(|named| *named = rename(*named));
        }
        self.entry = rename(self.entry);
        for value_def in self.values.values_mut() {
            value_def.def = match value_def.def {
                Def::Inst(block, index) => Def::Inst(rename(block), index),
                Def::BlockParam(block, index) => Def::BlockParam(rename(block), index),
                Def::UnplacedParam(block) => panic!("a sealed body has no unplaced parameter, and {block:?} has one"),
                def @ (Def::Param(_) | Def::Capture(_) | Def::OrderParam | Def::Slot(_)) => def,
            };
        }
        for sites in self.uses.values_mut() {
            sites.rename_blocks(rename);
        }
        renamed
    }

    /// The order token current just before position `index` of `block`: the
    /// last one yielded or merged above it in the block, or else the one
    /// that enters the block. Cost: O(the instructions scanned), and for a
    /// block with no token of its own, O(the body's blocks and edges).
    ///
    /// # Panics
    /// The body is not ordered, or no token reaches `block`, which no edge
    /// from the entry reaches.
    pub(super) fn reaching_token(&self, block: BlockId, index: usize) -> Result<ValueId, InFlight> {
        if let Some(token) = self.token_above(block, index)? {
            return Ok(token);
        }
        let preds = self.predecessors();
        let mut visited = FxHashSet::default();
        let mut pending = vec![block];
        while let Some(at) = pending.pop() {
            if !visited.insert(at) {
                continue;
            }
            if at != block
                && let Some(token) = self.token_above(at, self.blocks[at.index()].insts.len())?
            {
                return Ok(token);
            }
            if let Some(param) = self.blocks[at.index()].params.iter().find(|param| self.is_token(**param)) {
                return Ok(*param);
            }
            if at == self.entry {
                return Ok(self.order_param.expect("an order token is read only in an ordered body"));
            }
            pending.extend(preds[&at].iter().copied());
        }
        panic!("no order token reaches {block:?}; no edge from the entry reaches it")
    }

    /// The last token yielded or merged above position `index` of `block`.
    fn token_above(&self, block: BlockId, index: usize) -> Result<Option<ValueId>, InFlight> {
        for (at, inst) in self.blocks[block.index()].insts[..index].iter().enumerate().rev() {
            if let Some(OrderUse::Start(_)) = inst.order {
                return Err(InFlight { start_block: block, start_index: at });
            }
            if let Some(token) = inst.order.and_then(OrderUse::yielded) {
                return Ok(Some(token));
            }
            if let Op::Merge { .. } = inst.op {
                return Ok(inst.result);
            }
        }
        Ok(None)
    }

    /// Moves the instruction at `from_index` of `from` to stand before
    /// position `to_index` of `to`, counted before the move, the block's
    /// length naming its end. It returns the position it left and the one it
    /// took, each in the coordinates left by the one before.
    ///
    /// An effectful call's token is rethreaded (effects §3): its old place
    /// hands on the token it consumed, and at its new place it consumes the
    /// token current there and hands its own on to what that token reached,
    /// with token parameters added where paths now bring different tokens
    /// (Braun et al. 2013). Any other instruction moves as it is.
    ///
    /// Cost: O(operands of the instructions of both blocks from the
    /// positions on); for an effectful call, also O(the body's blocks, edges
    /// and the uses of the token it takes).
    ///
    /// # Panics
    /// A split call's start or taking, or a `Merge`, is moved to another
    /// block or across an instruction on the order chain, which would take
    /// the chain apart; an effectful call is moved to where a split call is
    /// in flight, where no token is current; a position is past its block's
    /// end.
    pub(super) fn move_inst(
        &mut self,
        types: &Types<Concrete>,
        from: BlockId,
        from_index: usize,
        to: BlockId,
        to_index: usize,
    ) -> [InstShift; 2] {
        assert!(to_index <= self.blocks[to.index()].insts.len(), "{to:?} has no position {to_index}");
        let moved = &self.blocks[from.index()].insts[from_index];
        let edge = match moved.order {
            Some(OrderUse::Edge(edge)) => Some(edge),
            Some(OrderUse::Start(_) | OrderUse::Take(_)) | None if moved.is_on_order_chain() => {
                let (low, high) = (from_index.min(to_index), from_index.max(to_index));
                let crossed = &self.blocks[from.index()].insts[low..high];
                assert!(
                    from == to && crossed.iter().enumerate().all(|(at, inst)| low + at == from_index || !inst.is_on_order_chain()),
                    "a split call's start or taking, or a Merge, moves only within its block and across nothing on the \
                     order chain (effects §3, §6)"
                );
                None
            }
            Some(OrderUse::Start(_) | OrderUse::Take(_)) | None => None,
        };
        if let Some(edge) = edge {
            self.replace_all_uses(edge.after, edge.before);
        }
        let mut inst = self.take_inst(from, from_index);
        let to_index = if to == from && to_index > from_index { to_index - 1 } else { to_index };
        let Some(edge) = edge else {
            self.insert_inst(to, to_index, inst);
            return [InstShift::Removed { block: from, index: from_index }, InstShift::Inserted { block: to, index: to_index }];
        };
        let before = self.reaching_token(to, to_index).unwrap_or_else(|InFlight { start_block, start_index }| {
            panic!(
                "a split call started at {start_block:?}[{start_index}] is in flight at {to:?}[{to_index}]; no token \
                 is current there"
            )
        });
        inst.order = Some(OrderUse::Edge(OrderEdge { before, after: edge.after }));
        self.insert_inst(to, to_index, inst);
        self.hand_on_token(types, before, edge.after, to, to_index);
        [InstShift::Removed { block: from, index: from_index }, InstShift::Inserted { block: to, index: to_index }]
    }

    /// The effectful call at `index` of `block` now consumes `old` and
    /// yields `new`: every use of `old` the call now precedes takes `new`,
    /// through token parameters where paths bring different tokens.
    fn hand_on_token(&mut self, types: &Types<Concrete>, old: ValueId, new: ValueId, block: BlockId, index: usize) {
        let old_def = self.def_of(old);
        let mut reader = Reconstruct::new(self, self.ty_of(old), Made::Token, old);
        reader.at_end.insert(block, new);
        match old_def {
            Def::Inst(defined, _) if defined != block => {
                reader.at_end.insert(defined, old);
            }
            Def::Inst(..) => {}
            Def::BlockParam(defined, _) => {
                reader.at_start.insert(defined, old);
            }
            Def::OrderParam => reader.seed_entry(self, old),
            def @ (Def::Param(_) | Def::Capture(_) | Def::Slot(_) | Def::UnplacedParam(_)) => {
                panic!("{old:?} is an order token, defined by an instruction or an order parameter, not at {def:?}")
            }
        }
        let call = Site::Inst(block, index);
        let sites: Vec<Site> = self.users(old).filter(|site| *site != call).collect();
        for site in sites {
            let defined_in = |at: BlockId, before: Option<usize>| match old_def {
                Def::Inst(defined, def_index) => defined == at && before.is_none_or(|index| def_index < index),
                Def::BlockParam(defined, _) => defined == at,
                Def::OrderParam | Def::Param(_) | Def::Capture(_) | Def::Slot(_) | Def::UnplacedParam(_) => false,
            };
            let value = match site {
                Site::Inst(at, at_index) if at == block => match at_index > index {
                    true => new,
                    false => old,
                },
                Site::Term(at) if at == block => new,
                Site::Inst(at, at_index) if defined_in(at, Some(at_index)) => old,
                Site::Term(at) if defined_in(at, None) => old,
                Site::Inst(at, _) | Site::Term(at) => reader.read_start(self, types, at),
            };
            if value != old {
                self.replace_use_at(site, old, value);
            }
        }
        self.remove_trivial_params(&reader.added);
    }

    /// Replaces the whole-slot `Take`s and `Assign`s of `slot` with values:
    /// a `Take` reads what the last `Assign` above it wrote, through block
    /// parameters at joins and loop headers, and an `Undef` at the entry
    /// where no `Assign` came first (Braun et al. 2013). The slot goes with
    /// them. It returns the positions taken out and put in, in order, each
    /// in the coordinates left by the ones before. Cost: O(the body's
    /// blocks, edges and the slot's uses), and the trivial-parameter removal's.
    ///
    /// # Panics
    /// `slot` is not a slot; something other than a whole-slot `Take` or
    /// `Assign` names it, which a promotion does not rewrite; or a read
    /// reaches the start of an entry block that an edge enters again, where
    /// a parameter cannot stand.
    pub(super) fn promote_slot(&mut self, types: &Types<Concrete>, slot: ValueId) -> Vec<InstShift> {
        let Def::Slot(_) = self.def_of(slot) else {
            panic!("{slot:?} is not a slot")
        };
        let ty = self.ty_of(slot);
        let mut loads: Vec<SlotAccess> = Vec::new();
        let mut stores: Vec<SlotAccess> = Vec::new();
        for site in self.users(slot) {
            let Site::Inst(block, index) = site else {
                panic!("{slot:?} is named by a terminator, which a promotion does not rewrite")
            };
            let inst = &self.blocks[block.index()].insts[index];
            match &inst.op {
                Op::Take { target: RefTarget::Var(named), path, .. } if *named == slot && path.is_empty() => {
                    let loaded = inst.result.expect("a Take yields a value");
                    loads.push(SlotAccess { block, index, value: loaded });
                }
                Op::Assign { target: RefTarget::Var(named), path, value, .. } if *named == slot && path.is_empty() => {
                    stores.push(SlotAccess { block, index, value: *value });
                }
                other => panic!(
                    "a promotion rewrites only whole-slot Take and Assign of {slot:?}, and {block:?}[{index}] is \
                     {other:?}; the facts that let it be promoted are wrong"
                ),
            }
        }
        loads.sort_by_key(SlotAccess::position);
        stores.sort_by_key(SlotAccess::position);

        let undef = self.make_value(types, ty, Def::Inst(self.entry, 0));
        let mut reader = Reconstruct::new(self, ty, Made::Value, undef);
        reader.seed_entry(self, undef);
        for store in &stores {
            reader.at_end.insert(store.block, store.value);
        }
        let last_store_above = |block: BlockId, index: usize| {
            stores
                .iter()
                .filter(|store| store.block == block && store.index < index)
                .last()
                .map(|store| store.value)
        };
        let mut read: FxHashMap<ValueId, ValueId> = FxHashMap::default();
        for load in &loads {
            let value = match last_store_above(load.block, load.index) {
                Some(value) => value,
                None => reader.read_start(self, types, load.block),
            };
            read.insert(load.value, value);
        }
        for load in &loads {
            let mut value = read[&load.value];
            while let Some(next) = read.get(&value) {
                value = *next;
            }
            self.replace_all_uses(load.value, value);
        }

        let mut shifts = Vec::new();
        let mut taken: Vec<(BlockId, usize)> = loads.iter().chain(&stores).map(SlotAccess::position).collect();
        taken.sort_by(|a, b| b.cmp(a));
        for (block, index) in taken {
            let removed = self
                .remove_inst_at(block, index)
                .expect("a promoted Take's uses were all replaced, and an Assign yields nothing");
            shifts.extend(removed);
        }
        self.remove_value(slot).expect("a promoted slot's every Take and Assign is removed");
        self.remove_trivial_params(&reader.added);
        match self.use_count(undef) {
            0 => self.drop_value(undef),
            _ => {
                let entry = self.entry;
                self.insert_inst(entry, 0, Inst { op: Op::Undef, result: Some(undef), order: None, span: Span::ZERO });
                shifts.push(InstShift::Inserted { block: entry, index: 0 });
            }
        }
        shifts
    }
}

/// Where an edit took an instruction out or put one in, for a cursor
/// walking the body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InstShift {
    Inserted { block: BlockId, index: usize },
    Removed { block: BlockId, index: usize },
}

/// A position between a split call's start and its taking: the start
/// consumed the token, and the taking has not yielded the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct InFlight {
    pub start_block: BlockId,
    pub start_index: usize,
}

/// A whole-slot `Take`, with the value it yields, or `Assign`, with the
/// value it stores.
struct SlotAccess {
    block: BlockId,
    index: usize,
    value: ValueId,
}

impl SlotAccess {
    fn position(&self) -> (BlockId, usize) {
        (self.block, self.index)
    }
}

/// Whether a reconstructed variable's parameters are order tokens.
#[derive(Clone, Copy)]
enum Made {
    Token,
    Value,
}

/// On-the-fly SSA construction of one variable over a finished body (Braun
/// et al. 2013, every block sealed): the value at a block's start is its one
/// predecessor's value at its end, or a new parameter where edges from more
/// than one block enter it.
struct Reconstruct {
    ty: Ty,
    made: Made,
    /// Every edge into each block, by source block and edge position.
    edges_into: Vec<SmallVec<[(BlockId, usize); 2]>>,
    at_end: FxHashMap<BlockId, ValueId>,
    at_start: FxHashMap<BlockId, ValueId>,
    /// Single-predecessor blocks being read, to close a cycle no edge from
    /// outside enters.
    reading: FxHashSet<BlockId>,
    /// The value code no edge from the entry reaches reads: it never runs.
    unreached: ValueId,
    added: Vec<ValueId>,
}

impl Reconstruct {
    fn new(body: &Body, ty: Ty, made: Made, unreached: ValueId) -> Reconstruct {
        let mut edges_into: Vec<SmallVec<[(BlockId, usize); 2]>> = vec![SmallVec::new(); body.blocks.len()];
        for (from, block) in body.blocks() {
            for (position, edge) in block.terminator().edges().iter().enumerate() {
                edges_into[edge.to.index()].push((from, position));
            }
        }
        Reconstruct {
            ty,
            made,
            edges_into,
            at_end: FxHashMap::default(),
            at_start: FxHashMap::default(),
            reading: FxHashSet::default(),
            unreached,
            added: Vec::new(),
        }
    }

    /// `value` stands at the entry's start; an entry an edge enters again
    /// is left unseeded, and a read that reaches it panics.
    fn seed_entry(&mut self, body: &Body, value: ValueId) {
        if self.edges_into[body.entry.index()].is_empty() {
            self.at_start.insert(body.entry, value);
        }
    }

    fn read_end(&mut self, body: &mut Body, types: &Types<Concrete>, block: BlockId) -> ValueId {
        match self.at_end.get(&block) {
            Some(value) => *value,
            None => self.read_start(body, types, block),
        }
    }

    fn read_start(&mut self, body: &mut Body, types: &Types<Concrete>, block: BlockId) -> ValueId {
        if let Some(value) = self.at_start.get(&block) {
            return *value;
        }
        assert_ne!(
            block, body.entry,
            "the entry block is entered by an edge; a value read at its start would need a parameter the entry \
             cannot take"
        );
        let edges = self.edges_into[block.index()].clone();
        let Some(&(first, _)) = edges.first() else {
            return self.unreached;
        };
        let value = match edges.iter().all(|(from, _)| *from == first) {
            true => {
                if !self.reading.insert(block) {
                    return self.unreached;
                }
                let value = self.read_end(body, types, first);
                self.reading.remove(&block);
                value
            }
            false => {
                let param = match self.made {
                    Made::Token => body.make_token(types, self.ty, Def::UnplacedParam(block)),
                    Made::Value => body.make_value(types, self.ty, Def::UnplacedParam(block)),
                };
                self.at_start.insert(block, param);
                let incoming: Vec<(BlockId, usize, ValueId)> = edges
                    .iter()
                    .map(|&(from, position)| (from, position, self.read_end(body, types, from)))
                    .collect();
                body.place_param(block, param, &incoming);
                self.added.push(param);
                param
            }
        };
        self.at_start.insert(block, value);
        value
    }
}

#[derive(Clone, Copy)]
enum Shift {
    Up,
    Down,
}
