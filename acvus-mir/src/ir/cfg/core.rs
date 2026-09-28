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
//! - A `Diamond`'s or a `Switch`'s join and a `For`'s or `While`'s stages
//!   and exit follow every retarget, edge split, block split and block
//!   removal.

use acvus_ast::Span;
use acvus_utils::LocalFactory;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::body::{Block, BlockId, Body, Def, Edge, Inst, OrderUse, Site, Terminator, UseSites, ValueDef};
use super::op::Op;
use crate::ir::{DebugInfo, IndexBound, OrderEdge, PathSeg, RefTarget, SwitchKey, ValueId, fn_effect};
use crate::ty::{Concrete, ObjectTy, Reissue, SplitCost, Ty, TyKind, Types};

/// A value that still has uses, removed without a replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StillUsed {
    pub value: ValueId,
}

impl Body {
    /// A body with no block and no value yet, whose returns yield a `ret`;
    /// its entry is the first block added to it.
    pub(super) fn empty(ret: Ty) -> Body {
        Body {
            ret,
            blocks: Vec::new(),
            entry: BlockId::of_index(0),
            params: Vec::new(),
            captures: Vec::new(),
            order_param: None,
            slots: Vec::new(),
            slot_spans: FxHashMap::default(),
            values: FxHashMap::default(),
            uses: FxHashMap::default(),
            value_ids: LocalFactory::new(),
            next_inst: 0,
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

    /// A new value of `like`'s type, an order token exactly where `like` is
    /// one, defined at `def`. `like` may be a value of another body of the
    /// module, whose types are the same table's. Cost: O(1).
    fn make_like(&mut self, like: ValueDef, def: Def) -> ValueId {
        self.insert_value(ValueDef { def, ..like })
    }

    fn value_def(&self, value: ValueId) -> ValueDef {
        *self.values.get(&value).unwrap_or_else(|| panic!("{value:?} is not a value of this body"))
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
            Def::Slot { index, .. } => {
                self.refuse_used(value)?;
                self.slots.remove(index);
                self.slot_spans.remove(&value);
                let later: SmallVec<[ValueId; 4]> = self.slots[index..].iter().copied().collect();
                for (offset, slot) in later.into_iter().enumerate() {
                    let ValueDef { def: Def::Slot { index: at, .. }, .. } = self.value_def_mut(slot) else {
                        panic!("{slot:?} stands among the slots and is defined as one")
                    };
                    *at = index + offset;
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
        let start = self.make_inst(
            Op::Spawn { callee, callee_ty, args },
            Some(handle),
            edge.map(|OrderEdge { before, .. }| OrderUse::Start(before)),
            span,
        );
        let taking = self.make_inst(
            Op::TakeEffect { src: handle },
            Some(call),
            edge.map(|OrderEdge { after, .. }| OrderUse::Take(after)),
            span,
        );
        self.insert_inst(block, index, start);
        self.insert_inst(block, index + 1, taking);
    }

    /// Inserts an instruction of `op` at `index` of `block` and returns its
    /// result, a new value of `ty`. The instruction takes no order token.
    /// Cost: O(operands of the instructions from `index` on).
    ///
    /// # Panics
    /// `op` yields no value; it is a `Spawn`, a `TakeEffect` or a `Merge`,
    /// which only the core writes; it is a call whose effect is not pure or
    /// a context's load, which take an order token (effects §3, context §3); `index` is past the block's
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
        assert!(
            !op.touches_context(),
            "a context's load or commit is an effect (context §3), which takes an order token; it is not inserted as pure"
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
        let inst = self.make_inst(op, Some(result), None, span);
        self.insert_inst(block, index, inst);
        result
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
                def @ (Def::Param(_) | Def::Capture(_) | Def::OrderParam | Def::Slot { .. }) => def,
            };
        }
        for sites in self.uses.values_mut() {
            sites.rename_blocks(rename);
        }
        renamed
    }

    /// Moves the instruction at `from_index` of `from` to stand before
    /// position `to_index` of `to`, counted before the move, the block's
    /// length naming its end. It keeps every operand and every value it
    /// defines, order tokens included: the order chain is their def-use, so
    /// a move the editor admits, where each still dominates its uses, leaves
    /// the chain whole. It returns the position it left and the one it took,
    /// each in the coordinates left by the one before. Cost: O(operands and
    /// results of the instructions of both blocks from the positions on).
    ///
    /// # Panics
    /// A position is past its block's end.
    pub(super) fn move_inst(&mut self, from: BlockId, from_index: usize, to: BlockId, to_index: usize) -> [InstShift; 2] {
        assert!(to_index <= self.blocks[to.index()].insts.len(), "{to:?} has no position {to_index}");
        let inst = self.take_inst(from, from_index);
        let to_index = if to == from && to_index > from_index { to_index - 1 } else { to_index };
        self.insert_inst(to, to_index, inst);
        [InstShift::Removed { block: from, index: from_index }, InstShift::Inserted { block: to, index: to_index }]
    }

    /// Replaces the whole-slot `Take`s and `Assign`s of `slot` with values:
    /// a `Take` reads what the last `Assign` above it wrote, through block
    /// parameters at joins and loop headers, and an `Undef` at the entry
    /// where no `Assign` came first, with the slot's span (Braun et al.
    /// 2013). No drop is written: an `Assign` drops nothing, every drop being
    /// an explicit `Drop` (ownership §6). The slot goes with
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
        let Def::Slot { .. } = self.def_of(slot) else {
            panic!("{slot:?} is not a slot")
        };
        let ty = self.ty_of(slot);
        let span = self.slot_spans[&slot];
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
        let mut reader = Reconstruct::new(self, ty, undef);
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
                let inst = self.make_inst(Op::Undef, Some(undef), None, span);
                self.insert_inst(entry, 0, inst);
                shifts.push(InstShift::Inserted { block: entry, index: 0 });
            }
        }
        shifts
    }

    /// `block`'s `Switch` becomes a jump into the arm whose key is `key`, or
    /// into its default where no arm has that key, passing that edge's
    /// arguments; the tag and every other edge's arguments lose their use
    /// here. The `Switch`'s join stays a block of the body (D45, D57). Cost:
    /// O(the terminator's operands).
    ///
    /// # Panics
    /// `block` does not end in a `Switch`, or no arm has `key` and the
    /// `Switch` has no default.
    pub(super) fn take_case(&mut self, block: BlockId, key: &SwitchKey) {
        let terminator = &mut self.blocks[block.index()].terminator;
        let Terminator::Switch { arms, default, .. } = terminator else {
            panic!("{block:?} ends in {terminator:?}, not in a Switch")
        };
        let chosen = arms.iter().position(|(arm_key, _, _)| arm_key == key);
        assert!(
            chosen.is_some() || default.is_some(),
            "{block:?}'s Switch has no arm for {key:?} and no default, so no run reaches it with that key"
        );
        let Terminator::Switch { tag, arms, default, .. } = std::mem::replace(terminator, Terminator::Diverge) else {
            panic!("{block:?} ended in a Switch a moment ago")
        };
        let mut released = vec![tag];
        let mut taken = None;
        for (position, (_, to, args)) in arms.into_iter().enumerate() {
            match Some(position) == chosen {
                true => taken = Some((to, args)),
                false => released.extend(args),
            }
        }
        if let Some((to, args)) = default {
            match taken {
                Some(_) => released.extend(args),
                None => taken = Some((to, args)),
            }
        }
        let (to, args) = taken.expect("an arm or the default is taken, as checked above");
        self.blocks[block.index()].terminator = Terminator::Jump { to, args };
        let site = Site::Term(block);
        for value in released {
            self.use_sites_mut(value).remove_one(site);
        }
    }

    /// A new block stands on every edge of `edges`, each named by its block
    /// and its position in that block's terminator, all into one target: it
    /// takes parameters like the target, each edge enters it with the
    /// arguments it passed and the parameters it supplied, and it jumps to
    /// the target passing every parameter on. A loop's body or exit, named
    /// by an edge, is the new block from now on, as every structural label
    /// follows its edge. Cost: O(the target's parameters + |edges|).
    ///
    /// # Panics
    /// `edges` is empty; a terminator has no edge at its position; or the
    /// edges do not all enter one block.
    pub(super) fn split_edges(&mut self, edges: &[(BlockId, usize)]) -> BlockId {
        let targets: Vec<BlockId> =
            edges.iter().map(|(from, position)| *self.blocks[from.index()].terminator.edge_target_mut(*position)).collect();
        let Some((to, others)) = targets.split_first() else {
            panic!("a split names at least one edge")
        };
        let to = *to;
        assert!(others.iter().all(|other| *other == to), "the split edges all enter {to:?}, not {targets:?}");
        let middle = BlockId::of_index(self.blocks.len());
        let params: Vec<ValueId> = self.blocks[to.index()]
            .params
            .clone()
            .into_iter()
            .enumerate()
            .map(|(index, param)| self.make_like(self.value_def(param), Def::BlockParam(middle, index)))
            .collect();
        let terminator = Terminator::Jump { to, args: params.clone() };
        self.record_terminator(middle, &terminator);
        self.new_block(Block { params, insts: Vec::new(), terminator });
        for (from, position) in edges {
            *self.blocks[from.index()].terminator.edge_target_mut(*position) = middle;
        }
        middle
    }

    /// `block`, whose only predecessor is `pred` and which `pred` enters by
    /// its one `Jump`, joins `pred`: its parameters become the jump's
    /// arguments, its instructions follow `pred`'s, its terminator is
    /// `pred`'s, and the block goes. The result names each old block's new
    /// id, as `remove_blocks` does. Cost: O(the moved instructions' operands
    /// and results, and the parameters' uses), and `remove_blocks`'.
    ///
    /// # Panics
    /// `pred` does not end in a `Jump` to `block`; or an argument of that
    /// jump is a parameter of `block`, which only a cycle no run enters
    /// passes.
    pub(super) fn merge_into_predecessor(&mut self, block: BlockId, pred: BlockId) -> Vec<Option<BlockId>> {
        let jump = std::mem::replace(&mut self.blocks[pred.index()].terminator, Terminator::Diverge);
        let Terminator::Jump { to, args } = jump else {
            panic!("{pred:?} ends in {jump:?}, not in a jump to {block:?}")
        };
        assert_eq!(to, block, "{pred:?} jumps to {to:?}, not to {block:?}");
        for arg in &args {
            self.use_sites_mut(*arg).remove_one(Site::Term(pred));
        }
        let params = std::mem::take(&mut self.blocks[block.index()].params);
        for (param, arg) in params.iter().zip(&args) {
            assert!(
                !params.contains(arg),
                "{pred:?} passes {block:?}'s own parameter {arg:?} to it, on a cycle no run enters"
            );
            self.replace_all_uses(*param, *arg);
        }
        for param in params {
            self.drop_value(param);
        }
        let base = self.blocks[pred.index()].insts.len();
        let insts = std::mem::take(&mut self.blocks[block.index()].insts);
        for (offset, inst) in insts.iter().enumerate() {
            for value in inst.defs() {
                self.value_def_mut(value).def = Def::Inst(pred, base + offset);
            }
            for value in inst.operands() {
                self.use_sites_mut(value).move_site(Site::Inst(block, offset), Site::Inst(pred, base + offset));
            }
        }
        let terminator = std::mem::replace(&mut self.blocks[block.index()].terminator, Terminator::Diverge);
        for value in terminator.operands() {
            self.use_sites_mut(value).move_site(Site::Term(block), Site::Term(pred));
        }
        let merged = &mut self.blocks[pred.index()];
        merged.insts.extend(insts);
        merged.terminator = terminator;
        let keep: Vec<bool> = (0..self.blocks.len()).map(|index| index != block.index()).collect();
        self.remove_blocks(&keep)
    }

    /// The operands of the instruction or terminator at `site` that are `of`
    /// become `with`; no other site changes. Cost: O(the site's operands).
    ///
    /// # Panics
    /// Their types differ, or `site` does not read `of`.
    pub(super) fn replace_uses_at(&mut self, site: Site, of: ValueId, with: ValueId) {
        let (of_ty, with_ty) = (self.ty_of(of), self.ty_of(with));
        assert!(
            of_ty == with_ty,
            "{of:?} of {of_ty:?} is replaced only by a value of its type, not {with:?} of {with_ty:?}"
        );
        if of == with {
            return;
        }
        let mut count = 0;
        let swap = |value: &mut ValueId| {
            if *value == of {
                *value = with;
                count += 1;
            }
        };
        match site {
            Site::Inst(block, index) => self.blocks[block.index()].insts[index].for_each_operand_mut(swap),
            Site::Term(block) => self.blocks[block.index()].terminator.for_each_operand_mut(swap),
        }
        assert!(count > 0, "{site:?} does not read {of:?}");
        for _ in 0..count {
            self.use_sites_mut(of).remove_one(site);
        }
        self.use_sites_mut(with).add(site, count);
    }

    /// The effectful calls at `first` and `second` of `block`, where
    /// `second` consumes what `first` yields and nothing else does, trade
    /// places on the chain (effects §3): `second` consumes what `first`
    /// consumed, `first` consumes what `second` yields, and what used
    /// `second`'s yield uses `first`'s. `second` moves to stand just before
    /// `first`, so every token is defined before its uses. It returns the
    /// position `second` left and the one it took. Cost: O(the uses of
    /// `second`'s yield and the block's instructions from `first` on).
    ///
    /// # Panics
    /// `first` does not come before `second`; either is not an effectful
    /// call; `second` does not consume `first`'s yield, or another site
    /// also does; or `second` reads a value defined from `first` up to it.
    pub(super) fn swap_effects(&mut self, block: BlockId, first: usize, second: usize) -> [InstShift; 2] {
        assert!(first < second, "{block:?}[{first}] comes before {block:?}[{second}] on the chain");
        let edge_at = |body: &Body, index: usize| match body.blocks[block.index()].insts[index].order {
            Some(OrderUse::Edge(edge)) => edge,
            other => panic!("{block:?}[{index}] is not an effectful call on one chain: {other:?}"),
        };
        let (a, b) = (edge_at(self, first), edge_at(self, second));
        assert_eq!(b.before, a.after, "{block:?}[{second}] consumes what {block:?}[{first}] yields");
        assert_eq!(self.use_count(a.after), 1, "only {block:?}[{second}] consumes {:?}", a.after);
        let defined_between = |body: &Body, value: ValueId| {
            matches!(body.def_of(value), Def::Inst(at, index) if at == block && (first..second).contains(&index))
        };
        let mut reads_between = None;
        self.blocks[block.index()].insts[second].op.for_each_operand(|value| {
            if reads_between.is_none() && defined_between(self, value) {
                reads_between = Some(value);
            }
        });
        assert!(reads_between.is_none(), "{block:?}[{second}] reads {reads_between:?}, defined since {block:?}[{first}]");

        self.replace_all_uses(b.after, a.after);
        self.replace_uses_at(Site::Inst(block, second), a.after, a.before);
        self.replace_uses_at(Site::Inst(block, first), a.before, b.after);
        self.move_inst(block, second, block, first)
    }

    /// `slot`, of a tuple or a structural object, becomes one slot per
    /// field, in field order, each with `slot`'s span; every `Take`,
    /// `Assign` and `Ref` of `slot` goes to the slot of the field its path
    /// names first, with the rest of its path, and `slot` goes. Cost: O(the
    /// fields and the slot's uses), and the removal of a slot's.
    ///
    /// # Panics
    /// `slot` is not a slot of a tuple or structural object type; or a use
    /// other than a `Take`, `Assign` or `Ref` whose path names a field of
    /// it names it.
    pub(super) fn split_slot(&mut self, types: &Types<Concrete>, slot: ValueId) -> Vec<ValueId> {
        let Def::Slot { .. } = self.def_of(slot) else {
            panic!("{slot:?} is not a slot")
        };
        let fields = structural_fields(types, self.ty_of(slot))
            .unwrap_or_else(|| panic!("{slot:?} is not of a tuple or a structural object type"));
        let span = self.slot_spans[&slot];
        let parts: Vec<ValueId> = fields
            .iter()
            .map(|(_, ty)| {
                let part = self.make_value(types, *ty, Def::Slot { index: self.slots.len(), context: None });
                self.slots.push(part);
                self.slot_spans.insert(part, span);
                part
            })
            .collect();
        let sites: Vec<Site> = self.users(slot).collect();
        for site in sites {
            let Site::Inst(block, index) = site else {
                panic!("{slot:?} is named whole by {site:?}; a split rewrites only paths into its fields")
            };
            let inst = &mut self.blocks[block.index()].insts[index];
            let Some((target, path)) = inst.op.place_mut() else {
                panic!("{slot:?} is named whole by {block:?}[{index}] {:?}", inst.op)
            };
            let field = path
                .first()
                .and_then(|seg| fields.iter().position(|(named, _)| named == seg))
                .unwrap_or_else(|| panic!("{block:?}[{index}] names {slot:?} whole, through {path:?}"));
            assert_eq!(*target, RefTarget::Var(slot), "{block:?}[{index}] names {slot:?} as its place");
            *target = RefTarget::Var(parts[field]);
            path.remove(0);
            self.use_sites_mut(slot).remove_one(site);
            self.use_sites_mut(parts[field]).add(site, 1);
        }
        self.remove_value(slot).expect("every use of a split slot went to a field's slot");
        parts
    }

    /// The `Index` or `IndexSet` at `index` of `block`, whose bound is
    /// checked, takes its bound as proven: its check is gone and the
    /// instruction stays. Cost: O(1).
    ///
    /// # Panics
    /// The instruction is not an `Index` or `IndexSet` whose bound is
    /// checked.
    pub(super) fn prove_bound(&mut self, block: BlockId, index: usize) {
        let op = &mut self.blocks[block.index()].insts[index].op;
        match op {
            Op::Index { bound, .. } | Op::IndexSet { bound, .. } if *bound == IndexBound::Checked => {
                *bound = IndexBound::Proven;
            }
            other => panic!("{block:?}[{index}] is {other:?}, not an index whose bound is checked"),
        }
    }

    /// Inserts a `Drop` of `value` at `index` of `block`. Cost: O(operands
    /// of the instructions from `index` on).
    ///
    /// # Panics
    /// `index` is past the block's end, or `value` is an order token, which
    /// no drop ends.
    pub(super) fn insert_drop(&mut self, block: BlockId, index: usize, value: ValueId, span: Span) {
        assert!(index <= self.blocks[block.index()].insts.len(), "{block:?} has no position {index}");
        assert!(!self.is_token(value), "{value:?} is an order token, which no drop ends");
        let inst = self.make_inst(Op::Drop { src: value }, None, None, span);
        self.insert_inst(block, index, inst);
    }

    /// The call at `index` of `block` becomes a copy of `callee`: `block`
    /// ends before it and jumps to the copy's entry; the copy's parameters
    /// are the call's arguments, its captures are `captures`, its order
    /// parameter is the token the call consumed, and each of its returns
    /// jumps to a new continuation block that holds what followed the call.
    /// The continuation's parameters are the call's result and, where the
    /// call is on the chain, the token it yielded; a return of a callee
    /// without an order parameter passes the token the call consumed.
    /// Values of the callee are copied with their types, the module's one
    /// table typing both bodies. It returns the continuation. Cost:
    /// O(the callee's values, instructions and edges), and O(the uses of
    /// the call's result and yield).
    ///
    /// # Panics
    /// The instruction is not a `FunctionCall` whose order, if any, is an
    /// edge; the arguments or `captures` do not match the callee's
    /// parameters or captures in count; the callee is ordered and the call
    /// is not on the chain; or the callee's entry has parameters.
    pub(super) fn inline_call(&mut self, block: BlockId, index: usize, callee: &Body, captures: &[ValueId]) -> BlockId {
        let call = &self.blocks[block.index()].insts[index];
        let Op::FunctionCall { args, .. } = &call.op else {
            panic!("{block:?}[{index}] is {:?}, not a call", call.op)
        };
        let args = args.clone();
        let result = call.result.expect("a call yields a value");
        let edge = call.order.map(|order| match order {
            OrderUse::Edge(edge) => edge,
            OrderUse::Start(_) | OrderUse::Take(_) => panic!("a call's order is an edge"),
        });
        assert_eq!(args.len(), callee.params.len(), "a call passes one argument per parameter of its callee");
        assert_eq!(captures.len(), callee.captures.len(), "a closure holds one value per capture of its body");
        assert!(
            callee.order_param.is_none() || edge.is_some(),
            "an ordered callee body is called on the order chain (effects §3)"
        );
        assert!(callee.block(callee.entry).params.is_empty(), "a body's entry takes no parameters");

        let continuation = self.split_block(block, index + 1);
        let result_param = self.make_like(self.value_def(result), Def::BlockParam(continuation, 0));
        self.blocks[continuation.index()].params.push(result_param);
        self.replace_all_uses(result, result_param);
        if let Some(edge) = edge {
            let token_param = self.make_like(self.value_def(edge.after), Def::BlockParam(continuation, 1));
            self.blocks[continuation.index()].params.push(token_param);
            self.replace_all_uses(edge.after, token_param);
        }
        let removed = self.take_inst(block, index);
        for defined in removed.defs() {
            self.drop_value(defined);
        }

        let mut copied: FxHashMap<ValueId, ValueId> = FxHashMap::default();
        let interface = callee.params.iter().map(|(_, param)| *param).zip(args);
        copied.extend(interface.chain(callee.captures.iter().map(|(_, capture)| *capture).zip(captures.iter().copied())));
        if let (Some(order_param), Some(edge)) = (callee.order_param, edge) {
            copied.insert(order_param, edge.before);
        }
        for slot in &callee.slots {
            let Def::Slot { context, .. } = callee.def_of(*slot) else {
                panic!("{slot:?} stands among the callee's slots and is defined as one")
            };
            let copy = self.make_like(callee.value_def(*slot), Def::Slot { index: self.slots.len(), context });
            self.slots.push(copy);
            self.slot_spans.insert(copy, callee.slot_spans[slot]);
            copied.insert(*slot, copy);
        }
        let base = self.blocks.len();
        let moved = |block: BlockId| BlockId::of_index(base + block.index());
        for (id, callee_block) in callee.blocks() {
            for (at, param) in callee_block.params.iter().enumerate() {
                copied.insert(*param, self.make_like(callee.value_def(*param), Def::BlockParam(moved(id), at)));
            }
            for (at, inst) in callee_block.insts.iter().enumerate() {
                for value in inst.defs() {
                    copied.insert(value, self.make_like(callee.value_def(value), Def::Inst(moved(id), at)));
                }
            }
        }
        let copy_of = |value: &mut ValueId| {
            *value = *copied.get(value).unwrap_or_else(|| panic!("{value:?} is a value the callee defines"));
        };
        for (_, callee_block) in callee.blocks() {
            let insts: Vec<Inst> = callee_block
                .insts
                .iter()
                .map(|inst| {
                    let mut inst = inst.clone();
                    inst.for_each_operand_mut(copy_of);
                    inst.result.iter_mut().for_each(copy_of);
                    match &mut inst.order {
                        Some(OrderUse::Edge(OrderEdge { after, .. })) | Some(OrderUse::Take(after)) => copy_of(after),
                        Some(OrderUse::Start(_)) | None => {}
                    }
                    inst
                })
                .collect();
            let terminator = match &callee_block.terminator {
                Terminator::Return { value, order, .. } => {
                    let mut args = vec![*value];
                    args.iter_mut().for_each(copy_of);
                    match (order, edge) {
                        (Some(order), Some(_)) => {
                            let mut order = *order;
                            copy_of(&mut order);
                            args.push(order);
                        }
                        (None, Some(edge)) => args.push(edge.before),
                        (None, None) => {}
                        (Some(_), None) => panic!("an ordered callee body is called on the order chain (effects §3)"),
                    }
                    Terminator::Jump { to: continuation, args }
                }
                other => {
                    let mut terminator = other.clone();
                    terminator.for_each_operand_mut(copy_of);
                    terminator.for_each_block_mut(|named| *named = moved(*named));
                    terminator
                }
            };
            let params: Vec<ValueId> = callee_block.params.iter().map(|param| copied[param]).collect();
            let insts = insts.into_iter().map(|inst| Inst { id: self.fresh_inst_id(), ..inst }).collect();
            self.new_block(Block { params, insts, terminator });
        }
        for id in (base..self.blocks.len()).map(BlockId::of_index) {
            for at in 0..self.blocks[id.index()].insts.len() {
                let inst = self.blocks[id.index()].insts[at].clone();
                self.record_inst(id, at, &inst);
            }
            let terminator = self.blocks[id.index()].terminator.clone();
            self.record_terminator(id, &terminator);
        }
        for (value, origin) in &callee.debug.val_origins {
            if let Some(copy) = copied.get(value) {
                self.debug.val_origins.insert(*copy, origin.clone());
            }
        }
        self.blocks[block.index()].terminator = Terminator::Jump { to: moved(callee.entry), args: Vec::new() };
        continuation
    }
}

/// The fields of a tuple or a structural object, each with the path segment
/// that names it; `None` for any other type, an extern-defined object among
/// them (compilation §3).
pub(super) fn structural_fields(types: &Types<Concrete>, ty: Ty) -> Option<Vec<(PathSeg, Ty)>> {
    match types.kind(ty) {
        TyKind::Tuple(elements) => Some(elements.iter().enumerate().map(|(index, ty)| (PathSeg::Index(index), *ty)).collect()),
        TyKind::Object(ObjectTy::Structural(fields, _)) => {
            Some(fields.iter().map(|(name, ty)| (PathSeg::Field(*name), *ty)).collect())
        }
        _ => None,
    }
}

/// Where an edit took an instruction out or put one in, for a cursor
/// walking the body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InstShift {
    Inserted { block: BlockId, index: usize },
    Removed { block: BlockId, index: usize },
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

/// On-the-fly SSA construction of one variable over a finished body (Braun
/// et al. 2013, every block sealed): the value at a block's start is its one
/// predecessor's value at its end, or a new parameter where edges from more
/// than one block enter it.
struct Reconstruct {
    ty: Ty,
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
    fn new(body: &Body, ty: Ty, unreached: ValueId) -> Reconstruct {
        let mut edges_into: Vec<SmallVec<[(BlockId, usize); 2]>> = vec![SmallVec::new(); body.blocks.len()];
        for (from, block) in body.blocks() {
            for (position, edge) in block.terminator().edges().iter().enumerate() {
                edges_into[edge.to.index()].push((from, position));
            }
        }
        Reconstruct {
            ty,
            edges_into,
            at_end: FxHashMap::default(),
            at_start: FxHashMap::default(),
            reading: FxHashSet::default(),
            unreached,
            added: Vec::new(),
        }
    }

    /// `value` stands at the entry's start. An entry an edge enters again is
    /// left unseeded, and a read that reaches it panics: the Builder never
    /// makes an entry that is a loop target, so no such body is promoted.
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
            "the entry block is entered by an edge, which the Builder never makes; a value read at its start would \
             need a parameter the entry cannot take"
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
                let param = body.make_value(types, self.ty, Def::UnplacedParam(block));
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
