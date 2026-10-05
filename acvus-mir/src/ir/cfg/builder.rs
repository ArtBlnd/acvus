//! lower's face (ir-core-structure §3): appends at a position, may hold open
//! state, and yields a `Body` only by `seal`, which checks that no open state
//! is left.
//!
//! The order token is an SSA variable only the Builder writes, built on the
//! fly over the Builder's blocks (Braun et al. 2013, "Simple and Efficient
//! Construction of Static Single Assignment Form"). A read in a block with
//! no definition yet adds a parameter to it, completed at `seal` once every
//! edge is known, and the parameters that turn out trivial are removed
//! then. An open `anyorder` scope threads a second such variable, the
//! merge of what its instructions yielded.

use acvus_ast::Span;
use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::FxHashMap;

use super::body::{BlockData, BlockId, Body, BodyContext, BodyId, BodySetChanged, Def, Inst, InstList, Module, OrderUse, Terminator};
use super::order::At;
use super::op::Op;
use crate::ir::{OrderEdge, ValueId, fn_effect};
use crate::ty::{Concrete, Reissue, Ty, TyKind, Types};

/// Open state left at `seal`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsealed {
    BlockWithoutTerminator(BlockId),
    EdgeToUnmadeBlock(BlockId),
    ReservedNeverMade(BlockId),
    OpenAnyorder,
}

/// Made by `Module::build_body`; building borrows the module, and the
/// module's one table types every value.
pub struct Builder<'m> {
    module: &'m mut Module,
    body: Body,
    at: BlockId,
    /// Every block of the body, by id. The body's own block list is filled
    /// from these at `seal`, when each has its terminator.
    blocks: Vec<Building>,
    /// Present exactly when the body is ordered.
    order: Option<OrderThread>,
    anyorder_depth: usize,
}

/// A block while its body is built.
enum Building {
    /// Reserved by `reserve_block`: edges may name it before it is made.
    Unmade,
    Made(MadeBlock),
}

struct MadeBlock {
    params: Vec<ValueId>,
    insts: Vec<Inst>,
    terminator: Option<(Terminator, Span)>,
}

/// A variable of the order chain the Builder threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum OrderVar {
    /// The current order token.
    Token,
    /// The merge of the tokens an `anyorder` scope's instructions yielded,
    /// one variable per scope.
    Gathered(u32),
}

struct IncompleteParam {
    var: OrderVar,
    block: BlockId,
    param: ValueId,
}

struct AnyorderScope {
    id: u32,
    entry: ValueId,
    /// Blocks from this id on were made or reserved inside the scope.
    first_member: usize,
}

/// The order chain of an ordered body.
struct OrderThread {
    order_ty: Ty,
    /// The latest definition of each variable in each block.
    defs: FxHashMap<(OrderVar, BlockId), ValueId>,
    incomplete: Vec<IncompleteParam>,
    /// Every parameter the Builder added, for the trivial-parameter
    /// removal at `seal`.
    added: Vec<ValueId>,
    /// The block whose `For` or `While` enters each loop body block.
    loop_entries: FxHashMap<BlockId, BlockId>,
    /// The token an edge that leaves an `anyorder` scope passes, by source
    /// block and edge position: the merge gathered so far.
    leaving: FxHashMap<(BlockId, usize), ValueId>,
    scope: Option<AnyorderScope>,
    scopes_opened: u32,
}

impl<'m> Builder<'m> {
    /// Starts a body at its entry block, which has no parameters; an
    /// ordered body gets its order parameter here.
    pub(super) fn new(module: &'m mut Module, ordered: bool, ret: Ty) -> Builder<'m> {
        let entry_block = BlockId::of_index(0);
        let mut body = Body::empty(ret);
        let order = ordered.then(|| {
            let order_ty = module.types.intern(TyKind::Order);
            let token = body.make_token(&module.types, order_ty, Def::OrderParam);
            body.order_param = Some(token);
            OrderThread {
                order_ty,
                defs: FxHashMap::from_iter([((OrderVar::Token, entry_block), token)]),
                incomplete: Vec::new(),
                added: Vec::new(),
                loop_entries: FxHashMap::default(),
                leaving: FxHashMap::default(),
                scope: None,
                scopes_opened: 0,
            }
        });
        let entry = MadeBlock { params: Vec::new(), insts: Vec::new(), terminator: None };
        Builder {
            module,
            body,
            at: entry_block,
            blocks: vec![Building::Made(entry)],
            order,
            anyorder_depth: 0,
        }
    }

    /// A Builder for another body of the module, built while this one is
    /// open, as a lambda's body is: this one is paused until the nested one
    /// seals, and that seal yields the body's id for this body's
    /// `MakeClosure`. Cost: O(1).
    pub fn build_nested(&mut self, ordered: bool, ret: Ty) -> Builder<'_> {
        Builder::new(self.module, ordered, ret)
    }

    /// The id of `kind` in the module's one type table, for a type the IR's
    /// shape needs and typeck did not record. Cost: O(kind).
    ///
    /// # Panics
    /// `kind` is a reference to a reference (types §4), or names a type of
    /// another table.
    pub fn intern(&mut self, kind: TyKind<Concrete>) -> Ty {
        if let TyKind::Ref(_, target) = &kind {
            assert!(
                !matches!(self.module.types.kind(*target), TyKind::Ref(..)),
                "a reference's target is never a reference (types §4): {kind:?}"
            );
        }
        self.module.types.intern(kind)
    }

    /// The module's one type table, for reading. Cost: O(1).
    pub fn types(&self) -> &Types<Concrete> {
        &self.module.types
    }

    /// The type `value` was made with. Cost: O(1).
    ///
    /// # Panics
    /// `value` is not a value of this body.
    pub fn ty_of(&self, value: ValueId) -> Ty {
        self.body.ty_of(value)
    }

    /// The block instructions are appended to. Cost: O(1).
    pub fn current_block(&self) -> BlockId {
        self.at
    }

    /// A parameter of the body, after those declared so far; `span` is the
    /// span of its binder. Cost: O(1).
    ///
    /// # Panics
    /// `ty` is `Order`: only the core makes an order token.
    pub fn param(&mut self, name: Astr, ty: Ty, span: Span) -> ValueId {
        let def = Def::Param(self.body.params.len());
        let value = self.body.make_value(&self.module.types, ty, def);
        self.body.params.push((name, value));
        self.body.binder_spans.insert(value, span);
        value
    }

    /// A capture of the body, after those declared so far; `span` is the
    /// span of its binder. Cost: O(1).
    ///
    /// # Panics
    /// `ty` is `Order`: only the core makes an order token.
    pub fn capture(&mut self, name: Astr, ty: Ty, span: Span) -> ValueId {
        let def = Def::Capture(self.body.captures.len());
        let value = self.body.make_value(&self.module.types, ty, def);
        self.body.captures.push((name, value));
        self.body.binder_spans.insert(value, span);
        value
    }

    /// A storage slot of the body holding a `ty`, which `RefTarget::Var`
    /// names; `span` is the span of the binding it stores. Cost: O(1).
    ///
    /// # Panics
    /// `ty` is `Order`: only the core makes an order token.
    pub fn slot(&mut self, ty: Ty, span: Span) -> ValueId {
        let def = Def::Slot { index: self.body.slots.len() };
        let value = self.body.make_value(&self.module.types, ty, def);
        self.body.slots.push(value);
        self.body.slot_spans.insert(value, span);
        value
    }

    /// States that the body names `context`, of type `ty`, volatile as its
    /// declaration states, which `RefTarget::Context` then addresses
    /// (context §3). Cost: O(the contexts named so far).
    ///
    /// # Panics
    /// The body names `context` already.
    pub fn context(&mut self, context: QualifiedRef, ty: Ty, volatile: bool) {
        assert!(self.body.named_context(context).is_none(), "the body names {context:?} once");
        self.body.contexts.push(BodyContext { context, ty, volatile });
    }

    /// Appends `op`, a `Take` or a `Ref`, typed `ty`, and returns its
    /// result: as an effect where it names a volatile context, whose every
    /// place operation is one (context §3), and as `push` does otherwise.
    /// Cost: `push`'s or `push_effect`'s.
    pub fn push_place(&mut self, op: Op, ty: Ty, span: Span) -> ValueId {
        match self.body.on_volatile_context(&op) {
            true => self.push_effect(op, ty, span),
            false => self.push(op, ty, span),
        }
    }

    /// Appends `op`, an `Assign`, as `push_place` appends a `Take`. Cost:
    /// `push_unit`'s or `push_effect_unit`'s.
    pub fn push_place_unit(&mut self, op: Op, span: Span) {
        match self.body.on_volatile_context(&op) {
            true => self.push_effect_unit(op, span),
            false => self.push_unit(op, span),
        }
    }

    /// Appends a non-effectful instruction with one result, typed `ty`, and
    /// returns the result. Cost: O(its operands).
    ///
    /// # Panics
    /// See `push_unit`; also, `op` yields no value, or `ty` is `Order`,
    /// since only the core makes an order token.
    pub fn push(&mut self, op: Op, ty: Ty, span: Span) -> ValueId {
        assert!(op.has_result(), "{op:?} yields no value; it is pushed by push_unit");
        self.assert_not_on_chain(&op);
        let result = self.new_result(ty);
        let inst = self.body.make_inst(op, Some(result), None, span);
        self.append(inst);
        result
    }

    /// Appends a non-effectful instruction with no result. Cost: O(its
    /// operands).
    ///
    /// # Panics
    /// `op` yields a value; it is a call whose effect is not pure, or a place
    /// operation on a volatile context, which are effects; it is a `Spawn`,
    /// a `TakeEffect` or a `Merge`, which only the core writes; it names a
    /// context the body does not name; an operand is not a value of this
    /// body; the current block is terminated.
    pub fn push_unit(&mut self, op: Op, span: Span) {
        assert!(!op.has_result(), "{op:?} yields a value; it is pushed by push");
        self.assert_not_on_chain(&op);
        let inst = self.body.make_inst(op, None, None, span);
        self.append(inst);
    }

    /// Appends an effect with one result, typed `ty`, and returns the
    /// result: a call whose callee's effect is not pure, or a `Take` or a
    /// `Ref` of a volatile context (context §3). It consumes the current
    /// token and yields the next (effects §3); inside an `anyorder` scope it
    /// consumes the scope's entry token instead, and what it yields joins the
    /// scope's merge, a `Merge` placed with the effect's `span`. Cost: O(its
    /// operands), and O(1) for the token.
    ///
    /// # Panics
    /// The body is not ordered; `op` is neither such a call nor a place
    /// operation on a volatile context the body names; an operand is not a
    /// value of this body; the current block is terminated.
    pub fn push_effect(&mut self, op: Op, ty: Ty, span: Span) -> ValueId {
        assert!(
            self.effectful_call(&op) || self.body.on_volatile_context(&op),
            "push_effect takes a call whose effect is not pure or a place operation on a volatile context, not {op:?}"
        );
        let result = self.new_result(ty);
        self.append_effect(op, Some(result), span);
        result
    }

    /// Appends an effect with no result: an `Assign` into a volatile context
    /// (context §3). It stands on the order chain as `push_effect` states.
    /// Cost: O(its operands), and O(1) for the token.
    ///
    /// # Panics
    /// The body is not ordered; `op` yields a value or is not a place
    /// operation on a volatile context the body names; an operand is not a
    /// value of this body; the current block is terminated.
    pub fn push_effect_unit(&mut self, op: Op, span: Span) {
        assert!(!op.has_result(), "{op:?} yields a value; it is pushed by push_effect");
        assert!(
            self.body.on_volatile_context(&op),
            "push_effect_unit takes a place operation on a volatile context, not {op:?}"
        );
        self.append_effect(op, None, span);
    }

    fn append_effect(&mut self, op: Op, result: Option<ValueId>, span: Span) {
        assert!(self.order.is_some(), "a body whose effect is pure issues no effect (effects §3)");
        let at = self.at;
        let before = match &self.order().scope {
            Some(scope) => scope.entry,
            None => self.read_order(OrderVar::Token, at),
        };
        let order_ty = self.order().order_ty;
        let after = self.new_token_result(order_ty);
        let inst = self.body.make_inst(op, result, Some(OrderUse::Edge(OrderEdge { before, after })), span);
        self.append(inst);
        match self.order().scope.as_ref().map(|scope| scope.id) {
            Some(scope) => {
                let gathered = self.read_order(OrderVar::Gathered(scope), at);
                let merged = self.new_token_result(order_ty);
                let merge = self.body.make_inst(Op::Merge { orders: vec![gathered, after] }, Some(merged), None, span);
                self.append(merge);
                self.write_order(OrderVar::Gathered(scope), at, merged);
            }
            None => self.write_order(OrderVar::Token, at, after),
        }
    }

    /// Test only: the order token current at the end of the current block.
    ///
    /// # Panics
    /// The body is not ordered.
    #[cfg(test)]
    pub(crate) fn current_token(&mut self) -> ValueId {
        assert!(self.order.is_some(), "a body whose effect is pure has no order token (effects §3)");
        let at = self.at;
        self.read_order(OrderVar::Token, at)
    }

    /// Test only: appends a `Merge` of `orders`, which the Builder itself
    /// never writes in this shape, such as a merge of one token or one
    /// repeating another's set, and the chain goes on from its result.
    /// Cost: O(|orders|).
    ///
    /// # Panics
    /// The body is not ordered, or an input is not a token of it.
    #[cfg(test)]
    pub(crate) fn push_merge(&mut self, orders: Vec<ValueId>, span: Span) -> ValueId {
        assert!(self.order.is_some(), "a body whose effect is pure has no order token to merge (effects §3)");
        for order in &orders {
            assert!(self.body.is_token(*order), "{order:?} is not an order token");
        }
        let order_ty = self.order().order_ty;
        let merged = self.new_token_result(order_ty);
        let merge = self.body.make_inst(Op::Merge { orders }, Some(merged), None, span);
        self.append(merge);
        let at = self.at;
        self.write_order(OrderVar::Token, at, merged);
        merged
    }

    /// A block with parameters of `params`, made now. Cost: O(|params|).
    pub fn new_block(&mut self, params: &[Ty]) -> BlockId {
        let block = BlockId::of_index(self.blocks.len());
        self.blocks.push(Building::Unmade);
        self.make_block(block, params);
        block
    }

    /// A block edges may name before it is made by `make_block`; `seal`
    /// refuses a reserved block left unmade. Cost: O(1).
    pub fn reserve_block(&mut self) -> BlockId {
        let block = BlockId::of_index(self.blocks.len());
        self.blocks.push(Building::Unmade);
        block
    }

    /// Makes the reserved `block` with parameters of `params`, which agree
    /// with every edge already made into it. Cost: O(|params|) and O(the
    /// terminators made so far).
    ///
    /// # Panics
    /// `block` is not a reserved block of this body; an edge into it
    /// disagrees with `params`; or a parameter is of `Order` type, since
    /// only the core makes an order token, and places a block's.
    pub fn make_block(&mut self, block: BlockId, params: &[Ty]) {
        let slot = self.building(block);
        assert!(matches!(slot, Building::Unmade), "{block:?} is made already");
        let params: Vec<ValueId> = params
            .iter()
            .enumerate()
            .map(|(index, ty)| self.body.make_value(&self.module.types, *ty, Def::BlockParam(block, index)))
            .collect();
        for (from, building) in self.blocks.iter().enumerate() {
            let Building::Made(MadeBlock { terminator: Some((terminator, _)), .. }) = building else {
                continue;
            };
            for edge in terminator.edges().iter().filter(|edge| edge.to == block) {
                self.body.assert_edge_agrees(BlockId::of_index(from), &params, edge);
            }
        }
        self.blocks[block.index()] = Building::Made(MadeBlock { params, insts: Vec::new(), terminator: None });
    }

    /// The parameters of a made block, as its maker stated them. Cost: O(1).
    ///
    /// # Panics
    /// `block` is not a made block of this body.
    pub fn block_params(&self, block: BlockId) -> &[ValueId] {
        let Building::Made(made) = self.building(block) else {
            panic!("{block:?} is reserved and not made yet")
        };
        &made.params
    }

    /// Cost: O(1).
    ///
    /// # Panics
    /// `block` is not a made block of this body, or it has its terminator.
    pub fn switch_to(&mut self, block: BlockId) {
        let Building::Made(made) = self.building(block) else {
            panic!("{block:?} is reserved and not made yet")
        };
        assert!(made.terminator.is_none(), "{block:?} is terminated; nothing is appended to it");
        self.at = block;
    }

    /// Ends the current block with `terminator`. Every block it names is a
    /// block of this body, and every edge into a made block agrees with its
    /// parameters; an edge into a reserved block is checked when that block
    /// is made. A `Return` is written with no order token, and the Builder
    /// fills in the current one where the body is ordered; inside an
    /// `anyorder` scope, a `Return` and every edge to a block outside the
    /// scope pass the scope's merge so far. Cost: O(the terminator's
    /// operands and the parameters of its targets).
    ///
    /// `span` is the span of the construct that made the terminator; a
    /// `Return` carries its own, and `span` is that one.
    ///
    /// # Panics
    /// The current block is terminated; an operand is not a value of this
    /// body; a named block is not a block of this body; an edge disagrees
    /// with its target's parameters; a `Return` names an order token, or
    /// carries a span other than `span`.
    pub fn terminate(&mut self, mut terminator: Terminator, span: Span) {
        if let Terminator::Return { span: own, .. } = &terminator {
            assert_eq!(*own, span, "a Return carries its own span, and terminate is given that one");
        }
        let at = self.at;
        let Building::Made(current) = self.building(at) else {
            panic!("the current block {at:?} is made")
        };
        assert!(current.terminator.is_none(), "{at:?} is terminated already");
        terminator.for_each_block(|named| {
            self.building(named);
        });
        for edge in terminator.edges() {
            if let Building::Made(target) = self.building(edge.to) {
                self.body.assert_edge_agrees(at, &target.params, &edge);
            }
        }
        if let Some(thread) = &self.order {
            let scope = thread.scope.as_ref().map(|scope| (scope.id, scope.first_member));
            if let Some((id, first_member)) = scope {
                let gathered = self.read_order(OrderVar::Gathered(id), at);
                let leaving: Vec<usize> = terminator
                    .edges()
                    .iter()
                    .enumerate()
                    .filter(|(_, edge)| edge.to.index() < first_member)
                    .map(|(position, _)| position)
                    .collect();
                for position in leaving {
                    self.order_mut().leaving.insert((at, position), gathered);
                }
            }
            if let Terminator::For { body, .. } | Terminator::While { body, .. } = &terminator {
                self.order_mut().loop_entries.insert(*body, at);
            }
        }
        if let Terminator::Return { order, .. } = &mut terminator {
            assert!(order.is_none(), "no caller names an order token; the Builder fills in a Return's");
            let var = self.order.as_ref().map(|thread| match &thread.scope {
                Some(scope) => OrderVar::Gathered(scope.id),
                None => OrderVar::Token,
            });
            *order = var.map(|var| self.read_order(var, at));
        }
        self.body.record_terminator(at, &terminator);
        let Building::Made(current) = &mut self.blocks[at.index()] else {
            panic!("the current block {at:?} is made")
        };
        current.terminator = Some((terminator, span));
    }

    /// Opens an `anyorder` scope (effects §3): every effectful call inside
    /// consumes the token current here. A scope opened inside an open one
    /// changes nothing. Cost: O(1).
    pub fn open_anyorder(&mut self) {
        self.anyorder_depth += 1;
        if self.anyorder_depth > 1 || self.order.is_none() {
            return;
        }
        let at = self.at;
        let entry = self.read_order(OrderVar::Token, at);
        let first_member = self.blocks.len();
        let thread = self.order_mut();
        let id = thread.scopes_opened;
        thread.scopes_opened += 1;
        thread.defs.insert((OrderVar::Gathered(id), at), entry);
        thread.scope = Some(AnyorderScope { id, entry, first_member });
    }

    /// Closes the innermost open `anyorder` scope. Closing the outermost one
    /// resumes the chain from the scope's merge in the current block where
    /// that block is open. Where it is terminated, no run reaches the
    /// scope's end: a `Return` or an edge out of the scope passed the merge
    /// at its own terminator (`terminate`), and nothing resumes. Cost: O(1).
    ///
    /// # Panics
    /// No scope is open.
    pub fn close_anyorder(&mut self) {
        assert!(self.anyorder_depth > 0, "close_anyorder without an open anyorder scope");
        self.anyorder_depth -= 1;
        if self.anyorder_depth > 0 || self.order.is_none() {
            return;
        }
        let at = self.at;
        let Building::Made(current) = self.building(at) else {
            panic!("the current block {at:?} is made")
        };
        let reached = current.terminator.is_none();
        let Some(scope) = self.order_mut().scope.take() else {
            panic!("an ordered body's outermost anyorder scope is recorded when it opens")
        };
        if reached {
            let gathered = self.read_order(OrderVar::Gathered(scope.id), at);
            self.write_order(OrderVar::Token, at, gathered);
        }
    }

    /// Checks that no open state is left, adds the body to the module and
    /// records that the module's body set changed.
    /// It reports an open `anyorder` scope first, then the first block, by
    /// id, left without its terminator or reserved and never made. In an
    /// ordered body it then completes the order token's parameters and
    /// removes the trivial ones. Cost: O(blocks), and for the token
    /// O(P · (edges + uses)) per round of removal over its P parameters.
    pub fn seal(mut self) -> Result<BodyId, Unsealed> {
        if self.anyorder_depth > 0 {
            return Err(Unsealed::OpenAnyorder);
        }
        let mut named = vec![false; self.blocks.len()];
        for building in &self.blocks {
            if let Building::Made(MadeBlock { terminator: Some((terminator, _)), .. }) = building {
                terminator.for_each_block(|block| named[block.index()] = true);
            }
        }
        let mut made = Vec::with_capacity(self.blocks.len());
        for (index, building) in std::mem::take(&mut self.blocks).into_iter().enumerate() {
            let block = BlockId::of_index(index);
            let MadeBlock { params, insts, terminator } = match building {
                Building::Made(made) => made,
                Building::Unmade if named[index] => return Err(Unsealed::EdgeToUnmadeBlock(block)),
                Building::Unmade => return Err(Unsealed::ReservedNeverMade(block)),
            };
            let Some((terminator, terminator_span)) = terminator else {
                return Err(Unsealed::BlockWithoutTerminator(block));
            };
            made.push((BlockData { params, list: InstList::default(), terminator, terminator_span }, insts));
        }
        for (block, insts) in made {
            let id = self.body.new_block(block);
            for inst in insts {
                self.body.place(At::End(id), inst);
            }
        }
        if self.order.is_some() {
            self.complete_order_params();
        }
        let Builder { module, body, .. } = self;
        let id = BodyId(u32::try_from(module.bodies.len()).expect("a module holds at most u32::MAX bodies"));
        module.bodies.push(body);
        module.body_set_change = Some(BodySetChanged::record());
        Ok(id)
    }

    /// Completes every parameter a read added, over the edges of the
    /// finished body, then removes the trivial ones.
    fn complete_order_params(&mut self) {
        let mut edges_into: Vec<Vec<(BlockId, usize)>> = vec![Vec::new(); self.body.blocks.len()];
        for (from, block) in self.body.blocks() {
            for (position, edge) in block.terminator().edges().iter().enumerate() {
                edges_into[edge.to.index()].push((from, position));
            }
        }
        while let Some(IncompleteParam { var, block, param }) = self.order_mut().incomplete.pop() {
            let mut incoming = Vec::with_capacity(edges_into[block.index()].len());
            for &(from, position) in &edges_into[block.index()] {
                let leaving = match var {
                    OrderVar::Token => self.order().leaving.get(&(from, position)).copied(),
                    OrderVar::Gathered(_) => None,
                };
                let value = match leaving {
                    Some(value) => value,
                    None => self.read_order(var, from),
                };
                incoming.push((from, position, value));
            }
            self.body.place_param(block, param, &incoming);
        }
        let added = std::mem::take(&mut self.order_mut().added);
        self.body.remove_trivial_params(&added);
    }

    /// The value of `var` at the end of what is written in `block` so far.
    ///
    /// # Panics
    /// `var` is read in the entry block before it is defined there.
    fn read_order(&mut self, var: OrderVar, block: BlockId) -> ValueId {
        if let Some(value) = self.order().defs.get(&(var, block)) {
            return *value;
        }
        if let Some(header) = self.order().loop_entries.get(&block).copied() {
            let value = self.read_order(var, header);
            self.write_order(var, block, value);
            return value;
        }
        assert_ne!(block, self.body.entry, "{var:?} is read in the entry block before it is defined");
        let order_ty = self.order().order_ty;
        let param = self.body.make_token(&self.module.types, order_ty, Def::UnplacedParam(block));
        let thread = self.order_mut();
        thread.incomplete.push(IncompleteParam { var, block, param });
        thread.added.push(param);
        thread.defs.insert((var, block), param);
        param
    }

    fn write_order(&mut self, var: OrderVar, block: BlockId, value: ValueId) {
        self.order_mut().defs.insert((var, block), value);
    }

    fn order(&self) -> &OrderThread {
        self.order.as_ref().expect("the order chain is read only in an ordered body")
    }

    fn order_mut(&mut self) -> &mut OrderThread {
        self.order.as_mut().expect("the order chain is written only in an ordered body")
    }

    fn effectful_call(&self, op: &Op) -> bool {
        let callee_ty = match op {
            Op::FunctionCall { callee_ty, .. } => *callee_ty,
            // `ArrayMap` calls its function value `f` on each element
            // (`Intrinsic`), so it is a call of `f`'s type.
            Op::Intrinsic { intrinsic: crate::ir::Intrinsic::ArrayMap, args } => {
                let [.., f] = args[..] else { panic!("`ArrayMap` takes `[a_1, …, a_k, f]`") };
                self.body.ty_of(f)
            }
            _ => return false,
        };
        let callee_ty = &callee_ty;
        let Some(effect) = fn_effect(&self.module.types, *callee_ty) else {
            panic!("a call's callee type is a function type")
        };
        effect.reissue() != Reissue::Pure
    }

    fn assert_not_on_chain(&self, op: &Op) {
        assert!(
            !matches!(op, Op::Spawn { .. } | Op::TakeEffect { .. } | Op::Merge { .. }),
            "{op:?} is written only by the core (effects §3, §6)"
        );
        assert!(!self.effectful_call(op), "a call whose effect is not pure is pushed by push_effect");
        assert!(
            !self.body.on_volatile_context(op),
            "a place operation on a volatile context is an effect (context §3), pushed by push_effect"
        );
    }

    /// A result of the next instruction made, which is appended next.
    fn new_result(&mut self, ty: Ty) -> ValueId {
        self.current();
        self.body.make_value(&self.module.types, ty, Def::Inst(self.body.peek_inst_id()))
    }

    /// A token the next instruction appended yields or, for a `Merge`,
    /// results in.
    fn new_token_result(&mut self, order_ty: Ty) -> ValueId {
        self.current();
        self.body.make_token(&self.module.types, order_ty, Def::Inst(self.body.peek_inst_id()))
    }

    fn append(&mut self, inst: Inst) {
        let at = self.at;
        self.current();
        self.body.record_inst(&inst);
        let Building::Made(current) = &mut self.blocks[at.index()] else {
            panic!("the current block {at:?} is made")
        };
        current.insts.push(inst);
    }

    /// # Panics
    /// The current block is terminated.
    fn current(&self) -> &MadeBlock {
        let Building::Made(current) = self.building(self.at) else {
            panic!("the current block {:?} is made", self.at)
        };
        assert!(current.terminator.is_none(), "{:?} is terminated; nothing is appended to it", self.at);
        current
    }

    /// # Panics
    /// `block` is not a block of this body.
    fn building(&self, block: BlockId) -> &Building {
        self.blocks
            .get(block.index())
            .unwrap_or_else(|| panic!("{block:?} is not a block of this body"))
    }
}
