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
use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use super::body::{Block, BlockId, Body, BodyId, Def, Inst, Module, OrderUse, Terminator};
use super::op::Op;
use crate::ir::{OrderEdge, ValueId, fn_effect};
use crate::ty::{Reissue, Ty, TyKind};

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
    span: Option<Span>,
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
    terminator: Option<Terminator>,
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
    pub(super) fn new(module: &'m mut Module, ordered: bool) -> Builder<'m> {
        let entry_block = BlockId::of_index(0);
        let mut body = Body::empty();
        let order = ordered.then(|| {
            let order_ty = module.types.intern(TyKind::Order);
            let token = body.make_value(&module.types, order_ty, Def::OrderParam);
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
            span: None,
            order,
            anyorder_depth: 0,
        }
    }

    /// The block instructions are appended to. Cost: O(1).
    pub fn current_block(&self) -> BlockId {
        self.at
    }

    /// The span every instruction appended from now on carries. Cost: O(1).
    pub fn set_span(&mut self, span: Span) {
        self.span = Some(span);
    }

    /// A parameter of the body, after those declared so far. Cost: O(1).
    pub fn param(&mut self, name: Astr, ty: Ty) -> ValueId {
        let def = Def::Param(self.body.params.len());
        let value = self.body.make_value(&self.module.types, ty, def);
        self.body.params.push((name, value));
        value
    }

    /// A capture of the body, after those declared so far. Cost: O(1).
    pub fn capture(&mut self, name: Astr, ty: Ty) -> ValueId {
        let def = Def::Capture(self.body.captures.len());
        let value = self.body.make_value(&self.module.types, ty, def);
        self.body.captures.push((name, value));
        value
    }

    /// A storage slot of the body holding a `ty`, which `RefTarget::Var`
    /// names. Cost: O(1).
    pub fn slot(&mut self, ty: Ty) -> ValueId {
        let def = Def::Slot(self.body.slots.len());
        let value = self.body.make_value(&self.module.types, ty, def);
        self.body.slots.push(value);
        value
    }

    /// Appends a non-effectful instruction with one result, typed `ty`, and
    /// returns the result. Cost: O(its operands).
    ///
    /// # Panics
    /// See `push_unit`; also, `op` yields no value.
    pub fn push(&mut self, op: Op, ty: Ty) -> ValueId {
        assert!(op.has_result(), "{op:?} yields no value; it is pushed by push_unit");
        self.assert_not_on_chain(&op);
        let result = self.new_result(ty);
        self.append(Inst { op, result: Some(result), order: None, span: self.span() });
        result
    }

    /// Appends a non-effectful instruction with no result. Cost: O(its
    /// operands).
    ///
    /// # Panics
    /// `op` yields a value; it is a call whose effect is not pure, a
    /// `Spawn`, a `TakeEffect` or a `Merge`, which only the core writes; an
    /// operand is not a value of this body; the current block is
    /// terminated; or no span was set.
    pub fn push_unit(&mut self, op: Op) {
        assert!(!op.has_result(), "{op:?} yields a value; it is pushed by push");
        self.assert_not_on_chain(&op);
        self.append(Inst { op, result: None, order: None, span: self.span() });
    }

    /// Appends an effectful call typed `ty` and returns its result. It
    /// consumes the current token and yields the next (effects §3); inside
    /// an `anyorder` scope it consumes the scope's entry token instead, and
    /// what it yields joins the scope's merge. Cost: O(its operands), and
    /// O(1) for the token.
    ///
    /// # Panics
    /// The body is not ordered; `op` is not a `FunctionCall` whose callee's
    /// effect is not pure; an operand is not a value of this body; the
    /// current block is terminated; or no span was set.
    pub fn push_effect(&mut self, op: Op, ty: Ty) -> ValueId {
        assert!(self.order.is_some(), "a body whose effect is pure issues no effect (effects §3)");
        assert!(
            self.effectful_call(&op),
            "push_effect takes a call whose effect is not pure, not {op:?}"
        );
        let at = self.at;
        let before = match &self.order().scope {
            Some(scope) => scope.entry,
            None => self.read_order(OrderVar::Token, at),
        };
        let order_ty = self.order().order_ty;
        let result = self.new_result(ty);
        let after = self.new_result(order_ty);
        let span = self.span();
        self.append(Inst { op, result: Some(result), order: Some(OrderUse::Edge(OrderEdge { before, after })), span });
        match self.order().scope.as_ref().map(|scope| scope.id) {
            Some(scope) => {
                let gathered = self.read_order(OrderVar::Gathered(scope), at);
                let merged = self.new_result(order_ty);
                self.append(Inst {
                    op: Op::Merge { orders: vec![gathered, after] },
                    result: Some(merged),
                    order: None,
                    span,
                });
                self.write_order(OrderVar::Gathered(scope), at, merged);
            }
            None => self.write_order(OrderVar::Token, at, after),
        }
        result
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
    /// `block` is not a reserved block of this body, or an edge into it
    /// disagrees with `params`.
    pub fn make_block(&mut self, block: BlockId, params: &[Ty]) {
        let slot = self.building(block);
        assert!(matches!(slot, Building::Unmade), "{block:?} is made already");
        let params: Vec<ValueId> = params
            .iter()
            .enumerate()
            .map(|(index, ty)| self.body.make_value(&self.module.types, *ty, Def::BlockParam(block, index)))
            .collect();
        for (from, building) in self.blocks.iter().enumerate() {
            let Building::Made(MadeBlock { terminator: Some(terminator), .. }) = building else {
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
    /// # Panics
    /// The current block is terminated; an operand is not a value of this
    /// body; a named block is not a block of this body; an edge disagrees
    /// with its target's parameters; or a `Return` names an order token.
    pub fn terminate(&mut self, mut terminator: Terminator) {
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
            if let Terminator::For { stages, .. } | Terminator::While { stages, .. } = &terminator {
                self.order_mut().loop_entries.insert(stages.body(), at);
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
        current.terminator = Some(terminator);
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

    /// Closes the innermost open `anyorder` scope; closing the outermost
    /// one resumes the chain from the scope's merge. Cost: O(1).
    ///
    /// # Panics
    /// No scope is open, or the current block is terminated.
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
        assert!(current.terminator.is_none(), "{at:?} is terminated; the chain cannot resume in it");
        let Some(scope) = self.order_mut().scope.take() else {
            panic!("an ordered body's outermost anyorder scope is recorded when it opens")
        };
        let gathered = self.read_order(OrderVar::Gathered(scope.id), at);
        self.write_order(OrderVar::Token, at, gathered);
    }

    /// Checks that no open state is left and adds the body to the module.
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
            if let Building::Made(MadeBlock { terminator: Some(terminator), .. }) = building {
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
            let Some(terminator) = terminator else {
                return Err(Unsealed::BlockWithoutTerminator(block));
            };
            made.push(Block { params, insts, terminator });
        }
        for block in made {
            self.body.new_block(block);
        }
        if self.order.is_some() {
            self.complete_order_params();
        }
        let Builder { module, body, .. } = self;
        let id = BodyId(u32::try_from(module.bodies.len()).expect("a module holds at most u32::MAX bodies"));
        module.bodies.push(body);
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
        let param = self.body.make_value(&self.module.types, order_ty, Def::UnplacedParam(block));
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
        let Op::FunctionCall { callee_ty, .. } = op else {
            return false;
        };
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
    }

    fn new_result(&mut self, ty: Ty) -> ValueId {
        let index = self.current().insts.len();
        self.body.make_value(&self.module.types, ty, Def::Inst(self.at, index))
    }

    fn append(&mut self, inst: Inst) {
        let at = self.at;
        let index = self.current().insts.len();
        self.body.record_inst(at, index, &inst);
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

    fn span(&self) -> Span {
        self.span.expect("a span is set before the first instruction")
    }

    /// # Panics
    /// `block` is not a block of this body.
    fn building(&self, block: BlockId) -> &Building {
        self.blocks
            .get(block.index())
            .unwrap_or_else(|| panic!("{block:?} is not a block of this body"))
    }
}
