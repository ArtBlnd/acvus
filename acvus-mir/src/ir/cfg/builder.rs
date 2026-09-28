//! lower's face (ir-core-structure §3): appends at a position, may hold open
//! state, and yields a `Body` only by `seal`, which checks that no open state
//! is left.

use acvus_ast::Span;
use acvus_utils::Astr;

use super::body::{Block, BlockId, Body, BodyId, Def, Module, Terminator};
use super::op::Op;
use crate::ir::ValueId;
use crate::ty::Ty;

/// Open state left at `seal`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsealed {
    BlockWithoutTerminator(BlockId),
    EdgeToUnmadeBlock(BlockId),
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
    insts: Vec<(Op, Span)>,
    terminator: Option<Terminator>,
}

impl<'m> Builder<'m> {
    /// Starts a body at its entry block, which has no parameters.
    pub(super) fn new(module: &'m mut Module) -> Builder<'m> {
        let entry = MadeBlock { params: Vec::new(), insts: Vec::new(), terminator: None };
        Builder {
            module,
            body: Body::empty(),
            at: BlockId::of_index(0),
            blocks: vec![Building::Made(entry)],
            anyorder_depth: 0,
        }
    }

    /// The block instructions are appended to. Cost: O(1).
    pub fn current_block(&self) -> BlockId {
        self.at
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

    /// Appends a non-effectful instruction; its result is typed `ty`.
    pub fn push(&mut self, op: Op, ty: Ty) -> ValueId {
        todo!("ir::cfg::Builder::push: blocked on how a caller names the result an Op carries (report Q1, Q2)")
    }

    /// Appends an effectful instruction; the builder threads the order token.
    pub fn push_effect(&mut self, op: Op, ty: Ty) -> ValueId {
        todo!("ir::cfg::Builder::push_effect: blocked on Q1 and on how the token crosses an edge (report Q3)")
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

    /// The parameters of a made block. Cost: O(1).
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
    /// is made. A `Return` carries an order token exactly when the body has
    /// an order parameter. Cost: O(the terminator's operands and the
    /// parameters of its targets).
    ///
    /// # Panics
    /// The current block is terminated; an operand is not a value of this
    /// body; a named block is not a block of this body; an edge disagrees
    /// with its target's parameters; or a `Return`'s order token is present
    /// where the body has no order parameter, or absent where it has one.
    pub fn terminate(&mut self, terminator: Terminator) {
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
        if let Terminator::Return { order, .. } = &terminator {
            assert_eq!(
                order.is_some(),
                self.body.order_param.is_some(),
                "a body returns an order token exactly when it takes one (effects §3)"
            );
        }
        self.body.record_terminator(at, &terminator);
        let Building::Made(current) = &mut self.blocks[at.index()] else {
            panic!("the current block {at:?} is made")
        };
        current.terminator = Some(terminator);
    }

    /// Opens an `anyorder` scope (effects §3). Cost: O(1).
    pub fn open_anyorder(&mut self) {
        match self.body.order_param {
            // A body with no order token issues no effect, so the scope
            // has no chain to cut.
            None => self.anyorder_depth += 1,
            Some(_) => todo!("ir::cfg::Builder::open_anyorder in an ordered body: blocked on report Q3"),
        }
    }

    /// Closes the innermost open `anyorder` scope. Cost: O(1).
    ///
    /// # Panics
    /// No scope is open.
    pub fn close_anyorder(&mut self) {
        assert!(self.anyorder_depth > 0, "close_anyorder without an open anyorder scope");
        match self.body.order_param {
            None => self.anyorder_depth -= 1,
            Some(_) => todo!("ir::cfg::Builder::close_anyorder in an ordered body: blocked on report Q3"),
        }
    }

    /// Checks that no open state is left and adds the body to the module.
    /// It reports an open `anyorder` scope first, then the first block, by
    /// id, left without its terminator or reserved and never made. Cost:
    /// O(blocks).
    pub fn seal(self) -> Result<BodyId, Unsealed> {
        let Builder { module, mut body, blocks, anyorder_depth, .. } = self;
        if anyorder_depth > 0 {
            return Err(Unsealed::OpenAnyorder);
        }
        let mut made = Vec::with_capacity(blocks.len());
        for (index, building) in blocks.into_iter().enumerate() {
            let block = BlockId::of_index(index);
            let MadeBlock { params, insts, terminator } = match building {
                Building::Made(made) => made,
                Building::Unmade => return Err(Unsealed::EdgeToUnmadeBlock(block)),
            };
            let Some(terminator) = terminator else {
                return Err(Unsealed::BlockWithoutTerminator(block));
            };
            made.push(Block { params, insts, terminator });
        }
        for block in made {
            body.new_block(block);
        }
        let id = BodyId(u32::try_from(module.bodies.len()).expect("a module holds at most u32::MAX bodies"));
        module.bodies.push(body);
        Ok(id)
    }

    /// # Panics
    /// `block` is not a block of this body.
    fn building(&self, block: BlockId) -> &Building {
        self.blocks
            .get(block.index())
            .unwrap_or_else(|| panic!("{block:?} is not a block of this body"))
    }
}
