//! lower's face (ir-core-structure §3): appends at a position, may hold open
//! state, and yields a `Body` only by `seal`, which checks that no open state
//! is left.

use super::body::{BlockId, Body, BodyId, Module, Terminator};
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
}

impl<'m> Builder<'m> {

    /// Appends a non-effectful instruction; its result is typed `ty`.
    pub fn push(&mut self, op: Op, ty: Ty) -> ValueId {
        todo!("ir::cfg::Builder::push")
    }

    /// Appends an effectful instruction; the builder threads the order token.
    pub fn push_effect(&mut self, op: Op, ty: Ty) -> ValueId {
        todo!("ir::cfg::Builder::push_effect")
    }

    pub fn new_block(&mut self, params: &[Ty]) -> BlockId {
        todo!("ir::cfg::Builder::new_block")
    }

    pub fn switch_to(&mut self, block: BlockId) {
        todo!("ir::cfg::Builder::switch_to")
    }

    pub fn terminate(&mut self, terminator: Terminator) {
        todo!("ir::cfg::Builder::terminate")
    }

    pub fn open_anyorder(&mut self) {
        todo!("ir::cfg::Builder::open_anyorder")
    }

    pub fn close_anyorder(&mut self) {
        todo!("ir::cfg::Builder::close_anyorder")
    }

    /// Checks that no open state is left and adds the body to the module.
    pub fn seal(self) -> Result<BodyId, Unsealed> {
        todo!("ir::cfg::Builder::seal")
    }
}
