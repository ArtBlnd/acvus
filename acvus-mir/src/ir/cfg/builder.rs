//! lower's face (ir-core-structure §3): appends at a position, may hold open
//! state, and yields a `Body` only by `seal`, which checks that no open state
//! is left.

use super::body::{BlockId, Body, Terminator};
use super::op::Op;
use crate::ir::ValueId;
use crate::ty::{Concrete, Ty, Types};

/// Open state left at `seal`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsealed {
    BlockWithoutTerminator(BlockId),
    EdgeToUnmadeBlock(BlockId),
    OpenAnyorder,
}

pub struct Builder<'t> {
    body: Body,
    types: &'t mut Types<Concrete>,
    at: BlockId,
}

impl<'t> Builder<'t> {
    pub fn new(types: &'t mut Types<Concrete>) -> Self {
        todo!("ir::cfg::Builder::new")
    }

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

    pub fn seal(self) -> Result<Body, Unsealed> {
        todo!("ir::cfg::Builder::seal")
    }
}
