//! passes' face (ir-core-structure §4): edits over a body that is whole
//! between any two operations, recording what each changed.

use super::aspect::Changes;
use super::body::{BlockId, Body, Module};
use super::core::StillUsed;
use crate::ir::ValueId;

/// Made by `Module::rewrite`, over one body of the module.
pub struct Rewriter<'m> {
    module: &'m mut Module,
    body: super::BodyId,
    changes: Changes,
}

impl<'m> Rewriter<'m> {

    pub fn replace_all_uses(&mut self, of: ValueId, with: ValueId) {
        todo!("ir::cfg::Rewriter::replace_all_uses")
    }

    pub fn remove(&mut self, value: ValueId) -> Result<(), StillUsed> {
        todo!("ir::cfg::Rewriter::remove")
    }

    pub fn split_block(&mut self, block: BlockId, at: usize) -> BlockId {
        todo!("ir::cfg::Rewriter::split_block")
    }

    pub fn split_call(&mut self, call: ValueId) {
        todo!("ir::cfg::Rewriter::split_call")
    }

    pub fn finish(self) -> Changes {
        self.changes
    }
}
