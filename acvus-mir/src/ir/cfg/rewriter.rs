//! passes' face (ir-core-structure §4): edits over a body that is whole
//! between any two operations, recording what each changed.

use super::aspect::Changes;
use super::body::{BlockId, Body};
use super::core::StillUsed;
use crate::ir::ValueId;
use crate::ty::{Concrete, Types};

pub struct Rewriter<'b> {
    body: &'b mut Body,
    types: &'b mut Types<Concrete>,
    changes: Changes,
}

impl<'b> Rewriter<'b> {
    pub fn new(body: &'b mut Body, types: &'b mut Types<Concrete>) -> Self {
        Self { body, types, changes: Changes::default() }
    }

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
