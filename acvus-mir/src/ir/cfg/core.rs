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

use super::body::{BlockId, Body};
use crate::ir::ValueId;
use crate::ty::{Concrete, Ty, Types};

/// A value that still has uses, removed without a replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StillUsed {
    pub value: ValueId,
}

impl Body {
    pub(super) fn make_value(&mut self, types: &Types<Concrete>, ty: Ty) -> ValueId {
        todo!("ir::cfg::core::make_value")
    }

    pub(super) fn replace_all_uses(&mut self, of: ValueId, with: ValueId) {
        todo!("ir::cfg::core::replace_all_uses")
    }

    pub(super) fn remove_value(&mut self, value: ValueId) -> Result<(), StillUsed> {
        todo!("ir::cfg::core::remove_value")
    }

    pub(super) fn new_block(&mut self) -> BlockId {
        todo!("ir::cfg::core::new_block")
    }
}
