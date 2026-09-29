//! The ir core (ir-core-structure): the CFG, the only IR form. Every IR
//! invariant is kept in `core`; `Builder` (lower) and `Rewriter` (passes)
//! are its two faces, and nothing else writes a body.

mod aspect;
mod body;
mod builder;
mod core;
pub mod edit;
mod op;
mod order;
pub mod print;
mod rewriter;
pub mod takings;

pub use aspect::{Aspect, Changes};
pub use body::{Block, BlockId, Body, BodyId, Definition, Edge, Inst, InstId, Insts, Module, OrderUse, Site, Stages, Terminator};
pub use builder::{Builder, Unsealed};
pub use op::Op;
pub use core::StillUsed;
pub(in crate::ir::cfg) use rewriter::Rewriter;

#[cfg(test)]
mod tests;
