//! The ir core (ir-core-structure): the CFG, the only IR form. Every IR
//! invariant is kept in `core`; `Builder` (lower) and `Rewriter` (passes)
//! are its two faces, and nothing else writes a body.

mod aspect;
mod body;
mod builder;
mod core;
mod op;
mod rewriter;

pub use aspect::{Aspect, Changes};
pub use body::{Block, BlockId, Body, BodyId, Module, Stages, Terminator};
pub use builder::{Builder, Unsealed};
pub use op::Op;
pub use core::StillUsed;
pub use rewriter::Rewriter;

#[cfg(test)]
mod tests;
