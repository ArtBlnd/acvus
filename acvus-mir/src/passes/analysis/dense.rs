//! The dense engine: a state per program point over the CFG, forward or
//! backward. Cost: each block's state rises at most the lattice's height
//! and each rise revisits its successors (O(edges × height)); a state of
//! infinite height widens at loop heads.

use rustc_hash::FxHashMap;

use crate::ir::cfg::{BlockId, Body, Op, Terminator};
use crate::passes::domain::JoinSemiLattice;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

/// A dense analysis: its state lattice and how an instruction and a
/// terminator change it. Storage and calls are read here, calls through
/// their declared facts only.
pub trait DenseAnalysis {
    type State: JoinSemiLattice;

    const DIRECTION: Direction;

    fn entry(&self, body: &Body) -> Self::State;
    fn transfer(&self, op: &Op, state: &mut Self::State);
    fn transfer_terminator(&self, terminator: &Terminator, state: &mut Self::State);
}

pub struct DenseFacts<S>
where
    S: JoinSemiLattice,
{
    at_entry: FxHashMap<BlockId, S>,
}

impl<S> DenseFacts<S>
where
    S: JoinSemiLattice,
{
    /// The state at a block's start (forward) or end (backward).
    ///
    /// # Panics
    /// `block` is not a block of the body the facts were computed for.
    pub fn at(&self, block: BlockId) -> &S {
        self.at_entry.get(&block).expect("dense facts cover every block of their body")
    }
}

pub fn solve_dense<A>(analysis: &A, body: &Body) -> DenseFacts<A::State>
where
    A: DenseAnalysis,
{
    todo!("passes::analysis::solve_dense")
}
