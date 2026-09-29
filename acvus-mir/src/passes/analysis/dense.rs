//! The dense engine: a state per block over the CFG, forward or backward,
//! by a worklist in reverse postorder, or postorder when backward (Kildall,
//! "A Unified Approach to Global Program Optimization", 1973).
//!
//! Cost: a block's stored state only rises, and each rise re-runs the
//! transfer of every block it flows into, so the engine is O(h · (blocks +
//! edges) · the transfer's cost per block), where h bounds how often one
//! state rises: the lattice's height, or, for a state of infinite height,
//! the length of its widening's chains. Every cycle of the CFG holds a back
//! edge of the depth-first walk from the entry, and the stored state of that
//! edge's target, a loop header, widens in either direction, so h is finite
//! whenever the widening terminates.

use std::collections::BTreeSet;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::ir::ValueId;
use crate::ir::cfg::{BlockId, Body, Edge, Inst, Terminator};
use crate::passes::domain::{JoinSemiLattice, Widen};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

/// A dense analysis: its state lattice and how an instruction and a
/// terminator change it. Storage and calls are read here, calls through
/// their declared facts only.
pub trait DenseAnalysis {
    type State: Widen;

    const DIRECTION: Direction;

    fn entry(&self, body: &Body) -> Self::State;
    fn transfer(&self, inst: &Inst, state: &mut Self::State);
    fn transfer_terminator(&self, terminator: &Terminator, state: &mut Self::State);
    /// A state flowing along `edge`, whose target's parameters are
    /// `target_params`: forward from the source's end to the target's start,
    /// backward from the target's start to the source's end. The edge
    /// defines the target's parameters and uses its arguments.
    fn transfer_edge(&self, edge: &Edge<'_>, target_params: &[ValueId], state: &mut Self::State);
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

/// Every block's state. Forward, the entry block starts from `entry`; backward,
/// every block with no successor ends in it. A block the entry does not
/// reach keeps `bottom`. Within a block, a forward analysis sees its
/// instructions in order and then its terminator, and a backward one its
/// terminator and then its instructions in reverse; between blocks, every
/// state passes `transfer_edge` of the edge it flows along.
pub fn solve_dense<A>(analysis: &A, body: &Body) -> DenseFacts<A::State>
where
    A: DenseAnalysis,
{
    let postorder = postorder(body);
    let visit_order: Vec<BlockId> = match A::DIRECTION {
        Direction::Forward => postorder.iter().rev().copied().collect(),
        Direction::Backward => postorder,
    };
    let rank: FxHashMap<BlockId, usize> =
        visit_order.iter().enumerate().map(|(rank, block)| (*block, rank)).collect();
    let loop_headers = body.loop_headers();

    let mut states: FxHashMap<BlockId, A::State> =
        body.blocks().map(|(block, _)| (block, A::State::bottom())).collect();
    let boundary: SmallVec<[BlockId; 2]> = match A::DIRECTION {
        Direction::Forward => smallvec::smallvec![body.entry()],
        Direction::Backward => visit_order
            .iter()
            .copied()
            .filter(|block| body.block(*block).terminator().edges().is_empty())
            .collect(),
    };
    for block in boundary {
        states.insert(block, analysis.entry(body));
    }
    let preds = body.predecessors();

    let mut worklist: BTreeSet<usize> = (0..visit_order.len()).collect();
    while let Some(rank_of_block) = worklist.pop_first() {
        let block = visit_order[rank_of_block];
        let mut state = states.get(&block).expect("every block has a state").clone();
        let record = body.block(block);
        let mut flows: SmallVec<[(BlockId, A::State); 2]> = SmallVec::new();
        match A::DIRECTION {
            Direction::Forward => {
                for inst in record.insts() {
                    analysis.transfer(inst, &mut state);
                }
                analysis.transfer_terminator(record.terminator(), &mut state);
                for edge in record.terminator().edges() {
                    let mut along = state.clone();
                    analysis.transfer_edge(&edge, body.block(edge.to).params(), &mut along);
                    flows.push((edge.to, along));
                }
            }
            Direction::Backward => {
                analysis.transfer_terminator(record.terminator(), &mut state);
                for inst in record.insts().rev() {
                    analysis.transfer(inst, &mut state);
                }
                for pred in preds.get(&block).expect("every block has a predecessor entry") {
                    let edges = body.block(*pred).terminator().edges();
                    for edge in edges.iter().filter(|edge| edge.to == block) {
                        let mut along = state.clone();
                        analysis.transfer_edge(edge, record.params(), &mut along);
                        flows.push((*pred, along));
                    }
                }
            }
        }
        for (next, along) in flows {
            let Some(next_rank) = rank.get(&next) else {
                continue;
            };
            let stored = states.get_mut(&next).expect("every block has a state");
            let rose = if loop_headers.contains(&next) { stored.widen(&along) } else { stored.join(&along) };
            if rose {
                worklist.insert(*next_rank);
            }
        }
    }
    DenseFacts { at_entry: states }
}

/// Each block's successors, each once. Cost: O(its edges).
fn successors(body: &Body, block: BlockId) -> SmallVec<[BlockId; 2]> {
    let mut successors: SmallVec<[BlockId; 2]> = SmallVec::new();
    for edge in body.block(block).terminator().edges() {
        if !successors.contains(&edge.to) {
            successors.push(edge.to);
        }
    }
    successors
}

/// The blocks the entry reaches, in the postorder of a depth-first walk of
/// the edges from it. Cost: O(blocks + edges).
fn postorder(body: &Body) -> Vec<BlockId> {
    let mut order = Vec::new();
    let mut visited = FxHashSet::default();
    let mut path: Vec<(BlockId, SmallVec<[BlockId; 2]>)> = vec![(body.entry(), successors(body, body.entry()))];
    visited.insert(body.entry());
    while let Some((block, pending)) = path.last_mut() {
        let Some(next) = pending.pop() else {
            order.push(*block);
            path.pop();
            continue;
        };
        if visited.insert(next) {
            path.push((next, successors(body, next)));
        }
    }
    order
}
