//! The values live at each block's start and end: the dense engine,
//! backward, over value sets. An instruction kills what it defines and makes
//! its operands live; an edge kills its target's parameters and makes its
//! arguments live, so a block's end holds, over its successors, what each
//! needs beyond its own parameters, and the arguments passed to it.
//!
//! Cost: a block's state only grows, at most once per value of the body,
//! and each growth re-runs the transfer of the block and of its incoming
//! edges, so O(values · (blocks + edges) · the largest block's operands).
//! The height is the body's values; the widening is the join.
//!
//! passes-analyses.md.

use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use crate::ir::ValueId;
use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Edge, Inst, Module, Terminator};
use crate::passes::analysis::{Analysis, AnalysisManager, DenseAnalysis, Direction, solve_dense};
use crate::passes::domain::{JoinSemiLattice, Widen};

pub struct Liveness {
    at_start: FxHashMap<BlockId, FxHashSet<ValueId>>,
    at_end: FxHashMap<BlockId, FxHashSet<ValueId>>,
}

impl Liveness {
    /// The values live at `block`'s start, its parameters defined: its
    /// parameters it reads, and what it and its successors read that is
    /// defined before it. Cost: O(1).
    ///
    /// # Panics
    /// `block` is not a block of the body.
    pub fn live_at_start(&self, block: BlockId) -> &FxHashSet<ValueId> {
        self.at_start.get(&block).expect("liveness covers every block of its body")
    }

    /// The values live after `block`'s terminator, on its way into a
    /// successor: the arguments its edges pass, and what its successors
    /// read beyond their own parameters. Cost: O(1).
    ///
    /// # Panics
    /// `block` is not a block of the body.
    pub fn live_at_end(&self, block: BlockId) -> &FxHashSet<ValueId> {
        self.at_end.get(&block).expect("liveness covers every block of its body")
    }
}

impl Analysis for Liveness {
    type Facts = Liveness;

    const READS: &'static [Aspect] = &[Aspect::CfgShape, Aspect::Instructions];

    fn compute(_: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Liveness {
        let body = module.body(body);
        let facts = solve_dense(&Backward, body);
        let mut at_start = FxHashMap::default();
        let mut at_end = FxHashMap::default();
        for (id, block) in body.blocks() {
            let end = facts.at(id).clone();
            let mut start = end.clone();
            Backward.transfer_terminator(block.terminator(), &mut start);
            for inst in block.insts().rev() {
                Backward.transfer(inst, &mut start);
            }
            at_start.insert(id, start.0);
            at_end.insert(id, end.0);
        }
        Liveness { at_start, at_end }
    }
}

/// A set of values; finite over a body, so its widening is its join.
#[derive(Debug, Clone, PartialEq)]
struct Live(FxHashSet<ValueId>);

impl JoinSemiLattice for Live {
    fn bottom() -> Self {
        Live(FxHashSet::default())
    }

    fn join(&mut self, other: &Self) -> bool {
        let before = self.0.len();
        self.0.extend(other.0.iter().copied());
        self.0.len() != before
    }

    fn le(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }
}

impl Widen for Live {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

struct Backward;

impl DenseAnalysis for Backward {
    type State = Live;

    const DIRECTION: Direction = Direction::Backward;

    fn entry(&self, _: &Body) -> Live {
        Live::bottom()
    }

    fn transfer(&self, inst: &Inst, state: &mut Live) {
        if let Some(result) = inst.result() {
            state.0.remove(&result);
        }
        if let Some(token) = inst.order().and_then(|order| order.yielded()) {
            state.0.remove(&token);
        }
        state.0.extend(inst.operands());
    }

    fn transfer_terminator(&self, terminator: &Terminator, state: &mut Live) {
        state.0.extend(terminator.operands());
    }

    fn transfer_edge(&self, edge: &Edge<'_>, target_params: &[ValueId], state: &mut Live) {
        for param in target_params {
            state.0.remove(param);
        }
        state.0.extend(edge.args.iter().copied());
    }
}
