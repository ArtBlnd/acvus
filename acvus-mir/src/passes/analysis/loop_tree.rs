//! Natural loops: a back edge is an edge whose target dominates its source,
//! and the loop of a header is the header with every block that reaches one
//! of its back edges' sources without passing the header. The loops of one
//! body nest or are disjoint; each has its blocks, the loop it nests in, and
//! its preheader where one exists. Reads `Dominators`.
//!
//! A cycle whose entry dominates none of its blocks, an irreducible one, has
//! no back edge and so is no loop here.
//!
//! Cost: O(headers · (blocks + edges)) for the loops' blocks, one backward
//! walk per header, and O(loops · log loops + Σ loop sizes) for the nesting.
//!
//! passes-analyses.md.

use crate::ir::cfg::{Aspect, BlockId, BodyId, Module};
use crate::passes::analysis::dominators::Graph;
use crate::passes::analysis::{Analysis, AnalysisManager, Dominators};

/// A loop of a `LoopTree`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoopId(usize);

pub struct LoopTree {
    loops: Vec<NaturalLoop>,
    /// Each block's innermost loop, by block index.
    innermost: Vec<Option<LoopId>>,
    graph: Graph,
}

struct NaturalLoop {
    header: BlockId,
    /// The loop's blocks, the header first.
    blocks: Vec<BlockId>,
    /// Membership by block index.
    contains: Vec<bool>,
    parent: Option<LoopId>,
    preheader: Option<BlockId>,
}

impl LoopTree {
    /// Every loop, in no stated order. Cost: O(loops).
    pub fn loops(&self) -> impl Iterator<Item = LoopId> + '_ {
        (0..self.loops.len()).map(LoopId)
    }

    /// Cost: O(1).
    pub fn header(&self, of: LoopId) -> BlockId {
        self.loops[of.0].header
    }

    /// The loop's blocks, its header first. Cost: O(1).
    pub fn blocks(&self, of: LoopId) -> &[BlockId] {
        &self.loops[of.0].blocks
    }

    /// Cost: O(1).
    pub fn contains(&self, of: LoopId, block: BlockId) -> bool {
        self.loops[of.0].contains[block.index()]
    }

    /// The innermost loop `of` nests in. Cost: O(1).
    pub fn parent(&self, of: LoopId) -> Option<LoopId> {
        self.loops[of.0].parent
    }

    /// The one block outside the loop with an edge into the header, where
    /// that block has no edge elsewhere. Cost: O(1).
    pub fn preheader(&self, of: LoopId) -> Option<BlockId> {
        self.loops[of.0].preheader
    }

    /// The innermost loop that holds `block`. Cost: O(1).
    pub fn innermost(&self, block: BlockId) -> Option<LoopId> {
        self.innermost[block.index()]
    }

    /// Whether some path from `from` to `to`, both included, passes a block
    /// of a loop that does not hold both: a loop the run may enter between
    /// the two and never leave. Cost: O(blocks + edges).
    pub fn loop_between(&self, from: BlockId, to: BlockId) -> bool {
        let after = reach(from.index(), &self.graph.successors);
        let before = reach(to.index(), &self.graph.predecessors);
        let holds_both = |of: LoopId| self.contains(of, from) && self.contains(of, to);
        (0..self.graph.blocks.len())
            .filter(|block| after[*block] && before[*block])
            .filter_map(|block| self.innermost[block])
            .any(|of| !holds_both(of))
    }
}

/// The nodes `start` reaches along `edges`, itself included. Cost: O(nodes
/// + edges).
fn reach(start: usize, edges: &[smallvec::SmallVec<[usize; 2]>]) -> Vec<bool> {
    let mut reached = vec![false; edges.len()];
    reached[start] = true;
    let mut pending = vec![start];
    while let Some(node) = pending.pop() {
        for next in edges[node].iter().copied() {
            if !reached[next] {
                reached[next] = true;
                pending.push(next);
            }
        }
    }
    reached
}

impl Analysis for LoopTree {
    type Facts = LoopTree;

    const READS: &'static [Aspect] = &[Aspect::CfgShape];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> LoopTree {
        let dominators = am.get::<Dominators>(module, body);
        let graph = Graph::of(module.body(body));
        let count = graph.blocks.len();

        let mut loops: Vec<NaturalLoop> = Vec::new();
        for header in graph.blocks.iter().copied().filter(|block| dominators.reachable(*block)) {
            let latches: Vec<usize> = graph.predecessors[header.index()]
                .iter()
                .copied()
                .filter(|pred| dominators.reachable(graph.blocks[*pred]))
                .filter(|pred| dominators.dominates(header, graph.blocks[*pred]))
                .collect();
            if latches.is_empty() {
                continue;
            }
            let contains = loop_blocks(&graph, &dominators, header.index(), latches);
            let blocks: Vec<BlockId> = std::iter::once(header)
                .chain(graph.blocks.iter().copied().filter(|block| *block != header && contains[block.index()]))
                .collect();
            let preheader = preheader(&graph, header.index(), &contains);
            loops.push(NaturalLoop { header, blocks, contains, parent: None, preheader });
        }

        let mut by_size: Vec<usize> = (0..loops.len()).collect();
        by_size.sort_by_key(|at| std::cmp::Reverse(loops[*at].blocks.len()));
        let mut innermost: Vec<Option<LoopId>> = vec![None; count];
        for at in by_size {
            loops[at].parent = innermost[loops[at].header.index()];
            for block in &loops[at].blocks {
                innermost[block.index()] = Some(LoopId(at));
            }
        }
        LoopTree { loops, innermost, graph }
    }
}

/// The blocks of the loop at `header`: the header, and every block the entry
/// reaches that reaches a latch without passing the header. Cost: O(blocks +
/// edges).
fn loop_blocks(graph: &Graph, dominators: &Dominators, header: usize, latches: Vec<usize>) -> Vec<bool> {
    let mut contains = vec![false; graph.blocks.len()];
    contains[header] = true;
    let mut pending = latches;
    while let Some(block) = pending.pop() {
        if contains[block] {
            continue;
        }
        contains[block] = true;
        let reached_preds = graph.predecessors[block]
            .iter()
            .copied()
            .filter(|pred| dominators.reachable(graph.blocks[*pred]));
        pending.extend(reached_preds);
    }
    contains
}

/// The header's one predecessor outside the loop, where it has no other
/// successor. Cost: O(the header's predecessors).
fn preheader(graph: &Graph, header: usize, contains: &[bool]) -> Option<BlockId> {
    let mut outside = graph.predecessors[header].iter().copied().filter(|pred| !contains[*pred]);
    let (Some(only), None) = (outside.next(), outside.next()) else {
        return None;
    };
    match graph.successors[only].as_slice() {
        [to] if *to == header => Some(graph.blocks[only]),
        _ => None,
    }
}
