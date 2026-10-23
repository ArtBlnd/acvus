//! Natural loops: a back edge is an edge whose target dominates its source,
//! and the loop of a header is the header with every block that reaches one
//! of its back edges' sources without passing the header. The loops of one
//! body nest or are disjoint; each has its blocks, the loop it nests in, and
//! its preheader where one exists. Reads `Dominators`.
//!
//! A cycle whose entry dominates none of its blocks, an irreducible one, has
//! no back edge and so is no loop here.
//!
//! The loops are found innermost first, headers taken deepest in the
//! dominator tree first, each by a backward walk from its latches in which a
//! loop found already stands as its header (Havlak, "Nesting of Reducible
//! and Irreducible Loops", 1997, over natural loops): a block of an inner
//! loop other than its header has all its predecessors inside that loop, so
//! the walk only needs the header's. A union-find keeps, for each block, the
//! header of the outermost loop found so far that holds it, so every block's
//! predecessors are walked once over the whole tree.
//!
//! Cost: O((blocks + edges) · α) for the loops and their nesting, O(blocks)
//! for the tree's depth-first intervals, which answer `contains` in O(1),
//! and O(Σ loop sizes) for the blocks each loop lists.
//!
//! passes-analyses.md.

use crate::ir::cfg::{Aspect, BlockId, BodyId, Module};
use crate::passes::analysis::dominators::Graph;
use crate::passes::analysis::{Analysis, Dominators, Queries};

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
    /// The loop's blocks, the header first and the rest in block order.
    blocks: Vec<BlockId>,
    parent: Option<LoopId>,
    preheader: Option<BlockId>,
    /// The loop's place in a depth-first walk of the loop tree: it holds
    /// exactly the loops whose interval its own holds.
    interval: TreeInterval,
}

#[derive(Debug, Clone, Copy, Default)]
struct TreeInterval {
    pre: usize,
    post: usize,
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

    /// Whether `block` is a block of `of`: its innermost loop is `of` or
    /// nests in it. Cost: O(1).
    pub fn contains(&self, of: LoopId, block: BlockId) -> bool {
        self.innermost[block.index()].is_some_and(|inner| self.nests_in(inner, of))
    }

    /// Whether `inner` is `outer` or nests in it.
    fn nests_in(&self, inner: LoopId, outer: LoopId) -> bool {
        let (inner, outer) = (self.loops[inner.0].interval, self.loops[outer.0].interval);
        outer.pre <= inner.pre && inner.post <= outer.post
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

    /// The preheader of the loop `header` heads, as `preheader` states it.
    /// A block no back edge enters heads no natural loop, so each of its
    /// predecessors is outside the loop it heads. Cost: O(the header's
    /// predecessors).
    pub fn preheader_of(&self, header: BlockId) -> Option<BlockId> {
        match self.innermost(header).filter(|of| self.header(*of) == header) {
            Some(of) => self.preheader(of),
            None => entering_block(&self.graph, header.index(), |_| false),
        }
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

    fn compute(queries: &mut Queries<'_, '_>, module: &Module, body: BodyId) -> LoopTree {
        let dominators = queries.get::<Dominators>(module, body);
        let graph = Graph::of(module.body(body));
        let count = graph.blocks.len();
        let reached = |node: usize| dominators.reachable(graph.blocks[node]);

        let mut loops: Vec<NaturalLoop> = Vec::new();
        let mut latches: Vec<Vec<usize>> = Vec::new();
        for header in graph.blocks.iter().copied().filter(|block| dominators.reachable(*block)) {
            let into: Vec<usize> = graph.predecessors[header.index()]
                .iter()
                .copied()
                .filter(|pred| reached(*pred) && dominators.dominates(header, graph.blocks[*pred]))
                .collect();
            if into.is_empty() {
                continue;
            }
            loops.push(NaturalLoop {
                header,
                blocks: vec![header],
                parent: None,
                preheader: None,
                interval: TreeInterval::default(),
            });
            latches.push(into);
        }

        let depth = dominator_depths(&graph, &dominators);
        let mut deepest_first: Vec<usize> = (0..loops.len()).collect();
        deepest_first.sort_by_key(|at| std::cmp::Reverse(depth[loops[*at].header.index()]));

        let mut innermost: Vec<Option<LoopId>> = vec![None; count];
        let mut outermost_header = UnionFind::new(count);
        let mut walked_by: Vec<Option<usize>> = vec![None; count];
        for at in deepest_first {
            let header = loops[at].header.index();
            innermost[header] = Some(LoopId(at));
            let mut pending: Vec<usize> = latches[at].iter().map(|latch| outermost_header.find(*latch)).collect();
            while let Some(block) = pending.pop() {
                if block == header || walked_by[block] == Some(at) {
                    continue;
                }
                walked_by[block] = Some(at);
                match innermost[block] {
                    None => innermost[block] = Some(LoopId(at)),
                    Some(inner) => loops[inner.0].parent = Some(LoopId(at)),
                }
                outermost_header.join_into(block, header);
                for pred in graph.predecessors[block].iter().copied().filter(|pred| reached(*pred)) {
                    let stands_for = outermost_header.find(pred);
                    if stands_for != header && walked_by[stands_for] != Some(at) {
                        pending.push(stands_for);
                    }
                }
            }
        }

        number_intervals(&mut loops);
        for block in 0..count {
            let mut enclosing = innermost[block];
            while let Some(of) = enclosing {
                if loops[of.0].header.index() != block {
                    loops[of.0].blocks.push(graph.blocks[block]);
                }
                enclosing = loops[of.0].parent;
            }
        }
        let mut tree = LoopTree { loops, innermost, graph };
        for at in 0..tree.loops.len() {
            tree.loops[at].preheader = preheader(&tree, LoopId(at));
        }
        tree
    }
}

/// Each block's depth in the dominator tree, the entry's being zero; a block
/// the entry does not reach has none that matters and is given zero. Cost:
/// O(blocks).
fn dominator_depths(graph: &Graph, dominators: &Dominators) -> Vec<usize> {
    let mut depth: Vec<Option<usize>> = vec![None; graph.blocks.len()];
    for block in 0..graph.blocks.len() {
        let mut unknown: Vec<usize> = Vec::new();
        let mut at = block;
        let known = loop {
            if let Some(known) = depth[at] {
                break known;
            }
            match dominators.idom(graph.blocks[at]) {
                Some(parent) => {
                    unknown.push(at);
                    at = parent.index();
                }
                None => {
                    depth[at] = Some(0);
                    break 0;
                }
            }
        };
        for (above, node) in unknown.into_iter().rev().enumerate() {
            depth[node] = Some(known + above + 1);
        }
    }
    depth.into_iter().map(|depth| depth.expect("every block's depth is filled")).collect()
}

/// Numbers each loop's interval by a depth-first walk of the loop tree.
/// Cost: O(loops).
fn number_intervals(loops: &mut [NaturalLoop]) {
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); loops.len()];
    let mut roots: Vec<usize> = Vec::new();
    for (at, natural) in loops.iter().enumerate() {
        match natural.parent {
            Some(parent) => children[parent.0].push(at),
            None => roots.push(at),
        }
    }
    let mut next = 0;
    for root in roots {
        let mut path: Vec<(usize, usize)> = vec![(root, 0)];
        loops[root].interval.pre = next;
        next += 1;
        while let Some((node, child)) = path.last_mut() {
            let Some(visit) = children[*node].get(*child).copied() else {
                loops[*node].interval.post = next;
                next += 1;
                path.pop();
                continue;
            };
            *child += 1;
            loops[visit].interval.pre = next;
            next += 1;
            path.push((visit, 0));
        }
    }
}

/// Blocks joined into sets, each set named by the header it was last joined
/// into (Tarjan's union-find, with path compression).
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(count: usize) -> UnionFind {
        UnionFind { parent: (0..count).collect() }
    }

    fn find(&mut self, node: usize) -> usize {
        let mut root = node;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut at = node;
        while self.parent[at] != root {
            let next = self.parent[at];
            self.parent[at] = root;
            at = next;
        }
        root
    }

    /// Joins the set named `node` into the set named `header`.
    fn join_into(&mut self, node: usize, header: usize) {
        self.parent[node] = header;
    }
}

/// The header's one predecessor outside the loop, where it has no other
/// successor. Cost: O(the header's predecessors).
fn preheader(tree: &LoopTree, of: LoopId) -> Option<BlockId> {
    let graph = &tree.graph;
    entering_block(graph, tree.loops[of.0].header.index(), |pred| tree.contains(of, graph.blocks[pred]))
}

/// The one predecessor of `header` that `inside` does not hold, where it
/// has no successor but `header`. Cost: O(the header's predecessors).
fn entering_block(graph: &Graph, header: usize, inside: impl Fn(usize) -> bool) -> Option<BlockId> {
    let mut outside = graph.predecessors[header].iter().copied().filter(|pred| !inside(*pred));
    let (Some(only), None) = (outside.next(), outside.next()) else {
        return None;
    };
    match graph.successors[only].as_slice() {
        [to] if *to == header => Some(graph.blocks[only]),
        _ => None,
    }
}
