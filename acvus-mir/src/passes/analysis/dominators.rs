//! Immediate dominators by Cooper, Harvey and Kennedy ("A Simple, Fast
//! Dominance Algorithm", 2001), and dominance between points of a body, a
//! block's points ordered by index.
//!
//! Cost: O(blocks + edges) per iteration, and at most d + 3 iterations, d
//! being the loop connectedness of the CFG, which is its loop nesting depth
//! when it is reducible. A query is O(1): the tree's depth-first intervals
//! answer it.
//!
//! passes-analyses.md.

use smallvec::SmallVec;

use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Module};
use crate::passes::analysis::{Analysis, AnalysisManager};

/// A place in a block: before one of its instructions, or at its end, after
/// every instruction and before the terminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Point {
    Before(BlockId, usize),
    End(BlockId),
}

impl Point {
    pub fn block(self) -> BlockId {
        match self {
            Point::Before(block, _) | Point::End(block) => block,
        }
    }

    /// Whether `self` comes no later than `other` in their one block.
    ///
    /// # Panics
    /// The two are in different blocks.
    pub(super) fn no_later_than(self, other: Point) -> bool {
        assert_eq!(self.block(), other.block(), "points are ordered within one block");
        match (self, other) {
            (Point::Before(_, a), Point::Before(_, b)) => a <= b,
            (Point::Before(..) | Point::End(_), Point::End(_)) => true,
            (Point::End(_), Point::Before(..)) => false,
        }
    }
}

pub struct Dominators {
    tree: DomTree,
    blocks: Vec<BlockId>,
}

impl Dominators {
    /// The immediate dominator of `block`; none for the entry and for a
    /// block the entry does not reach. Cost: O(1).
    pub fn idom(&self, block: BlockId) -> Option<BlockId> {
        self.tree.idom(block.index()).map(|node| self.blocks[node])
    }

    /// Whether some path from the entry reaches `block`. Cost: O(1).
    pub fn reachable(&self, block: BlockId) -> bool {
        self.tree.reached(block.index())
    }

    /// Whether every path from the entry to `below` passes `above`; a block
    /// is its own dominator. A block the entry does not reach is dominated
    /// by every block, since no path reaches it. Cost: O(1).
    pub fn dominates(&self, above: BlockId, below: BlockId) -> bool {
        if !self.reachable(below) {
            return true;
        }
        self.reachable(above) && self.tree.dominates(above.index(), below.index())
    }

    /// Whether every path from the entry to `below` passes `above`, which
    /// within one block means `above` is no later. Cost: O(1).
    pub fn point_dominates(&self, above: Point, below: Point) -> bool {
        if !self.reachable(below.block()) {
            return true;
        }
        if above.block() == below.block() {
            return above.no_later_than(below);
        }
        self.dominates(above.block(), below.block())
    }
}

impl Analysis for Dominators {
    type Facts = Dominators;

    const READS: &'static [Aspect] = &[Aspect::CfgShape];

    fn compute(_: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Dominators {
        let body = module.body(body);
        let graph = Graph::of(body);
        let tree = DomTree::of(body.entry().index(), &graph.successors, &graph.predecessors);
        Dominators { tree, blocks: graph.blocks }
    }
}

/// A body's blocks as the nodes `0..n` of a graph, by `BlockId::index`, with
/// each node's distinct successors and predecessors.
pub(super) struct Graph {
    pub blocks: Vec<BlockId>,
    pub successors: Vec<SmallVec<[usize; 2]>>,
    pub predecessors: Vec<SmallVec<[usize; 2]>>,
}

impl Graph {
    /// Cost: O(blocks + edges).
    pub(super) fn of(body: &Body) -> Graph {
        let blocks: Vec<BlockId> = body.blocks().map(|(id, _)| id).collect();
        let mut successors: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); blocks.len()];
        let mut predecessors: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); blocks.len()];
        for (from, block) in body.blocks() {
            for edge in block.terminator().edges() {
                let (from, to) = (from.index(), edge.to.index());
                if !successors[from].contains(&to) {
                    successors[from].push(to);
                    predecessors[to].push(from);
                }
            }
        }
        Graph { blocks, successors, predecessors }
    }
}

/// The dominator tree of the nodes a root reaches, with each node's
/// interval in a depth-first walk of the tree: `a` dominates `b` exactly
/// when `a`'s interval holds `b`'s.
pub(super) struct DomTree {
    /// Each reached node's immediate dominator; none for the root.
    idom: Vec<Option<usize>>,
    interval: Vec<Option<Interval>>,
}

/// A node's preorder and postorder number in a depth-first walk of the tree.
#[derive(Debug, Clone, Copy)]
struct Interval {
    pre: usize,
    post: usize,
}

/// A depth-first walk's place at one node: the next of its successors, or
/// children, to visit.
struct Visit {
    node: usize,
    next: usize,
}

impl DomTree {
    /// Cooper, Harvey and Kennedy's iteration in reverse postorder, then
    /// the tree's intervals. Cost: O(nodes + edges) per iteration, at most
    /// d + 3 iterations for the graph's loop connectedness d.
    pub(super) fn of(root: usize, successors: &[SmallVec<[usize; 2]>], predecessors: &[SmallVec<[usize; 2]>]) -> DomTree {
        let nodes = successors.len();
        let postorder = postorder(root, successors);
        let mut rank: Vec<Option<usize>> = vec![None; nodes];
        for (position, node) in postorder.iter().enumerate() {
            rank[*node] = Some(position);
        }

        let mut idom: Vec<Option<usize>> = vec![None; nodes];
        idom[root] = Some(root);
        let mut changed = true;
        while changed {
            changed = false;
            for node in postorder.iter().rev().copied().filter(|node| *node != root) {
                let mut processed = predecessors[node].iter().copied().filter(|pred| idom[*pred].is_some());
                let Some(first) = processed.next() else {
                    continue;
                };
                let new_idom = processed.fold(first, |finger, pred| intersect(&idom, &rank, finger, pred));
                if idom[node] != Some(new_idom) {
                    idom[node] = Some(new_idom);
                    changed = true;
                }
            }
        }
        idom[root] = None;

        let mut children: Vec<Vec<usize>> = vec![Vec::new(); nodes];
        for (node, parent) in idom.iter().enumerate() {
            if let Some(parent) = parent {
                children[*parent].push(node);
            }
        }
        let interval = intervals(root, &children);
        DomTree { idom, interval }
    }

    pub(super) fn reached(&self, node: usize) -> bool {
        self.interval[node].is_some()
    }

    pub(super) fn idom(&self, node: usize) -> Option<usize> {
        self.idom[node]
    }

    /// # Panics
    /// Either node is not reached from the root.
    pub(super) fn dominates(&self, above: usize, below: usize) -> bool {
        let interval = |node: usize| self.interval[node].expect("dominance is asked of reached nodes");
        let (above, below) = (interval(above), interval(below));
        above.pre <= below.pre && below.post <= above.post
    }
}

/// The nodes `root` reaches, in the postorder of a depth-first walk.
/// Cost: O(nodes + edges).
fn postorder(root: usize, successors: &[SmallVec<[usize; 2]>]) -> Vec<usize> {
    let mut visited = vec![false; successors.len()];
    let mut order = Vec::new();
    let mut path = vec![Visit { node: root, next: 0 }];
    visited[root] = true;
    while let Some(visit) = path.last_mut() {
        let Some(to) = successors[visit.node].get(visit.next).copied() else {
            order.push(visit.node);
            path.pop();
            continue;
        };
        visit.next += 1;
        if !visited[to] {
            visited[to] = true;
            path.push(Visit { node: to, next: 0 });
        }
    }
    order
}

fn intersect(idom: &[Option<usize>], rank: &[Option<usize>], mut a: usize, mut b: usize) -> usize {
    let rank_of = |node: usize| rank[node].expect("a processed node is reached");
    let up = |node: usize| idom[node].expect("a processed node has an immediate dominator");
    while a != b {
        while rank_of(a) < rank_of(b) {
            a = up(a);
        }
        while rank_of(b) < rank_of(a) {
            b = up(b);
        }
    }
    a
}

/// Each tree node's interval. Cost: O(nodes).
fn intervals(root: usize, children: &[Vec<usize>]) -> Vec<Option<Interval>> {
    let mut interval: Vec<Option<Interval>> = vec![None; children.len()];
    let mut pre = vec![0; children.len()];
    let mut next_pre = 1;
    let mut next_post = 0;
    let mut path = vec![Visit { node: root, next: 0 }];
    while let Some(visit) = path.last_mut() {
        let Some(child) = children[visit.node].get(visit.next).copied() else {
            interval[visit.node] = Some(Interval { pre: pre[visit.node], post: next_post });
            next_post += 1;
            path.pop();
            continue;
        };
        visit.next += 1;
        pre[child] = next_pre;
        next_pre += 1;
        path.push(Visit { node: child, next: 0 });
    }
    interval
}
