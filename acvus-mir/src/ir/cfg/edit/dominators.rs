//! Dominators of a body's blocks (Cooper, Harvey and Kennedy, "A Simple,
//! Fast Dominance Algorithm", 2001), and the definition points they order.
//! A block no edge from the entry reaches is dominated by every block: no
//! run reaches it. A query is O(1): each block's interval in a depth-first
//! walk of the tree holds exactly the intervals of the blocks it dominates.

use rustc_hash::FxHashMap;

use crate::ir::ValueId;
use crate::ir::cfg::body::InstId;
use crate::ir::cfg::{BlockId, Body, Site};

pub(super) struct Dominators {
    /// Each reachable block's immediate dominator, by block index; the
    /// entry is its own.
    idom: Vec<Option<BlockId>>,
    /// Each reachable block's position in reverse postorder.
    order: Vec<Option<usize>>,
    /// Each reachable block's preorder and postorder number in a
    /// depth-first walk of the dominator tree.
    interval: Vec<Option<(usize, usize)>>,
}

/// Where a value is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DefPoint {
    /// A parameter, a capture, the order parameter or a slot: defined
    /// before the entry runs.
    Entry,
    /// A block parameter.
    BlockStart(BlockId),
    /// An instruction's result or yielded token, and the block it stands in.
    Inst(BlockId, InstId),
}

impl Dominators {
    /// Cost: O(blocks + edges) per round, and rounds bounded by the loop
    /// nesting depth plus two.
    pub(super) fn of(body: &Body) -> Dominators {
        let count = body.blocks().count();
        let postorder = postorder(body, count);
        let mut order = vec![None; count];
        for (position, block) in postorder.iter().rev().enumerate() {
            order[block.index()] = Some(position);
        }
        let preds = body.predecessors();
        let mut idom: Vec<Option<BlockId>> = vec![None; count];
        idom[body.entry().index()] = Some(body.entry());
        let mut changed = true;
        while changed {
            changed = false;
            for block in postorder.iter().rev().filter(|block| **block != body.entry()) {
                let mut processed = preds[block].iter().copied().filter(|pred| idom[pred.index()].is_some());
                let Some(first) = processed.next() else {
                    continue;
                };
                let new_idom = processed.fold(first, |finger, pred| intersect(&idom, &order, finger, pred));
                if idom[block.index()] != Some(new_idom) {
                    idom[block.index()] = Some(new_idom);
                    changed = true;
                }
            }
        }
        let interval = tree_intervals(body.entry(), &idom);
        Dominators { idom, order, interval }
    }

    pub(super) fn reachable(&self, block: BlockId) -> bool {
        self.order[block.index()].is_some()
    }

    /// Whether every path from the entry to `below` passes `above`.
    pub(super) fn dominates_block(&self, above: BlockId, below: BlockId) -> bool {
        if !self.reachable(below) {
            return true;
        }
        let (Some(above), Some(below)) = (self.interval[above.index()], self.interval[below.index()]) else {
            return false;
        };
        above.0 <= below.0 && below.1 <= above.1
    }

    /// Whether a value defined at `def` is defined wherever `site` reads it.
    pub(super) fn dominates_site(&self, body: &Body, def: DefPoint, site: Site) -> bool {
        let (block, used_at) = match site {
            Site::Inst(inst) => (body.block_of(inst), Some(inst)),
            Site::Term(block) => (block, None),
        };
        match def {
            DefPoint::Entry => true,
            DefPoint::BlockStart(defined) => self.dominates_block(defined, block),
            DefPoint::Inst(defined, def_inst) if defined == block => {
                !self.reachable(block) || used_at.is_none_or(|used_at| body.comes_before(def_inst, used_at))
            }
            DefPoint::Inst(defined, _) => self.dominates_block(defined, block),
        }
    }
}

/// Where every value of the body is defined, read from the body's own
/// parameters, blocks and instructions. Cost: O(values).
pub(super) fn def_points(body: &Body) -> FxHashMap<ValueId, DefPoint> {
    let mut points = FxHashMap::default();
    let interface = body.params().iter().chain(body.captures()).map(|(_, value)| *value);
    for value in interface.chain(body.order_param()).chain(body.slots().iter().copied()) {
        points.insert(value, DefPoint::Entry);
    }
    for (id, block) in body.blocks() {
        for param in block.params() {
            points.insert(*param, DefPoint::BlockStart(id));
        }
        for inst in block.insts() {
            for value in inst.defs() {
                points.insert(value, DefPoint::Inst(id, inst.id()));
            }
        }
    }
    points
}

fn postorder(body: &Body, count: usize) -> Vec<BlockId> {
    let mut visited = vec![false; count];
    let mut postorder = Vec::with_capacity(count);
    let mut path: Vec<(BlockId, usize)> = vec![(body.entry(), 0)];
    visited[body.entry().index()] = true;
    while let Some((block, next)) = path.last_mut() {
        let edges = body.block(*block).terminator().edges();
        let Some(edge) = edges.get(*next) else {
            postorder.push(*block);
            path.pop();
            continue;
        };
        *next += 1;
        if !visited[edge.to.index()] {
            visited[edge.to.index()] = true;
            path.push((edge.to, 0));
        }
    }
    postorder
}

fn intersect(idom: &[Option<BlockId>], order: &[Option<usize>], mut a: BlockId, mut b: BlockId) -> BlockId {
    let position = |block: BlockId| order[block.index()].expect("a processed block is reachable");
    let up = |block: BlockId| idom[block.index()].expect("a processed block has an immediate dominator");
    while a != b {
        while position(a) > position(b) {
            a = up(a);
        }
        while position(b) > position(a) {
            b = up(b);
        }
    }
    a
}

/// Each reachable block's preorder and postorder number in a depth-first
/// walk of the tree `idom` makes, the entry its root. Cost: O(blocks).
fn tree_intervals(entry: BlockId, idom: &[Option<BlockId>]) -> Vec<Option<(usize, usize)>> {
    let mut children: Vec<Vec<BlockId>> = vec![Vec::new(); idom.len()];
    for (block, parent) in idom.iter().enumerate() {
        if let Some(parent) = parent.filter(|parent| parent.index() != block) {
            children[parent.index()].push(BlockId::of_index(block));
        }
    }
    let mut interval = vec![None; idom.len()];
    let mut pre = vec![0; idom.len()];
    let (mut next_pre, mut next_post) = (1, 0);
    let mut path: Vec<(BlockId, usize)> = vec![(entry, 0)];
    while let Some((block, next)) = path.last_mut() {
        let Some(child) = children[block.index()].get(*next).copied() else {
            interval[block.index()] = Some((pre[block.index()], next_post));
            next_post += 1;
            path.pop();
            continue;
        };
        *next += 1;
        pre[child.index()] = next_pre;
        next_pre += 1;
        path.push((child, 0));
    }
    interval
}
