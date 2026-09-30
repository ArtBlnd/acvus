//! Post-dominators: Cooper-Harvey-Kennedy on the reverse CFG, rooted at one
//! virtual exit that every `Return` and `Diverge` block reaches. Cost as
//! `Dominators`: O(blocks + edges) per iteration, at most d + 3 iterations,
//! and O(1) per query.
//!
//! A block from which no path reaches an exit, one inside a loop that never
//! ends, is post-dominated by itself alone. Taken vacuously, every block
//! would post-dominate it, and `control_equivalent` would pair it with
//! blocks its runs need not reach.
//!
//! passes-analyses.md.

use smallvec::SmallVec;

use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Module, Terminator};
use crate::passes::analysis::dominators::{DomTree, Graph};
use crate::passes::analysis::{Analysis, AnalysisManager, Point};

pub struct PostDominators {
    /// Over the blocks, by index, and the virtual exit after them.
    tree: DomTree,
    blocks: Vec<BlockId>,
}

impl PostDominators {
    /// The immediate post-dominator of `block`; none where it is the virtual
    /// exit, as for a `Return` or `Diverge` block, or where `block` reaches
    /// no exit (`reaches_exit` tells which). Cost: O(1).
    pub fn ipdom(&self, block: BlockId) -> Option<BlockId> {
        let node = self.tree.idom(block.index())?;
        match node == self.blocks.len() {
            true => None,
            false => Some(self.blocks[node]),
        }
    }

    /// Whether some path from `block` reaches a `Return` or a `Diverge`.
    /// Cost: O(1).
    pub fn reaches_exit(&self, block: BlockId) -> bool {
        self.tree.reached(block.index())
    }

    /// Whether every path from `before` to an exit passes `after`; a block
    /// post-dominates itself. Cost: O(1).
    pub fn post_dominates(&self, after: BlockId, before: BlockId) -> bool {
        if after == before {
            return true;
        }
        self.reaches_exit(after)
            && self.reaches_exit(before)
            && self.tree.dominates(after.index(), before.index())
    }

    /// Whether every path from `before` to an exit passes `after`, which
    /// within one block means `after` is no earlier. Cost: O(1).
    pub fn point_post_dominates(&self, body: &Body, after: Point, before: Point) -> bool {
        let (after_block, before_block) = (after.block(body), before.block(body));
        if after_block == before_block {
            return before.no_later_than(body, after);
        }
        self.post_dominates(after_block, before_block)
    }
}

impl Analysis for PostDominators {
    type Facts = PostDominators;

    const READS: &'static [Aspect] = &[Aspect::CfgShape];

    fn compute(_: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> PostDominators {
        let body = module.body(body);
        let Graph { blocks, successors, predecessors } = Graph::of(body);
        let exit = blocks.len();
        let exits: SmallVec<[usize; 2]> = body
            .blocks()
            .filter(|(_, block)| match block.terminator() {
                Terminator::Return { .. } | Terminator::Diverge | Terminator::NoArmMatched { .. } => true,
                Terminator::Jump { .. }
                | Terminator::JumpIf { .. }
                | Terminator::Diamond { .. }
                | Terminator::Switch { .. }
                | Terminator::For { .. }
                | Terminator::While { .. } => false,
            })
            .map(|(id, _)| id.index())
            .collect();

        let mut reverse_successors = predecessors;
        reverse_successors.push(exits.clone());
        let mut reverse_predecessors = successors;
        reverse_predecessors.push(SmallVec::new());
        for block in exits {
            reverse_predecessors[block].push(exit);
        }
        let tree = DomTree::of(exit, &reverse_successors, &reverse_predecessors);
        PostDominators { tree, blocks }
    }
}
