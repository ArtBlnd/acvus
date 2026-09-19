//! Dominator tree - Cooper, Harvey, Kennedy (2001) iterative algorithm.
//!
//! Block A **dominates** block B if every path from the entry to B passes
//! through A. The immediate dominator (idom) is the closest strict dominator.
//!
//! The entry block (index 0) dominates everything and has no dominator itself.
//!
//! # Unreachable blocks
//!
//! Blocks not reachable from the entry have `idom = UNREACHABLE`. All queries
//! (`idom`, `dominates`, `depth`) handle this explicitly - no panics.

use crate::cfg::{BlockIdx, CfgBody};
use smallvec::SmallVec;

/// Sentinel: idom not yet computed during fixpoint iteration.
const UNDEFINED: usize = usize::MAX;

/// Sentinel: block is unreachable from the entry.
/// After `build()` completes, any block still UNDEFINED is unreachable.
const UNREACHABLE: usize = usize::MAX;

/// Dominator tree computed from a CfgBody.
pub struct DomTree {
    /// `idom[b]` = immediate dominator of block `b`.
    ///
    /// - Entry block: `idom[0] == 0` (self-loop, no dominator).
    /// - Normal block: `idom[b]` = index of its immediate dominator.
    /// - Unreachable: `idom[b] == UNREACHABLE`.
    idom: Vec<usize>,
}

impl DomTree {
    /// Build the dominator tree from a CfgBody.
    pub fn build(cfg: &CfgBody) -> Self {
        let n = cfg.blocks.len();
        if n == 0 {
            return Self { idom: vec![] };
        }

        // -- Step 1: Reverse postorder ------------------------------
        //
        // RPO gives a topological-ish ordering where dominators come
        // before dominated blocks. The fixpoint converges faster.

        let rpo = reverse_postorder(cfg, n);
        let mut rpo_order = vec![UNDEFINED; n];
        for (pos, &block) in rpo.iter().enumerate() {
            rpo_order[block] = pos;
        }

        // -- Step 2: Predecessors -----------------------------------

        let preds = cfg.predecessors();

        // -- Step 3: Fixpoint iteration -----------------------------
        //
        // Starting from "entry dominates itself, everything else undefined",
        // iterate until no idom changes. Each block's idom is the intersection
        // (nearest common dominator) of all its processed predecessors.

        let mut idom = vec![UNDEFINED; n];
        idom[rpo[0]] = rpo[0]; // Entry dominates itself.

        let mut changed = true;
        while changed {
            changed = false;

            // Process in RPO, skip entry.
            for &b in &rpo[1..] {
                let pred_list: SmallVec<[usize; 2]> = preds
                    .get(&BlockIdx(b))
                    .map(|ps| ps.iter().map(|p| p.0).collect())
                    .unwrap_or_default();

                // Find first predecessor with a computed idom.
                let Some(&first_processed) = pred_list.iter().find(|&&p| idom[p] != UNDEFINED)
                else {
                    continue; // All predecessors unreachable - skip.
                };

                // Intersect with remaining processed predecessors.
                let mut new_idom = first_processed;
                for &p in &pred_list {
                    if p != first_processed && idom[p] != UNDEFINED {
                        new_idom = intersect(&idom, &rpo_order, new_idom, p);
                    }
                }

                if idom[b] != new_idom {
                    idom[b] = new_idom;
                    changed = true;
                }
            }
        }

        Self { idom }
    }

    /// Immediate dominator of `block`. None for entry or unreachable blocks.
    pub fn idom(&self, block: BlockIdx) -> Option<BlockIdx> {
        let b = block.0;
        if b >= self.idom.len() {
            return None;
        }
        let dom = self.idom[b];
        if dom == b || dom == UNREACHABLE {
            return None; // Entry (self-loop) or unreachable.
        }
        Some(BlockIdx(dom))
    }

    /// Does `a` dominate `b`? A block dominates itself.
    pub fn dominates(&self, a: BlockIdx, b: BlockIdx) -> bool {
        let mut cursor = b.0;
        loop {
            if cursor >= self.idom.len() {
                return false;
            }
            if cursor == a.0 {
                return true;
            }
            let parent = self.idom[cursor];
            if parent == UNREACHABLE {
                return false; // Hit unreachable block.
            }
            if parent == cursor {
                return false; // Reached entry without finding `a`.
            }
            cursor = parent;
        }
    }

    /// All blocks strictly dominated by `a` (excluding `a` itself).
    pub fn dominated_by(&self, a: BlockIdx) -> Vec<BlockIdx> {
        (0..self.idom.len())
            .filter(|&b| b != a.0 && self.dominates(a, BlockIdx(b)))
            .map(BlockIdx)
            .collect()
    }

    /// Depth in the dominator tree (entry = 0). Unreachable blocks return 0.
    pub fn depth(&self, block: BlockIdx) -> usize {
        let mut d = 0;
        let mut cursor = block.0;
        loop {
            if cursor >= self.idom.len() {
                break;
            }
            let parent = self.idom[cursor];
            if parent == cursor || parent == UNREACHABLE {
                break; // Entry or unreachable.
            }
            cursor = parent;
            d += 1;
        }
        d
    }
}

// -- Internal helpers -----------------------------------------------

/// Nearest common dominator of `a` and `b` (the "intersect" function from CHK01).
///
/// Walks both fingers up the dominator tree using RPO numbers until they meet.
/// Invariant: `idom[a]` and `idom[b]` are both != UNDEFINED.
fn intersect(idom: &[usize], rpo_order: &[usize], mut a: usize, mut b: usize) -> usize {
    while a != b {
        while rpo_order[a] > rpo_order[b] {
            a = idom[a];
        }
        while rpo_order[b] > rpo_order[a] {
            b = idom[b];
        }
    }
    a
}

/// Compute reverse postorder via iterative DFS from block 0.
fn reverse_postorder(cfg: &CfgBody, n: usize) -> Vec<usize> {
    let mut visited = vec![false; n];
    let mut postorder = Vec::with_capacity(n);

    // Stack entries: (block_index, next_successor_to_visit).
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    visited[0] = true;

    while let Some(&mut (block, ref mut next_succ)) = stack.last_mut() {
        let succs = cfg.successors(BlockIdx(block));
        if *next_succ < succs.len() {
            let succ = succs[*next_succ].0;
            *next_succ += 1;
            if !visited[succ] {
                visited[succ] = true;
                stack.push((succ, 0));
            }
        } else {
            postorder.push(block);
            stack.pop();
        }
    }

    postorder.reverse();
    postorder
}

// -- Post-dominator tree ---------------------------------------------
//
// Block A **post-dominates** block B if every path from B to any exit
// passes through A. Computed by running the dominator algorithm on the
// reverse CFG (edges flipped, exit block as entry).

/// Post-dominator tree.
pub struct PostDomTree {
    /// `ipdom[b]` = immediate post-dominator of block `b`.
    /// Exit block: self-loop. Unreachable: UNREACHABLE.
    ipdom: Vec<usize>,
}

impl PostDomTree {
    /// Build the post-dominator tree from a CfgBody.
    ///
    /// Identifies exit blocks (Return terminators) and computes post-dominance
    /// on the reverse CFG. If there are multiple exit blocks, a virtual exit
    /// node is added.
    pub fn build(cfg: &CfgBody) -> Self {
        let n = cfg.blocks.len();
        if n == 0 {
            return Self { ipdom: vec![] };
        }

        // Find exit blocks (blocks with Return terminator).
        let exits: Vec<usize> = (0..n)
            .filter(|&bi| {
                matches!(
                    cfg.blocks[bi].terminator,
                    crate::cfg::Terminator::Return { .. }
                )
            })
            .collect();

        if exits.is_empty() {
            // No return - all blocks unreachable in reverse.
            return Self {
                ipdom: vec![UNREACHABLE; n],
            };
        }

        // Build reverse CFG: reverse_succs[b] = blocks that b is a successor of
        // = predecessors of b in forward CFG become successors of b in reverse.
        // Actually: reverse edge (a->b) becomes (b->a).
        // reverse_succs[b] = { a | a->b in forward CFG } = preds[b] in forward.
        let fwd_preds = cfg.predecessors();
        let reverse_succs: Vec<SmallVec<[usize; 2]>> = (0..n)
            .map(|bi| {
                fwd_preds
                    .get(&BlockIdx(bi))
                    .map(|ps| ps.iter().map(|p| p.0).collect())
                    .unwrap_or_default()
            })
            .collect();

        // For single exit, that exit is the entry of reverse CFG.
        // For multiple exits, we add a virtual node that all exits point to.
        if exits.len() == 1 {
            let exit = exits[0];
            let ipdom = compute_domtree_on_reverse(n, exit, &reverse_succs);
            Self { ipdom }
        } else {
            // Virtual exit node at index `n`.
            let virtual_exit = n;
            let mut rev_succs_with_virtual: Vec<SmallVec<[usize; 2]>> =
                reverse_succs.into_iter().collect();
            // Add virtual node.
            rev_succs_with_virtual.push(SmallVec::new());
            // Each real exit -> virtual exit in forward = virtual exit -> each real exit in reverse.
            // So virtual node's successors = all real exits.
            // And each real exit's successors already include its reverse preds;
            // additionally, virtual exit has predecessor = each real exit, meaning
            // in reverse: each real exit's successor list gains virtual_exit.
            // Actually: forward edge: exit->virtual_exit. Reverse: virtual_exit->exit.
            // So rev_succs_with_virtual[virtual_exit] = exits.
            rev_succs_with_virtual[virtual_exit] = exits.iter().map(|&e| e).collect();

            let mut ipdom =
                compute_domtree_on_reverse(n + 1, virtual_exit, &rev_succs_with_virtual);
            // Remove virtual node, map any reference to it to UNREACHABLE.
            ipdom.truncate(n);
            for dom in ipdom.iter_mut() {
                if *dom == virtual_exit {
                    *dom = *dom; // exit post-dominates itself via virtual
                    // Actually, blocks whose ipdom is the virtual exit have no real post-dominator.
                    // For practical purposes, treat them as having no ipdom.
                    *dom = UNREACHABLE;
                }
            }
            // Exits that pointed to virtual should point to themselves (roots).
            for &exit in &exits {
                if ipdom[exit] == UNREACHABLE {
                    ipdom[exit] = exit; // self-loop = root
                }
            }
            Self { ipdom }
        }
    }

    /// Immediate post-dominator of `block`. None for exit or unreachable blocks.
    pub fn ipdom(&self, block: BlockIdx) -> Option<BlockIdx> {
        let b = block.0;
        if b >= self.ipdom.len() {
            return None;
        }
        let dom = self.ipdom[b];
        if dom == b || dom == UNREACHABLE {
            return None;
        }
        Some(BlockIdx(dom))
    }

    /// Does `a` post-dominate `b`?
    pub fn post_dominates(&self, a: BlockIdx, b: BlockIdx) -> bool {
        let mut cursor = b.0;
        loop {
            if cursor >= self.ipdom.len() {
                return false;
            }
            if cursor == a.0 {
                return true;
            }
            let parent = self.ipdom[cursor];
            if parent == UNREACHABLE || parent == cursor {
                return false;
            }
            cursor = parent;
        }
    }
}

/// Compute dominator tree on a graph given as successor lists.
/// `entry` is the root. Uses the same CHK01 algorithm as DomTree::build.
fn compute_domtree_on_reverse(
    n: usize,
    entry: usize,
    succs: &[SmallVec<[usize; 2]>],
) -> Vec<usize> {
    // Build predecessors from successors (in the reverse graph).
    let mut preds: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); n];
    for (src, dsts) in succs.iter().enumerate() {
        for &dst in dsts {
            if dst < n {
                preds[dst].push(src);
            }
        }
    }

    // Reverse postorder via DFS from entry.
    let mut visited = vec![false; n];
    let mut postorder = Vec::with_capacity(n);
    let mut stack: Vec<(usize, usize)> = vec![(entry, 0)];
    visited[entry] = true;

    while let Some(&mut (block, ref mut next_succ)) = stack.last_mut() {
        if block < succs.len() && *next_succ < succs[block].len() {
            let succ = succs[block][*next_succ];
            *next_succ += 1;
            if succ < n && !visited[succ] {
                visited[succ] = true;
                stack.push((succ, 0));
            }
        } else {
            postorder.push(block);
            stack.pop();
        }
    }
    postorder.reverse();
    let rpo = postorder;

    let mut rpo_order = vec![UNDEFINED; n];
    for (pos, &block) in rpo.iter().enumerate() {
        rpo_order[block] = pos;
    }

    // Fixpoint iteration.
    let mut idom = vec![UNDEFINED; n];
    idom[entry] = entry;

    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo[1..] {
            let pred_list = &preds[b];
            let Some(&first_processed) = pred_list.iter().find(|&&p| idom[p] != UNDEFINED) else {
                continue;
            };
            let mut new_idom = first_processed;
            for &p in pred_list {
                if p != first_processed && idom[p] != UNDEFINED {
                    new_idom = intersect(&idom, &rpo_order, new_idom, p);
                }
            }
            if idom[b] != new_idom {
                idom[b] = new_idom;
                changed = true;
            }
        }
    }

    idom
}

// -- Tests ----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use crate::ir::*;
    use acvus_utils::{LocalFactory, LocalIdOps};
    use rustc_hash::FxHashMap;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_cfg(insts: Vec<InstKind>) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..20 {
            factory.next();
        }
        promote(MirBody {
            demoted_diamonds: Default::default(),
            insts: insts
                .into_iter()
                .map(|kind| Inst {
                    span: acvus_ast::Span::ZERO,
                    kind,
                })
                .collect(),
            val_types: FxHashMap::default(),
            params: Vec::new(),
            captures: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
            order_param: None,
            task: crate::ty::Task::Sync,
        })
    }

    #[test]
    fn linear_chain() {
        let cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(0),
            },
            InstKind::Jump {
                label: Label(0),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Jump {
                label: Label(1),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
            },
            InstKind::Return {
                value: v(1),
                order: None,
            },
        ]);
        let dom = DomTree::build(&cfg);

        assert!(dom.dominates(BlockIdx(0), BlockIdx(0)));
        assert!(dom.dominates(BlockIdx(0), BlockIdx(1)));
        assert!(dom.dominates(BlockIdx(0), BlockIdx(2)));
        assert!(dom.dominates(BlockIdx(1), BlockIdx(2)));
        assert!(!dom.dominates(BlockIdx(1), BlockIdx(0)));

        assert_eq!(dom.idom(BlockIdx(0)), None);
        assert_eq!(dom.idom(BlockIdx(1)), Some(BlockIdx(0)));
        assert_eq!(dom.idom(BlockIdx(2)), Some(BlockIdx(1)));
    }

    #[test]
    fn diamond() {
        let cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::JumpIf {
                cond: v(0),
                then_label: Label(0),
                then_args: vec![],
                else_label: Label(1),
                else_args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(0),
            },
            InstKind::Return {
                value: v(1),
                order: None,
            },
        ]);
        let dom = DomTree::build(&cfg);

        assert!(dom.dominates(BlockIdx(0), BlockIdx(3)));
        assert!(!dom.dominates(BlockIdx(1), BlockIdx(3)));
        assert!(!dom.dominates(BlockIdx(2), BlockIdx(3)));
        assert_eq!(dom.idom(BlockIdx(3)), Some(BlockIdx(0)));
    }

    #[test]
    fn simple_loop() {
        let cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(0),
            },
            InstKind::Jump {
                label: Label(0),
                args: vec![v(0)],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![v(1)],
            },
            InstKind::Const {
                dst: v(2),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::JumpIf {
                cond: v(2),
                then_label: Label(0),
                then_args: vec![v(1)],
                else_label: Label(1),
                else_args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
            },
            InstKind::Return {
                value: v(1),
                order: None,
            },
        ]);
        let dom = DomTree::build(&cfg);

        assert!(dom.dominates(BlockIdx(1), BlockIdx(2)));
        assert_eq!(dom.idom(BlockIdx(1)), Some(BlockIdx(0)));
        assert_eq!(dom.idom(BlockIdx(2)), Some(BlockIdx(1)));
    }

    #[test]
    fn depth_linear() {
        let cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(0),
            },
            InstKind::Jump {
                label: Label(0),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Jump {
                label: Label(1),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
            },
            InstKind::Return {
                value: v(1),
                order: None,
            },
        ]);
        let dom = DomTree::build(&cfg);

        assert_eq!(dom.depth(BlockIdx(0)), 0);
        assert_eq!(dom.depth(BlockIdx(1)), 1);
        assert_eq!(dom.depth(BlockIdx(2)), 2);
    }

    #[test]
    fn single_block() {
        let cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(0),
            },
            InstKind::Return {
                value: v(0),
                order: None,
            },
        ]);
        let dom = DomTree::build(&cfg);

        assert_eq!(dom.idom(BlockIdx(0)), None);
        assert!(dom.dominates(BlockIdx(0), BlockIdx(0)));
    }

    #[test]
    fn empty_cfg() {
        let cfg = make_cfg(vec![]);
        let dom = DomTree::build(&cfg);
        // promote creates one empty block even for empty input.
        assert_eq!(dom.idom.len(), 1);
    }
}
