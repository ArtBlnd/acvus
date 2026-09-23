//! A block that only jumps is its target.
//!
//! A block with no parameters, whose instructions are all `Nop`, and whose
//! terminator is an unconditional `Jump` computes nothing and decides
//! nothing: every path that enters it leaves by the one edge it holds. The
//! pass hands that edge to each of its predecessors -- with the arguments
//! the block carried, since a block with no parameters is handed none --
//! removes the block, and repeats to a fixed point.
//!
//! # Why a block with parameters is not one
//!
//! Its parameters are phis: which value it holds is decided by which edge
//! arrived, and a predecessor redirected past it would have to carry that
//! decision itself. The blocks this pass was written for carry no such
//! decision. `optimize::sroa` threads a known tag straight to its arm,
//! `code_motion` then sinks the arm's body up into the branch that selected
//! it, and what is left of the arm is `jump merge(value)` over one `Nop`:
//! the parameterless case.
//!
//! # Why a loop's own blocks are not forwarders
//!
//! A loop is a header, the block that enters it, and the blocks it leaves
//! for, and both readers downstream match those three against the layout:
//! `lsr::Frame::of` wants the header's one entering block to end in a
//! `Jump` and writes the reduction's start and step into it, and
//! `prepare::recognize_loop` wants that block directly above the header and
//! the exit's label directly below the back edge. Each of the three is a
//! forwarder by shape whenever nothing was left in it, so they are read
//! from `analysis::loops` -- the definition `code_motion` and `lsr` read
//! too -- and held: they are the shape, not residue.
//!
//! # Why it runs where it does
//!
//! `code_motion` is the last pass that adds a block and the one whose sink
//! empties the arms, and `reorder` schedules within a block: this pass
//! changes which blocks there are, never what is in one.
//!
//! # Why the arguments still reach their new use
//!
//! A `Jump` argument is a value whose definition dominates the block that
//! jumps. The edge into a forwarder is the last thing its predecessor does,
//! so every path reaching that predecessor reaches the forwarder through
//! that one edge; a definition dominating the forwarder therefore dominates
//! the predecessor, and no definition is in the forwarder itself, which
//! holds only `Nop`. `graph::optimize::debug_validate` is the check.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::domtree::DomTree;
use crate::analysis::loops;
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{InstKind, Label, ValueId};

/// Where a block sends control, and what it sends with it.
#[derive(Clone)]
struct Edge {
    to: Label,
    args: Vec<ValueId>,
}

/// One outgoing edge of a terminator, in place.
struct EdgeMut<'a> {
    to: &'a mut Label,
    args: &'a mut Vec<ValueId>,
}

pub fn run(cfg: &mut CfgBody) {
    while collapse(cfg) {}
}

fn collapse(cfg: &mut CfgBody) -> bool {
    let shape = LoopShape::of(cfg);
    let forwarders: FxHashMap<Label, Edge> = (0..cfg.blocks.len())
        .filter_map(|bi| Some((cfg.blocks[bi].label, forwarded_edge(cfg, bi, &shape)?)))
        .collect();
    let resolved = resolve(&forwarders);
    if resolved.is_empty() {
        return false;
    }

    for block in &mut cfg.blocks {
        for edge in edges_mut(&mut block.terminator) {
            let Some(target) = resolved.get(edge.to) else {
                continue;
            };
            debug_assert!(
                edge.args.is_empty(),
                "a block with no parameters is handed no arguments"
            );
            *edge.to = target.to;
            *edge.args = target.args.clone();
        }
        if let Terminator::Diamond { join, .. } = &mut block.terminator
            && let Some(target) = resolved.get(join)
        {
            *join = target.to;
        }
    }

    cfg.blocks
        .retain(|block| !resolved.contains_key(&block.label));
    cfg.label_to_block = cfg
        .blocks
        .iter()
        .enumerate()
        .map(|(bi, block)| (block.label, BlockIdx(bi)))
        .collect();
    true
}

/// The edge `bi` forwards, or `None` where `bi` is not a forwarder. Block 0
/// is the entry and stands whatever it holds.
fn forwarded_edge(cfg: &CfgBody, bi: usize, shape: &LoopShape) -> Option<Edge> {
    let block = &cfg.blocks[bi];
    let Terminator::Jump { label, args } = &block.terminator else {
        return None;
    };
    let forwards = bi != 0
        && block.params.is_empty()
        && block
            .insts
            .iter()
            .all(|inst| matches!(inst.kind, InstKind::Nop))
        && *label != block.label
        && cfg.label_to_block.contains_key(label)
        && !shape.held.contains(&block.label)
        && !shape.headers.contains(label)
        && !falls_through_into(cfg, bi);
    forwards.then(|| Edge {
        to: *label,
        args: args.clone(),
    })
}

/// Every block a loop is made of, by label.
struct LoopShape {
    /// The header, the block that enters it, and the blocks the loop leaves
    /// for: a block here is never removed.
    held: FxHashSet<Label>,
    /// The headers alone: a header never gains a second way in, so no edge
    /// is redirected to one.
    headers: FxHashSet<Label>,
}

impl LoopShape {
    fn of(cfg: &CfgBody) -> LoopShape {
        let domtree = DomTree::build(cfg);
        let mut shape = LoopShape {
            held: FxHashSet::default(),
            headers: FxHashSet::default(),
        };
        for loop_ in loops::natural_loops_innermost_first(cfg, &domtree) {
            let header = cfg.blocks[loop_.header.0].label;
            shape.headers.insert(header);
            shape.held.insert(header);
            let entering = loop_.entering.iter().copied();
            let leaving = loop_
                .blocks()
                .flat_map(|block| cfg.successors(block))
                .filter(|block| !loop_.contains(*block));
            shape.held.extend(
                entering
                    .chain(leaving)
                    .map(|block| cfg.blocks[block.0].label),
            );
        }
        shape
    }
}

/// `Terminator::Fallthrough` names its successor by position rather than by
/// label, so the block below one is the one reference a redirect cannot
/// rewrite, and it stays.
fn falls_through_into(cfg: &CfgBody, bi: usize) -> bool {
    bi > 0 && matches!(cfg.blocks[bi - 1].terminator, Terminator::Fallthrough)
}

/// A forwarder whose target is itself a forwarder hands on the target's
/// edge, so each chain is followed to the block that is not one. A cycle of
/// forwarders is a loop with no exit to hand out: its blocks stay.
fn resolve(forwarders: &FxHashMap<Label, Edge>) -> FxHashMap<Label, Edge> {
    forwarders
        .iter()
        .filter_map(|(from, edge)| Some((*from, follow(forwarders, *from, edge)?)))
        .collect()
}

fn follow(forwarders: &FxHashMap<Label, Edge>, from: Label, edge: &Edge) -> Option<Edge> {
    let mut seen: FxHashSet<Label> = FxHashSet::from_iter([from]);
    let mut edge = edge.clone();
    while let Some(next) = forwarders.get(&edge.to) {
        if !seen.insert(edge.to) {
            return None;
        }
        edge = next.clone();
    }
    Some(edge)
}

fn edges_mut(term: &mut Terminator) -> Vec<EdgeMut<'_>> {
    match term {
        Terminator::Jump { label, args } => vec![EdgeMut { to: label, args }],
        Terminator::JumpIf {
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        }
        | Terminator::Diamond {
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        } => vec![
            EdgeMut {
                to: then_label,
                args: then_args,
            },
            EdgeMut {
                to: else_label,
                args: else_args,
            },
        ],
        Terminator::Switch { arms, default, .. } => arms
            .iter_mut()
            .map(|(_, label, args)| EdgeMut { to: label, args })
            .chain(
                default
                    .iter_mut()
                    .map(|(label, args)| EdgeMut { to: label, args }),
            )
            .collect(),
        // A `For`'s body edge is not redirected: the terminator fills that
        // block's leading parameters by position, and a block reached
        // through a forwarder is a different block (RFC-0057). The exit
        // edge carries the loop's carried values and nothing else, so it
        // redirects as a `Jump` does.
        Terminator::For {
            exit, exit_args, ..
        } => vec![EdgeMut {
            to: exit,
            args: exit_args,
        }],
        Terminator::Return { .. } | Terminator::Diverge | Terminator::Fallthrough => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg;
    use crate::ir::{DebugInfo, Inst, MirBody};
    use acvus_utils::{LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_cfg(insts: Vec<InstKind>) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..8 {
            factory.next();
        }
        cfg::promote(MirBody {
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
            order_param: None,
            task: crate::ty::Task::Sync,
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        })
    }

    fn block_label(label: u32, params: Vec<ValueId>) -> InstKind {
        InstKind::BlockLabel {
            label: Label(label),
            params,
        }
    }

    fn jump(label: u32, args: Vec<ValueId>) -> InstKind {
        InstKind::Jump {
            label: Label(label),
            args,
        }
    }

    fn branch(cond: ValueId, then_label: u32, else_label: u32) -> InstKind {
        InstKind::JumpIf {
            cond,
            then_label: Label(then_label),
            then_args: vec![],
            else_label: Label(else_label),
            else_args: vec![],
        }
    }

    fn truth(dst: ValueId) -> InstKind {
        InstKind::Const {
            dst,
            value: acvus_ast::Literal::Bool(true),
        }
    }

    fn ret(value: ValueId) -> InstKind {
        InstKind::Return { value, order: None }
    }

    fn diamond_of_two_forwarders() -> CfgBody {
        make_cfg(vec![
            truth(v(0)),
            branch(v(0), 0, 1),
            block_label(0, vec![]),
            InstKind::Nop,
            jump(2, vec![v(0)]),
            block_label(1, vec![]),
            InstKind::Nop,
            jump(2, vec![v(0)]),
            block_label(2, vec![v(1)]),
            ret(v(1)),
        ])
    }

    #[test]
    fn both_arms_of_a_diamond_that_only_jump_become_the_join() {
        let mut cfg = diamond_of_two_forwarders();

        run(&mut cfg);

        assert_eq!(cfg.blocks.len(), 2, "the two arms are gone");
        let Terminator::JumpIf {
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        } = &cfg.blocks[0].terminator
        else {
            panic!("the entry still branches");
        };
        assert_eq!(*then_label, Label(2));
        assert_eq!(*else_label, Label(2));
        assert_eq!(then_args.as_slice(), &[v(0)]);
        assert_eq!(else_args.as_slice(), &[v(0)]);
        assert_eq!(cfg.label_to_block[&Label(2)], BlockIdx(1));
    }

    #[test]
    fn a_forwarder_to_a_forwarder_collapses_to_the_block_that_is_neither() {
        let mut cfg = make_cfg(vec![
            truth(v(0)),
            jump(0, vec![]),
            block_label(0, vec![]),
            InstKind::Nop,
            jump(1, vec![]),
            block_label(1, vec![]),
            InstKind::Nop,
            jump(2, vec![v(0)]),
            block_label(2, vec![v(1)]),
            ret(v(1)),
        ]);

        run(&mut cfg);

        assert_eq!(cfg.blocks.len(), 2);
        let Terminator::Jump { label, args } = &cfg.blocks[0].terminator else {
            panic!("the entry still jumps");
        };
        assert_eq!(*label, Label(2));
        assert_eq!(args.as_slice(), &[v(0)]);
    }

    #[test]
    fn a_block_with_a_parameter_is_not_a_forwarder() {
        let mut cfg = make_cfg(vec![
            truth(v(0)),
            jump(0, vec![v(0)]),
            block_label(0, vec![v(1)]),
            InstKind::Nop,
            jump(2, vec![v(1)]),
            block_label(2, vec![v(2)]),
            ret(v(2)),
        ]);

        run(&mut cfg);

        assert_eq!(cfg.blocks.len(), 3, "the parameter is a phi and stays");
    }

    #[test]
    fn a_ring_of_blocks_that_only_jump_keeps_its_blocks() {
        let mut cfg = make_cfg(vec![
            truth(v(0)),
            jump(0, vec![]),
            block_label(0, vec![]),
            InstKind::Nop,
            jump(1, vec![]),
            block_label(1, vec![]),
            InstKind::Nop,
            jump(0, vec![]),
        ]);

        run(&mut cfg);

        assert_eq!(
            cfg.blocks.len(),
            3,
            "the ring is a loop: its header and the block entering it are its shape"
        );
    }

    #[test]
    fn a_chain_that_closes_on_itself_resolves_to_nothing() {
        let ring = FxHashMap::from_iter([
            (
                Label(0),
                Edge {
                    to: Label(1),
                    args: vec![],
                },
            ),
            (
                Label(1),
                Edge {
                    to: Label(0),
                    args: vec![],
                },
            ),
        ]);

        assert!(resolve(&ring).is_empty());
    }

    #[test]
    fn a_block_a_fallthrough_reaches_stays() {
        let mut cfg = make_cfg(vec![
            truth(v(0)),
            block_label(0, vec![]),
            InstKind::Nop,
            jump(1, vec![]),
            block_label(1, vec![]),
            ret(v(0)),
        ]);

        run(&mut cfg);

        assert_eq!(
            cfg.blocks.len(),
            3,
            "the entry names its successor by position"
        );
    }

    #[test]
    fn a_loop_headers_preheader_stays_though_it_only_jumps() {
        let mut cfg = make_cfg(vec![
            truth(v(0)),
            branch(v(0), 0, 3),
            block_label(0, vec![]),
            InstKind::Nop,
            jump(1, vec![]),
            block_label(1, vec![]),
            branch(v(0), 2, 3),
            block_label(2, vec![]),
            truth(v(1)),
            jump(1, vec![]),
            block_label(3, vec![]),
            ret(v(0)),
        ]);

        run(&mut cfg);

        assert_eq!(
            cfg.blocks.len(),
            5,
            "the header keeps the block that enters it"
        );
    }

    #[test]
    fn the_block_a_loop_leaves_for_stays_though_it_only_jumps() {
        let mut cfg = make_cfg(vec![
            truth(v(0)),
            jump(0, vec![]),
            block_label(0, vec![]),
            branch(v(0), 1, 2),
            block_label(1, vec![]),
            truth(v(1)),
            jump(0, vec![]),
            block_label(2, vec![]),
            InstKind::Nop,
            jump(3, vec![]),
            block_label(3, vec![]),
            ret(v(0)),
        ]);

        run(&mut cfg);

        assert_eq!(
            cfg.blocks.len(),
            5,
            "the exit is where the loop's shape ends"
        );
    }
}
