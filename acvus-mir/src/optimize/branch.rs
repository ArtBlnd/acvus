//! A branch on a known value is a jump (RFC-0071 rule 5).
//!
//! The arms the value decides against are then blocks no path reaches and
//! leave with them, so the `$` names only they read are no longer inputs the
//! host must supply. `analysis::known` says what a value is before the body
//! runs: a constant, text, or a variant, object, tuple or array built from
//! known parts, read out of a slot written once or through a reference to
//! one, tested by a pattern, or passed to a block by every edge alike. This
//! pass decides a `JumpIf` or a `Diamond` on a known `bool`, and a `Switch`
//! on a known variant's tag or a known scalar or text. Deciding one branch
//! removes an edge, which can make the next block parameter known, so the
//! pass repeats until a round decides nothing; every round replaces at least
//! one two-way or many-way terminator with a jump, so the rounds end.
//!
//! Obligation across artifacts: a decision here is the one
//! `acvus-interpreter` makes at run time on the same value, which
//! `analysis::known` states arm by arm. A decision it cannot reproduce
//! exactly is left to the machine.
//!
//! Obligation across passes: a `Diamond` this pass rewrites stops being a
//! branch, so its label leaves `demoted_diamonds` and `optimize::rejoin` has
//! nothing to restore.

use acvus_utils::Interner;

use crate::analysis::domtree::DomTree;
use crate::analysis::known::KnownValues;
use crate::cfg::{BlockIdx, CfgBody, Terminator, prune, reachable};
use crate::ir::{Label, ValueId};

pub fn run(interner: &Interner, cfg: &mut CfgBody) {
    while decide_once(interner, cfg) {}
}

fn decide_once(interner: &Interner, cfg: &mut CfgBody) -> bool {
    let known = KnownValues::of(interner, cfg);
    let headers = LoopHeaders::of(cfg);
    let decided: Vec<Decided> = (0..cfg.blocks.len())
        .map(BlockIdx)
        .filter(|at| !headers.holds(*at))
        .filter_map(|at| {
            let taken = decide(interner, cfg, &known, &cfg.blocks[at.0].terminator)?;
            Some(Decided { at, taken })
        })
        .collect();
    if decided.is_empty() {
        return false;
    }
    for decided in decided {
        let block = &mut cfg.blocks[decided.at.0];
        block.terminator = Terminator::Jump {
            label: decided.taken.label,
            args: decided.taken.args,
        };
        cfg.demoted_diamonds.remove(&block.label);
    }
    let alive = reachable(cfg);
    prune(cfg, &alive);
    true
}

/// Obligation across artifacts: `acvus_interpreter::prepare` reads a loop
/// off the test its header ends in, and the `break` inside the body escapes
/// that region rather than replacing it. A header left with an
/// unconditional jump is no longer a loop it recognizes, so this pass
/// decides every branch except the one a loop tests itself with.
struct LoopHeaders {
    held: Vec<bool>,
}

impl LoopHeaders {
    fn of(cfg: &CfgBody) -> LoopHeaders {
        let domtree = DomTree::build(cfg);
        let preds = cfg.predecessors();
        let held = (0..cfg.blocks.len())
            .map(BlockIdx)
            .map(|at| {
                preds
                    .get(&at)
                    .into_iter()
                    .flatten()
                    .any(|back| domtree.dominates(at, *back))
            })
            .collect();
        LoopHeaders { held }
    }

    fn holds(&self, at: BlockIdx) -> bool {
        self.held[at.0]
    }
}

struct Decided {
    at: BlockIdx,
    taken: Taken,
}

struct Taken {
    label: Label,
    args: Vec<ValueId>,
}

fn decide(
    interner: &Interner,
    cfg: &CfgBody,
    known: &KnownValues,
    terminator: &Terminator,
) -> Option<Taken> {
    match terminator {
        Terminator::JumpIf {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
        }
        | Terminator::Diamond {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        } => match known.bool(*cond)? {
            true => Some(Taken {
                label: *then_label,
                args: then_args.clone(),
            }),
            false => Some(Taken {
                label: *else_label,
                args: else_args.clone(),
            }),
        },
        // A `Switch` without a default is exhaustive over its arms
        // (RFC-0051), so a known key outside them names no edge and the
        // dispatch stands as the source wrote it.
        Terminator::Switch { tag, arms, default } => {
            let (kind, _, _) = arms.first()?;
            let key = known.switch_key(interner, *tag, cfg.val_types.get(tag)?, *kind)?;
            let arm = arms.iter().find(|(held, _, _)| *held == key);
            match arm {
                Some((_, label, args)) => Some(Taken {
                    label: *label,
                    args: args.clone(),
                }),
                None => default.as_ref().map(|(label, args)| Taken {
                    label: *label,
                    args: args.clone(),
                }),
            }
        }
        Terminator::Jump { .. }
        | Terminator::For { .. }
        | Terminator::Return { .. }
        | Terminator::Diverge
        | Terminator::Fallthrough => None,
    }
}
