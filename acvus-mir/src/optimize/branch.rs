//! A branch on a constant is a jump (RFC-0071 Decision 5).
//!
//! The arms the constant decides against are then blocks no path reaches and
//! leave with them, so the `$` names only they read are no longer inputs the
//! host must supply. `optimize::fold` decides the constant; this decides the
//! edge.
//!
//! Obligation across passes: a `Diamond` this pass rewrites stops being a
//! branch, so its label leaves `demoted_diamonds` and `optimize::rejoin` has
//! nothing to restore.

use acvus_ast::Literal;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::analysis::domtree::DomTree;
use crate::cfg::{BlockIdx, CfgBody, Terminator, prune, reachable};
use crate::ir::{InstKind, Label, SwitchKey, ValueId};

pub fn run(interner: &Interner, cfg: &mut CfgBody) {
    let constants = Constants::of(cfg);
    let headers = LoopHeaders::of(cfg);
    let decided: Vec<Decided> = (0..cfg.blocks.len())
        .map(BlockIdx)
        .filter(|at| !headers.holds(*at))
        .filter_map(|at| {
            let taken = constants.decide(interner, &cfg.blocks[at.0].terminator)?;
            Some(Decided { at, taken })
        })
        .collect();
    if decided.is_empty() {
        return;
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

struct Constants {
    scalars: FxHashMap<ValueId, Literal>,
    texts: FxHashMap<ValueId, String>,
}

impl Constants {
    fn of(cfg: &CfgBody) -> Constants {
        let mut scalars = FxHashMap::default();
        let mut texts = FxHashMap::default();
        let insts = cfg.blocks.iter().flat_map(|block| &block.insts);
        for inst in insts {
            match &inst.kind {
                InstKind::Const { dst, value } => {
                    scalars.insert(*dst, value.clone());
                }
                InstKind::ConstStr { dst, text } => {
                    texts.insert(*dst, text.clone());
                }
                _ => {}
            }
        }
        Constants { scalars, texts }
    }

    fn decide(&self, interner: &Interner, terminator: &Terminator) -> Option<Taken> {
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
            } => {
                let Some(Literal::Bool(held)) = self.scalars.get(cond) else {
                    return None;
                };
                match held {
                    true => Some(Taken {
                        label: *then_label,
                        args: then_args.clone(),
                    }),
                    false => Some(Taken {
                        label: *else_label,
                        args: else_args.clone(),
                    }),
                }
            }
            // A `Switch` without a default is exhaustive over its arms
            // (RFC-0051), so a constant tag outside them names no edge and
            // the dispatch stands as the source wrote it.
            Terminator::Switch { tag, arms, default } => {
                let key = self.key(interner, *tag)?;
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

    fn key(&self, interner: &Interner, tag: ValueId) -> Option<SwitchKey> {
        if let Some(text) = self.texts.get(&tag) {
            return Some(SwitchKey::Str(interner.intern(text)));
        }
        SwitchKey::of_literal(self.scalars.get(&tag)?, interner)
    }
}
