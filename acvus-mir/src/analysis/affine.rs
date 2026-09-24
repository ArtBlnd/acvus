//! Which values of a loop advance by a fixed step: the affine values.
//!
//! A value `v` of a loop `L` is affine when `v = base + k·step` over `L`'s
//! iteration number `k`, with `base` and `step` invariant in `L`
//! (`analysis::loops::Invariants`). Three rules make one, and nothing else
//! does:
//!
//! - A `for`'s counter is affine from its terminator: a range's element is
//!   `{at, 1}` and a slice's or an array's index is `{0, 1}` (RFC-0057
//!   rule 3).
//! - A carried header parameter whose entering edges all send `b` and
//!   whose back edges all send `p + c`, with `c` invariant, is `{b, c}`.
//! - `a·v` and `v + b` of an affine `v`, with `a` and `b` invariant, are
//!   affine: `{a·base, a·step}` and `{base + b, step}`.
//!
//! Only integers are affine. `+` and `*` wrap at the operand's width
//! (RFC-0037), and arithmetic modulo `2^width` is a ring, so `base + k·step`
//! is the value exactly however it was accumulated. A float accumulated `k`
//! times carries the rounding of every step, which is a different number
//! (RFC-0056), so a float counter is not an induction variable, and
//! `analysis::carried` classifies it as a merge or a recurrence.
//!
//! The analysis is one loop deep. A value affine in an inner loop whose
//! base moves with the outer loop's counter is affine in the inner loop
//! with that base, and nothing here says how the base moves.
//!
//! `base` and `step` are `analysis::loops::Term`s: the type the trip count
//! is written in, over the same atoms — a constant, a value invariant in
//! the loop, a source's length — so a reader that evaluates a trip count
//! evaluates these with the same code. Beside the terms, each value keeps
//! the rule that made it, with the operands typed as the rule read them;
//! `optimize::lsr` rewrites from that and does not match the instructions
//! a second time.

use crate::ir::BinOp;
use rustc_hash::FxHashMap;

use crate::analysis::loops::{Invariant, Invariants, Loop, LoopKind, Term};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{ForSource, InstKind, ValueId};
use crate::ty::Ty;

/// Whether `+` and `*` at `ty` are arithmetic in a ring: exact under
/// reassociation and distribution, as wrapping integers are (RFC-0037).
pub fn exact_under_wrapping(ty: &Ty) -> bool {
    matches!(ty, Ty::Int(_))
}

/// An invariant operand of the instruction a rule read: the value the
/// instruction names, and how a reader above the header reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct Operand {
    pub value: ValueId,
    pub invariant: Invariant,
}

/// The rule that made a value affine, and what it read.
#[derive(Clone, Debug, PartialEq)]
pub enum Derivation {
    /// The counter a `for` terminator advances.
    Counter,
    /// A header parameter entered as `init` and advanced by `step`.
    Carried {
        init: ValueId,
        step: Operand,
    },
    Scaled {
        of: ValueId,
        factor: Operand,
    },
    Offset {
        of: ValueId,
        offset: Operand,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Affine {
    pub base: Term,
    pub step: Term,
    pub derivation: Derivation,
}

/// The affine values of one loop.
pub struct AffineValues {
    values: FxHashMap<ValueId, Affine>,
}

impl AffineValues {
    pub fn of(cfg: &CfgBody, loop_: &Loop, invariants: &Invariants) -> Self {
        let natural = &loop_.natural;
        let mut values: FxHashMap<ValueId, Affine> = FxHashMap::default();
        let integer = |value: ValueId| exact_under_wrapping(&cfg.val_types[&value]);

        if let LoopKind::For { source } = loop_.kind {
            let body = for_body(cfg, natural.header);
            let counter = cfg.blocks[body.0].params[source.counter_param()];
            let base = match source {
                ForSource::Range { at, .. } => Term::from(
                    invariants
                        .at(natural, at)
                        .expect("a `for` source is settled before the header runs"),
                ),
                ForSource::Slice(_) | ForSource::SliceMut(_) | ForSource::Array(_) => Term::int(0),
            };
            values.insert(
                counter,
                Affine {
                    base,
                    step: Term::int(1),
                    derivation: Derivation::Counter,
                },
            );
        }

        let arithmetic = Arithmetic::in_loop(cfg, loop_);

        for (index, &param) in cfg.blocks[natural.header.0].params.iter().enumerate() {
            if !integer(param) {
                continue;
            }
            let (Some(init), Some(next)) =
                (natural.entry_arg(cfg, index), natural.back_arg(cfg, index))
            else {
                continue;
            };
            let Some(sum) = arithmetic.get(next).filter(|a| a.op == BinOp::Add) else {
                continue;
            };
            let Some(step) = sum.other_than(param) else {
                continue;
            };
            let Some(invariant) = invariants.at(natural, step) else {
                continue;
            };
            values.insert(
                param,
                Affine {
                    base: Term::Value(init),
                    step: Term::from(invariant.clone()),
                    derivation: Derivation::Carried {
                        init,
                        step: Operand {
                            value: step,
                            invariant,
                        },
                    },
                },
            );
        }

        loop {
            let mut found: Vec<(ValueId, Affine)> = Vec::new();
            for (&dst, operation) in &arithmetic.by_dst {
                if values.contains_key(&dst) || !integer(dst) {
                    continue;
                }
                let (of, other) = match (
                    values.contains_key(&operation.left),
                    values.contains_key(&operation.right),
                ) {
                    (true, false) => (operation.left, operation.right),
                    (false, true) => (operation.right, operation.left),
                    _ => continue,
                };
                let Some(invariant) = invariants.at(natural, other) else {
                    continue;
                };
                let known = &values[&of];
                let term = Term::from(invariant.clone());
                let operand = Operand {
                    value: other,
                    invariant,
                };
                let affine = match operation.op {
                    BinOp::Mul => Affine {
                        base: known.base.clone().mul(term.clone()),
                        step: known.step.clone().mul(term),
                        derivation: Derivation::Scaled {
                            of,
                            factor: operand,
                        },
                    },
                    BinOp::Add => Affine {
                        base: known.base.clone().add(term),
                        step: known.step.clone(),
                        derivation: Derivation::Offset {
                            of,
                            offset: operand,
                        },
                    },
                    _ => continue,
                };
                found.push((dst, affine));
            }
            if found.is_empty() {
                break;
            }
            values.extend(found);
        }

        Self { values }
    }

    pub fn get(&self, value: ValueId) -> Option<&Affine> {
        self.values.get(&value)
    }
}

/// One `BinOp` of a loop's body, by the value it defines.
#[derive(Clone, Copy)]
pub struct Operation {
    pub op: BinOp,
    pub left: ValueId,
    pub right: ValueId,
}

impl Operation {
    /// The operand that is not `of`, when exactly one of the two is.
    pub fn other_than(&self, of: ValueId) -> Option<ValueId> {
        match (self.left == of, self.right == of) {
            (true, false) => Some(self.right),
            (false, true) => Some(self.left),
            _ => None,
        }
    }

    pub fn one_operand_is(&self, of: ValueId) -> bool {
        self.other_than(of).is_some()
    }
}

/// Every `BinOp` inside one loop, by the value it defines.
pub struct Arithmetic {
    by_dst: FxHashMap<ValueId, Operation>,
}

impl Arithmetic {
    pub fn in_loop(cfg: &CfgBody, loop_: &Loop) -> Self {
        let by_dst = loop_
            .natural
            .blocks()
            .flat_map(|b| cfg.blocks[b.0].insts.iter())
            .filter_map(|inst| match &inst.kind {
                InstKind::BinOp {
                    dst,
                    op,
                    left,
                    right,
                } => Some((
                    *dst,
                    Operation {
                        op: *op,
                        left: *left,
                        right: *right,
                    },
                )),
                _ => None,
            })
            .collect();
        Self { by_dst }
    }

    pub fn get(&self, dst: ValueId) -> Option<&Operation> {
        self.by_dst.get(&dst)
    }
}

/// The block a `for` header's body edge enters.
///
/// # Panics
/// If `header` does not end in a `For`, or its body label names no block.
pub fn for_body(cfg: &CfgBody, header: BlockIdx) -> BlockIdx {
    let Terminator::For { stages, .. } = &cfg.blocks[header.0].terminator else {
        panic!(
            "block {} is a `for` header and does not end in `For`",
            header.0
        )
    };
    cfg.label_to_block[&stages.body()]
}
