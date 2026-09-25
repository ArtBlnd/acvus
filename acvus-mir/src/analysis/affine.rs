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
//!   whose back edges all send `p + c`, with `c` invariant, is `{b, c}`;
//!   one whose back edges all send `p − 1` is `{b, −1}` (RFC-0094 rule 2).
//! - `a·v`, `v + b`, `v − b` and `b − v` of an affine `v`, with `a` and `b`
//!   invariant, are affine: `{a·base, a·step}`, `{base + b, step}`,
//!   `{base − b, step}` and `{b − base, 0 − step}` (RFC-0066 rule 4).
//! - A call's result its postcondition states `= len(s)`, read before a
//!   call every iteration makes exactly once whose postcondition states
//!   `len(s) = old(len(s)) + c`, with nothing else in the loop writing `s`,
//!   is `{len(s) on entry, c}` (RFC-0066 rule 4, RFC-0082 rule 6): the
//!   iterations before it made the call once each and nothing else moved
//!   the length.
//!
//! An invariant here is RFC-0066 rule 3's, a standing operation of the
//! body among them (`analysis::loops::Invariance`). A reader that writes
//! `base + k·step` above the header or at the head of the body asks each
//! operand for [`Invariance::above`] and declines a standing one.
//!
//! Only integers are affine, and a `+`, `−` or `*` of either kind makes one
//! (`ir::Overflow`). A wrapping one is arithmetic modulo `2^width`, a
//! ring; a trapping one gives the integer result on every run that goes
//! past it, since a run whose result does not fit ends there (RFC-0037
//! rule 3), and that result is the same word modulo `2^width`. So on every
//! run that reaches it, `base + k·step` computed modulo `2^width` is the
//! value exactly however it was accumulated. A float accumulated `k` times
//! carries the rounding of every step, which is a different number
//! (RFC-0056), so a float counter is not an induction variable, and
//! `analysis::carried` classifies it as state.
//!
//! The analysis is one loop deep. A value affine in an inner loop whose
//! base moves with the outer loop's counter is affine in the inner loop
//! with that base, and nothing here says how the base moves.
//!
//! `base` and `step` are `analysis::loops::Term`s: the type the trip count
//! is written in, over the same atoms — a constant, a value invariant in
//! the loop, a source's length, a storage's length on entry — so a reader that evaluates a trip count
//! evaluates these with the same code. Beside the terms, each value keeps
//! the rule that made it, with the operands typed as the rule read them;
//! `optimize::lsr` rewrites from that and does not match the instructions
//! a second time.

use acvus_ast::Literal;

use crate::ir::BinOp;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::domtree::DomTree;
use crate::analysis::loans::Loans;
use crate::analysis::loops::{
    Invariance, Invariant, Invariants, Loop, LoopKind, NaturalLoop, Term,
};
use crate::analysis::targets;
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{ForSource, InstKind, RefTarget, ValueId};
use crate::laws::{LawTable, PostTerm, Postcondition, Relation, Subject};
use crate::ty::{Mutability, Ty};

/// Whether `+` and `*` at `ty` are arithmetic in a ring on every run that
/// reaches them: exact under reassociation and distribution modulo
/// `2^width`, as a wrapping integer operation is and as a trapping one is
/// wherever it does not end the run (RFC-0037 rule 3).
pub fn exact_under_wrapping(ty: &Ty) -> bool {
    matches!(ty, Ty::Int(_))
}

/// An invariant operand of the instruction a rule read: the value the
/// instruction names, and why it is invariant, which says whether and how a
/// reader above the header reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct Operand {
    pub value: ValueId,
    pub invariance: Invariance,
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
    /// A header parameter entered as `init` that every back edge sends
    /// `p − 1`: `{init, −1}` (RFC-0094 rule 2).
    CountsDown { init: ValueId, one: Operand },
    Scaled {
        of: ValueId,
        factor: Operand,
    },
    Offset {
        of: ValueId,
        offset: Operand,
    },
    /// `of − offset`.
    Lowered {
        of: ValueId,
        offset: Operand,
    },
    /// `from − of`: the step negated.
    Reflected {
        of: ValueId,
        from: Operand,
    },
    /// The length of storage slot `storage`, read before the one call each
    /// iteration makes that grows it.
    Length { storage: ValueId },
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
    pub fn of(cfg: &CfgBody, loop_: &Loop, invariants: &Invariants, laws: &LawTable) -> Self {
        let natural = &loop_.natural;
        let mut values: FxHashMap<ValueId, Affine> = FxHashMap::default();
        let integer = |value: ValueId| exact_under_wrapping(&cfg.val_types[&value]);

        if let LoopKind::For { source } = loop_.kind {
            let body = for_body(cfg, natural.header);
            let counter = cfg.blocks[body.0].params[source.counter_param()];
            let base = match source {
                ForSource::Range { at, .. } => Term::from(
                    invariants
                        .above(natural, at)
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
            if let Some(counted_down) = counts_down(&arithmetic, invariants, natural, param, init, next)
            {
                values.insert(param, counted_down);
                continue;
            }
            let Some(sum) = arithmetic
                .get(next)
                .filter(|a| matches!(a.op, BinOp::Add(_)))
            else {
                continue;
            };
            let Some(step) = sum.other_than(param) else {
                continue;
            };
            let Some(invariance) = invariants.in_loop(natural, step) else {
                continue;
            };
            values.insert(
                param,
                Affine {
                    base: Term::Value(init),
                    step: Term::from(invariance.clone()),
                    derivation: Derivation::Carried {
                        init,
                        step: Operand {
                            value: step,
                            invariance,
                        },
                    },
                },
            );
        }

        values.extend(lengths(cfg, natural, invariants, laws));

        loop {
            let mut found: Vec<(ValueId, Affine)> = Vec::new();
            for (&dst, operation) in &arithmetic.by_dst {
                if values.contains_key(&dst) || !integer(dst) {
                    continue;
                }
                let (of, other, affine_on_left) = match (
                    values.contains_key(&operation.left),
                    values.contains_key(&operation.right),
                ) {
                    (true, false) => (operation.left, operation.right, true),
                    (false, true) => (operation.right, operation.left, false),
                    _ => continue,
                };
                let Some(invariance) = invariants.in_loop(natural, other) else {
                    continue;
                };
                let known = &values[&of];
                let term = Term::from(invariance.clone());
                let operand = Operand {
                    value: other,
                    invariance,
                };
                let affine = match operation.op {
                    BinOp::Mul(_) => Affine {
                        base: known.base.clone().mul(term.clone()),
                        step: known.step.clone().mul(term),
                        derivation: Derivation::Scaled {
                            of,
                            factor: operand,
                        },
                    },
                    BinOp::Add(_) => Affine {
                        base: known.base.clone().add(term),
                        step: known.step.clone(),
                        derivation: Derivation::Offset {
                            of,
                            offset: operand,
                        },
                    },
                    BinOp::Sub(_) if affine_on_left => Affine {
                        base: known.base.clone().sub(term),
                        step: known.step.clone(),
                        derivation: Derivation::Lowered {
                            of,
                            offset: operand,
                        },
                    },
                    BinOp::Sub(_) => Affine {
                        base: term.sub(known.base.clone()),
                        step: Term::int(0).sub(known.step.clone()),
                        derivation: Derivation::Reflected { of, from: operand },
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

/// `param` entered as `init` whose back edges send `next`, where `next` is
/// `param − 1` with `1` the integer one invariant in the loop.
fn counts_down(
    arithmetic: &Arithmetic,
    invariants: &Invariants,
    natural: &NaturalLoop,
    param: ValueId,
    init: ValueId,
    next: ValueId,
) -> Option<Affine> {
    let difference = arithmetic
        .get(next)
        .filter(|a| matches!(a.op, BinOp::Sub(_)) && a.left == param)?;
    let invariance = invariants.in_loop(natural, difference.right)?;
    let is_one = match invariance.above()? {
        Invariant::Word(literal) => literal.desugared() == Literal::Int(1),
        Invariant::Outside(value) => invariants
            .word(*value)
            .is_some_and(|literal| literal.desugared() == Literal::Int(1)),
    };
    is_one.then(|| Affine {
        base: Term::Value(init),
        step: Term::int(0).sub(Term::from(invariance.clone())),
        derivation: Derivation::CountsDown {
            init,
            one: Operand {
                value: difference.right,
                invariance,
            },
        },
    })
}

/// Where an instruction stands: its block and its index there.
#[derive(Clone, Copy, PartialEq, Eq)]
struct At {
    block: BlockIdx,
    index: usize,
}

/// A call whose postcondition states `len(p) = old(len(p)) + c` of the
/// storage its argument `p` lends whole and exclusively.
struct Grows {
    at: At,
    storage: ValueId,
    by: Term,
}

/// A call whose postcondition states `ret = len(p)` of the storage its
/// argument `p` lends whole.
struct ReadsLength {
    at: At,
    dst: ValueId,
    storage: ValueId,
}

/// RFC-0066 rule 4's `len(s)` clause: each value a call states `= len(s)`,
/// read before the one call every iteration makes that states
/// `len(s) = old(len(s)) + c`, where nothing else in the loop writes `s`.
fn lengths(
    cfg: &CfgBody,
    natural: &NaturalLoop,
    invariants: &Invariants,
    laws: &LawTable,
) -> Vec<(ValueId, Affine)> {
    let refs: FxHashMap<ValueId, (&RefTarget, bool, Mutability)> = cfg
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter_map(|inst| match &inst.kind {
            InstKind::Ref {
                dst,
                target,
                path,
                mutability,
            } => Some((*dst, (target, path.is_empty(), *mutability))),
            _ => None,
        })
        .collect();
    // The slot `value` is a reference to, whole, taken at `mutability`.
    let whole = |value: ValueId, mutability: Mutability| match refs.get(&value) {
        Some((RefTarget::Var(slot) | RefTarget::Param(slot), true, taken))
            if *taken == mutability =>
        {
            Some(*slot)
        }
        _ => None,
    };
    let mut grows: Vec<Grows> = Vec::new();
    let mut reads: Vec<ReadsLength> = Vec::new();
    for block in natural.blocks().filter(|block| *block != natural.header) {
        for (index, inst) in cfg.blocks[block.0].insts.iter().enumerate() {
            let InstKind::FunctionCall {
                dst, callee, args, ..
            } = &inst.kind
            else {
                continue;
            };
            let at = At { block, index };
            for postcondition in laws.postconditions_of(callee) {
                if let Some((param, by)) = growth(postcondition) {
                    let by = match by {
                        PostTerm::Const(c) => Some(Term::int(*c)),
                        PostTerm::Param(k) => args
                            .get(*k)
                            .and_then(|arg| invariants.in_loop(natural, *arg))
                            .map(Term::from),
                        _ => None,
                    };
                    if let (Some(by), Some(storage)) = (
                        by,
                        args.get(param).and_then(|arg| whole(*arg, Mutability::Mut)),
                    ) {
                        grows.push(Grows { at, storage, by });
                    }
                }
                if let Some(param) = length_of(postcondition)
                    && let Some(storage) =
                        args.get(param).and_then(|arg| whole(*arg, Mutability::Shared))
                    && exact_under_wrapping(&cfg.val_types[dst])
                {
                    reads.push(ReadsLength {
                        at,
                        dst: *dst,
                        storage,
                    });
                }
            }
        }
    }
    if grows.is_empty() || reads.is_empty() {
        return Vec::new();
    }
    let loans = Loans::build(cfg);
    let domtree = DomTree::build(cfg);
    let mut found = Vec::new();
    for read in reads {
        let [grow] = &grows
            .iter()
            .filter(|grow| grow.storage == read.storage)
            .collect::<Vec<_>>()[..]
        else {
            continue;
        };
        let only_writer = writers(cfg, &loans, natural, read.storage) == [grow.at];
        if only_writer
            && once_per_iteration(cfg, &domtree, natural, grow.at.block)
            && !reached_after(cfg, natural, grow.at, read.at)
        {
            found.push((
                read.dst,
                Affine {
                    base: Term::LenOnEntry(read.storage),
                    step: grow.by.clone(),
                    derivation: Derivation::Length {
                        storage: read.storage,
                    },
                },
            ));
        }
    }
    found
}

/// The parameter `p` and the term `c` of `len(p) = old(len(p)) + c`, in
/// either order of the relation and of the sum.
fn growth(postcondition: &Postcondition) -> Option<(usize, &PostTerm)> {
    let Postcondition {
        left,
        relation: Relation::Eq,
        right,
    } = postcondition
    else {
        return None;
    };
    let grown = |now: &PostTerm, sum: &'_ PostTerm| -> Option<usize> {
        let PostTerm::Len(Subject::Param(param)) = now else {
            return None;
        };
        let PostTerm::Old(was) = sum else {
            return None;
        };
        (**was == PostTerm::Len(Subject::Param(*param))).then_some(*param)
    };
    [(left, right), (right, left)]
        .into_iter()
        .find_map(|(now, other)| {
            let PostTerm::Add(a, b) = other else {
                return None;
            };
            [(&**a, &**b), (&**b, &**a)]
                .into_iter()
                .find_map(|(was, by)| grown(now, was).map(|param| (param, by)))
        })
}

/// The parameter `p` of `ret = len(p)`, in either order.
fn length_of(postcondition: &Postcondition) -> Option<usize> {
    match postcondition {
        Postcondition {
            left: PostTerm::Ret,
            relation: Relation::Eq,
            right: PostTerm::Len(Subject::Param(param)),
        }
        | Postcondition {
            left: PostTerm::Len(Subject::Param(param)),
            relation: Relation::Eq,
            right: PostTerm::Ret,
        } => Some(*param),
        _ => None,
    }
}

/// Every place in the loop that writes `storage`: an instruction whose
/// storage effect writes it, and a `for` whose exclusive source holds it.
fn writers(cfg: &CfgBody, loans: &Loans<'_>, natural: &NaturalLoop, storage: ValueId) -> Vec<At> {
    let mut found = Vec::new();
    for block in natural.blocks() {
        let held = &cfg.blocks[block.0];
        for (index, inst) in held.insts.iter().enumerate() {
            if targets::effect(loans, &inst.kind).writes.contains(&storage) {
                found.push(At { block, index });
            }
        }
        if let Terminator::For {
            source: ForSource::SliceMut(source),
            ..
        } = &held.terminator
            && loans
                .holds(*source)
                .any(|loan| loan.storage.slot() == Some(storage))
        {
            found.push(At {
                block,
                index: held.insts.len(),
            });
        }
    }
    found
}

/// The blocks of the loop an iteration reaches after `from`, without
/// passing the header.
fn reached_within(cfg: &CfgBody, natural: &NaturalLoop, from: BlockIdx) -> FxHashSet<BlockIdx> {
    let mut reached: FxHashSet<BlockIdx> = FxHashSet::default();
    let mut work = vec![from];
    while let Some(block) = work.pop() {
        for succ in cfg.successors(block) {
            if succ != natural.header && natural.contains(succ) && reached.insert(succ) {
                work.push(succ);
            }
        }
    }
    reached
}

/// Whether every iteration runs `block` exactly once: no path returns to it
/// within one iteration, and it dominates every latch and every block an
/// iteration can leave the loop from, so no iteration goes on or ends
/// without it.
fn once_per_iteration(
    cfg: &CfgBody,
    domtree: &DomTree,
    natural: &NaturalLoop,
    block: BlockIdx,
) -> bool {
    if block == natural.header || reached_within(cfg, natural, block).contains(&block) {
        return false;
    }
    natural
        .blocks()
        .filter(|at| *at != natural.header)
        .filter(|at| {
            natural.latches.contains(at)
                || matches!(
                    cfg.blocks[at.0].terminator,
                    Terminator::Return { .. } | Terminator::Diverge
                )
                || cfg
                    .successors(*at)
                    .iter()
                    .any(|succ| !natural.contains(*succ))
        })
        .all(|at| domtree.dominates(block, at))
}

/// Whether an iteration can reach `later` after `earlier` has run.
fn reached_after(cfg: &CfgBody, natural: &NaturalLoop, earlier: At, later: At) -> bool {
    (later.block == earlier.block && later.index > earlier.index)
        || reached_within(cfg, natural, earlier.block).contains(&later.block)
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
