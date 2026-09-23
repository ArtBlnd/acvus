//! A `while` that is a counted traversal becomes a `for` over a range
//! (RFC-0081).
//!
//! In `let i = b; while i < n { …; i = i + 1; }` the header's `jump_if`
//! becomes `For { source: Range { at: b, hi: n }, … }` and the body block
//! gains the counter that terminator fills. The pass does not touch `i` and
//! does not compute its value after the loop: the exit edge carries it as
//! before, and normalizing it is IV canonicalization's (RFC-0066 rule 7).
//!
//! The rewrite is one half of a contract whose other half is the
//! interpreter. `acvus_interpreter::ops::control::Range` runs the body while
//! `T::read(at) < T::read(hi)` and then advances with `wrapping_add` at `T`;
//! the `while` compares with `ops::arith::word::lt` at the same `T` and its
//! `i + 1` wraps at `T` (RFC-0037). Both start at `b` on every entry, so both
//! loops see `b + k` on their `k`-th header visit. If either operation
//! changes, the corpus under `acvus-interpreter-test/tests/soundness/
//! while-to-for` is what disagrees across optimization levels.
//!
//! The machine reads a range's bounds on the edge that enters the loop,
//! before the header runs. A bound the header computes is therefore
//! computed again at the end of the entering block, from the same operands:
//! a literal, as `while i < 10` lowers, the way `optimize::lsr` writes a
//! literal again above a header (RFC-0056), and `@n * 2` the same way.
//!
//! A [`Step`] is deterministic: a word operation, or a call of an extern
//! declared `pure` whose reference arguments are shared and defined
//! outside the loop, which the borrow check keeps unwritten while the
//! header reads them (RFC-0064). Over operands defined outside the loop it
//! gives the same value, and raises the same trap, on every header visit.
//! The header runs on every entry before any body block, so one evaluation
//! on the entering edge is that first visit moved ahead of the header's
//! other instructions. A `/`, a `%` and a call can trap, so a bound holding
//! one is promoted only when no header instruction before its last such
//! step, other than a step of the bound, has an effect or can trap: then
//! the entry raises exactly the trap the first visit raised, and in the
//! header's order.

use acvus_ast::{Literal, Span, SuffixedInt};

use crate::ir::BinOp;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::analysis::affine::{AffineValues, Derivation};
use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loops::{Invariant, Invariants, Loop, LoopKind, LoopNest};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Callee, ExitTrip, ForSource, Inst, InstKind, Label, ValOrigin, ValueId};
use crate::ty::{Mutability, Ty};

pub fn run(cfg: &mut CfgBody) {
    let domtree = DomTree::build(cfg);
    let invariants = Invariants::of(cfg);
    let nest = LoopNest::of(cfg, &domtree, &invariants);
    let preds = cfg.predecessors();
    let literals = const_literals(cfg);
    let counted: Vec<Counted> = nest
        .iter()
        .filter_map(|(_, loop_)| {
            Recognizer {
                cfg,
                loop_,
                invariants: &invariants,
                preds: &preds,
                literals: &literals,
            }
            .counted()
        })
        .collect();
    for loop_ in counted {
        loop_.apply(cfg);
    }
}

enum Bound {
    Outside(ValueId),
    Computed { steps: Vec<Step>, value: ValueId },
}

struct Step {
    span: Span,
    /// The copies are written in this order, so two steps that trap raise
    /// in the order the header raised them.
    header_position: usize,
    dst: ValueId,
    kind: StepKind,
}

enum StepKind {
    Word(Literal),
    Arith {
        op: Arith,
        left: Operand,
        right: Operand,
    },
    Call {
        callee: Callee,
        callee_ty: Ty,
        args: Vec<Operand>,
    },
}

impl StepKind {
    /// Whether evaluating the step may end the run. A `pure` extern
    /// declares that the call may be reissued, not that it returns:
    /// `unwrap` is `pure` and panics.
    fn traps(&self) -> bool {
        match self {
            Self::Word(_) => false,
            Self::Arith { op, .. } => op.traps(),
            Self::Call { .. } => true,
        }
    }
}

#[derive(Clone, Copy)]
enum Operand {
    Outside(ValueId),
    Step(ValueId),
}

/// The integer operations a bound may hold. Each is a function of its two
/// words at the width (RFC-0037): `+`, `-` and `*` wrap, and `/` and `%`
/// panic on a zero divisor and on the signed minimum divided by `-1`.
#[derive(Clone, Copy)]
enum Arith {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

impl Arith {
    fn of(op: BinOp) -> Option<Self> {
        match op {
            BinOp::Add => Some(Self::Add),
            BinOp::Sub => Some(Self::Sub),
            BinOp::Mul => Some(Self::Mul),
            BinOp::Div => Some(Self::Div),
            BinOp::Mod => Some(Self::Rem),
            _ => None,
        }
    }

    fn op(self) -> BinOp {
        match self {
            Self::Add => BinOp::Add,
            Self::Sub => BinOp::Sub,
            Self::Mul => BinOp::Mul,
            Self::Div => BinOp::Div,
            Self::Rem => BinOp::Mod,
        }
    }

    fn traps(self) -> bool {
        match self {
            Self::Add | Self::Sub | Self::Mul => false,
            Self::Div | Self::Rem => true,
        }
    }
}

struct Condition {
    counter: ValueId,
    bound: ValueId,
}

struct Counted {
    header: BlockIdx,
    entering: BlockIdx,
    body: Label,
    body_block: BlockIdx,
    body_args: Vec<ValueId>,
    exit: Label,
    exit_args: Vec<ValueId>,
    at: ValueId,
    hi: Bound,
}

struct Recognizer<'a> {
    cfg: &'a CfgBody,
    loop_: &'a Loop,
    invariants: &'a Invariants,
    preds: &'a FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>>,
    literals: &'a FxHashMap<ValueId, Literal>,
}

impl Recognizer<'_> {
    fn counted(&self) -> Option<Counted> {
        let LoopKind::While = self.loop_.kind else {
            return None;
        };
        let natural = &self.loop_.natural;
        let header = natural.header;
        let [entering] = natural.entering[..] else {
            return None;
        };
        let Terminator::JumpIf {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
        } = &self.cfg.blocks[header.0].terminator
        else {
            return None;
        };
        let body_block = self.cfg.label_to_block[then_label];
        let exit_block = self.cfg.label_to_block[else_label];
        if body_block == header
            || !natural.contains(body_block)
            || natural.contains(exit_block)
            || !self.only_from_header(body_block)
            || !self.only_from_header(exit_block)
            || !self.leaves_only_from_header()
        {
            return None;
        }

        let condition = self.condition(header, *cond)?;
        let affine = AffineValues::of(self.cfg, self.loop_, self.invariants);
        let Some(Derivation::Carried { init: at, step }) =
            affine.get(condition.counter).map(|a| &a.derivation)
        else {
            return None;
        };
        if !self.is_one(&step.invariant) {
            return None;
        }
        let hi = self.bound(condition.bound)?;
        if matches!(hi, Bound::Computed { .. }) && !self.only_enters_the_header(entering) {
            return None;
        }

        let ty = self.cfg.val_types[&condition.counter].clone();
        if !matches!(ty, Ty::Int(_))
            || self.cfg.val_types[&condition.bound] != ty
            || self.cfg.val_types[at] != ty
        {
            return None;
        }

        Some(Counted {
            header,
            entering,
            body: *then_label,
            body_block,
            body_args: then_args.clone(),
            exit: *else_label,
            exit_args: else_args.clone(),
            at: *at,
            hi,
        })
    }

    fn bound(&self, value: ValueId) -> Option<Bound> {
        let mut steps = Vec::new();
        Some(match self.evaluate(value, &mut steps)? {
            Operand::Outside(value) => Bound::Outside(value),
            Operand::Step(value) => {
                if !self.moves_ahead_unobserved(&steps) {
                    return None;
                }
                steps.sort_by_key(|step| step.header_position);
                Bound::Computed { steps, value }
            }
        })
    }

    fn evaluate(&self, value: ValueId, steps: &mut Vec<Step>) -> Option<Operand> {
        let natural = &self.loop_.natural;
        if let Some(Invariant::Outside(value)) = self.invariants.at(natural, value) {
            return Some(Operand::Outside(value));
        }
        if steps.iter().any(|step| step.dst == value) {
            return Some(Operand::Step(value));
        }
        let (header_position, inst) = self.cfg.blocks[natural.header.0]
            .insts
            .iter()
            .enumerate()
            .find(|(_, inst)| inst_info::defs(&inst.kind).contains(&value))?;
        let kind = match &inst.kind {
            InstKind::Const { .. } => match self.invariants.at(natural, value)? {
                Invariant::Word(literal) => StepKind::Word(literal),
                Invariant::Outside(_) => return None,
            },
            InstKind::BinOp {
                op, left, right, ..
            } if matches!(self.cfg.val_types[&value], Ty::Int(_)) => StepKind::Arith {
                op: Arith::of(*op)?,
                left: self.evaluate(*left, steps)?,
                right: self.evaluate(*right, steps)?,
            },
            InstKind::FunctionCall {
                callee: callee @ Callee::Extern { .. },
                callee_ty,
                args,
                ..
            } if callee_ty.effect().is_some_and(|effect| effect.is_empty()) => StepKind::Call {
                callee: callee.clone(),
                callee_ty: callee_ty.clone(),
                args: args
                    .iter()
                    .map(|&arg| self.argument(arg, steps))
                    .collect::<Option<_>>()?,
            },
            _ => return None,
        };
        steps.push(Step {
            span: inst.span,
            header_position,
            dst: value,
            kind,
        });
        Some(Operand::Step(value))
    }

    fn argument(&self, arg: ValueId, steps: &mut Vec<Step>) -> Option<Operand> {
        match &self.cfg.val_types[&arg] {
            Ty::Ref(mutability, _) => {
                if *mutability == Mutability::Mut {
                    return None;
                }
                match self.evaluate(arg, steps)? {
                    outside @ Operand::Outside(_) => Some(outside),
                    Operand::Step(_) => None,
                }
            }
            ty if ty.is_scalar() => self.evaluate(arg, steps),
            _ => None,
        }
    }

    fn moves_ahead_unobserved(&self, steps: &[Step]) -> bool {
        let Some(last_trap) = steps
            .iter()
            .filter(|step| step.kind.traps())
            .map(|step| step.header_position)
            .max()
        else {
            return true;
        };
        self.cfg.blocks[self.loop_.natural.header.0].insts[..last_trap]
            .iter()
            .enumerate()
            .filter(|(position, _)| !steps.iter().any(|step| step.header_position == *position))
            .all(|(_, inst)| self.neither_acts_nor_traps(&inst.kind))
    }

    fn neither_acts_nor_traps(&self, kind: &InstKind) -> bool {
        match kind {
            InstKind::Const { .. } | InstKind::Ref { .. } | InstKind::Cast { .. } => true,
            InstKind::BinOp { op, left, .. } => {
                !(matches!(op, BinOp::Div | BinOp::Mod)
                    && matches!(self.cfg.val_types[left], Ty::Int(_)))
            }
            _ => false,
        }
    }

    fn only_enters_the_header(&self, block: BlockIdx) -> bool {
        let header = self.cfg.blocks[self.loop_.natural.header.0].label;
        matches!(
            &self.cfg.blocks[block.0].terminator,
            Terminator::Jump { label, .. } if *label == header
        )
    }

    fn condition(&self, header: BlockIdx, cond: ValueId) -> Option<Condition> {
        self.cfg.blocks[header.0]
            .insts
            .iter()
            .find_map(|inst| match inst.kind {
                InstKind::BinOp {
                    dst,
                    op: BinOp::Lt,
                    left,
                    right,
                } if dst == cond => Some(Condition {
                    counter: left,
                    bound: right,
                }),
                InstKind::BinOp {
                    dst,
                    op: BinOp::Gt,
                    left,
                    right,
                } if dst == cond => Some(Condition {
                    counter: right,
                    bound: left,
                }),
                _ => None,
            })
    }

    fn only_from_header(&self, block: BlockIdx) -> bool {
        match self.preds.get(&block) {
            Some(preds) => preds[..] == [self.loop_.natural.header],
            None => false,
        }
    }

    fn leaves_only_from_header(&self) -> bool {
        let natural = &self.loop_.natural;
        natural
            .blocks()
            .filter(|&block| block != natural.header)
            .all(|block| {
                !matches!(
                    self.cfg.blocks[block.0].terminator,
                    Terminator::Return { .. } | Terminator::Diverge
                ) && self
                    .cfg
                    .successors(block)
                    .iter()
                    .all(|&succ| natural.contains(succ))
            })
    }

    fn is_one(&self, step: &Invariant) -> bool {
        let literal = match step {
            Invariant::Word(literal) => Some(literal),
            Invariant::Outside(value) => self.literals.get(value),
        };
        matches!(
            literal,
            Some(Literal::Int(1) | Literal::IntOf(SuffixedInt { value: 1, .. }))
        )
    }
}

fn const_literals(cfg: &CfgBody) -> FxHashMap<ValueId, Literal> {
    cfg.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| match &inst.kind {
            InstKind::Const { dst, value } => Some((*dst, value.clone())),
            _ => None,
        })
        .collect()
}

impl Counted {
    fn apply(self, cfg: &mut CfgBody) {
        let ty = cfg.val_types[&self.at].clone();
        let hi = match self.hi {
            Bound::Outside(hi) => hi,
            Bound::Computed { steps, value } => {
                let mut renamed: FxHashMap<ValueId, ValueId> = FxHashMap::default();
                for step in steps {
                    let copy = fresh(cfg, &cfg.val_types[&step.dst].clone());
                    let read = |operand: Operand| match operand {
                        Operand::Outside(value) => value,
                        Operand::Step(value) => renamed[&value],
                    };
                    let kind = match step.kind {
                        StepKind::Word(value) => InstKind::Const { dst: copy, value },
                        StepKind::Arith { op, left, right } => InstKind::BinOp {
                            dst: copy,
                            op: op.op(),
                            left: read(left),
                            right: read(right),
                        },
                        StepKind::Call {
                            callee,
                            callee_ty,
                            args,
                        } => InstKind::FunctionCall {
                            dst: copy,
                            callee,
                            callee_ty,
                            args: args.into_iter().map(read).collect(),
                            order: None,
                        },
                    };
                    cfg.blocks[self.entering.0].insts.push(Inst {
                        span: step.span,
                        kind,
                    });
                    renamed.insert(step.dst, copy);
                }
                renamed[&value]
            }
        };
        let counter = fresh(cfg, &ty);
        cfg.blocks[self.body_block.0].params.insert(0, counter);
        cfg.blocks[self.header.0].terminator = Terminator::For {
            source: ForSource::Range { at: self.at, hi },
            body: self.body,
            body_args: self.body_args,
            exit: self.exit,
            exit_args: self.exit_args,
            exit_trip: ExitTrip::Absent,
        };
    }
}

fn fresh(cfg: &mut CfgBody, ty: &Ty) -> ValueId {
    let value = cfg.val_factory.next();
    let previous = cfg.val_types.insert(value, ty.clone());
    assert!(
        previous.is_none(),
        "{value:?} is fresh from the factory and already carried a type"
    );
    cfg.debug.set(value, ValOrigin::Expr);
    value
}
