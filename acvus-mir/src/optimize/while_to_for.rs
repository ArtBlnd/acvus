//! A `while` that is a counted traversal becomes a `for` over a range
//! (RFC-0079).
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
//! before the header runs. A bound written as a literal inside the loop, as
//! `while i < 10` lowers, is therefore written again at the end of the
//! entering block, as `optimize::lsr` writes a literal again above a header
//! (RFC-0056).

use acvus_ast::{BinOp, Literal, Span, SuffixedInt};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::analysis::affine::{AffineValues, Derivation};
use crate::analysis::domtree::DomTree;
use crate::analysis::loops::{Invariant, Invariants, Loop, LoopKind, LoopNest};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{ForSource, Inst, InstKind, Label, ValOrigin, ValueId};
use crate::ty::Ty;

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

enum Hi {
    Outside(ValueId),
    ReemittedWord(Literal),
}

struct Condition {
    counter: ValueId,
    bound: ValueId,
    span: Span,
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
    hi: Hi,
    ty: Ty,
    span: Span,
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
        let hi = match self.invariants.at(natural, condition.bound)? {
            Invariant::Outside(hi) => Hi::Outside(hi),
            Invariant::Word(literal) => Hi::ReemittedWord(literal),
        };

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
            ty,
            span: condition.span,
        })
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
                    span: inst.span,
                }),
                InstKind::BinOp {
                    dst,
                    op: BinOp::Gt,
                    left,
                    right,
                } if dst == cond => Some(Condition {
                    counter: right,
                    bound: left,
                    span: inst.span,
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
        let hi = match self.hi {
            Hi::Outside(hi) => hi,
            Hi::ReemittedWord(value) => {
                let dst = fresh(cfg, &self.ty);
                cfg.blocks[self.entering.0].insts.push(Inst {
                    span: self.span,
                    kind: InstKind::Const { dst, value },
                });
                dst
            }
        };
        let counter = fresh(cfg, &self.ty);
        cfg.blocks[self.body_block.0].params.insert(0, counter);
        cfg.blocks[self.header.0].terminator = Terminator::For {
            source: ForSource::Range { at: self.at, hi },
            body: self.body,
            body_args: self.body_args,
            exit: self.exit,
            exit_args: self.exit_args,
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
