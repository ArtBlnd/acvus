//! The expander (LLVM's SCEVExpander): the one place an evolution becomes
//! values, made with the body's edits at a position the caller names.
//!
//! An invariant ring element is made from its atoms: an atom `Value(v)` is
//! `v`, an atom `Len(s)` is the length of `s`, each cast to the width asked
//! for where its own differs. A recurrence is made only at a position inside
//! its loop, as that loop's header parameter whose evolution it is: there the
//! parameter holds the recurrence's value on the current iteration. An
//! element is made term by term, each term its factors multiplied and then
//! scaled by its coefficient's magnitude, the terms with a positive
//! coefficient added first and the others subtracted after them.
//!
//! Every value made is one the run as written did not compute, so each,
//! every partial product and sum included, is made only where the interval
//! domain proves it within the width asked for (undefined behaviour §2). The
//! expansion answers none where a value it needs does not dominate the
//! position, where no header parameter is the recurrence, where a value is
//! not proven within the width, or where the evolution is unknown; it then
//! edits nothing.
//!
//! Cost: O(the element's terms · MAX_DEGREE) planned and made values, and
//! O(the header's parameters) to find a recurrence's parameter.

use acvus_ast::{Literal, Span};

use crate::ir::ValueId;
use crate::ir::cfg::edit::{DominanceFacts, Editor, Position, ReplacesValues};
use crate::ir::cfg::{Body, Definition, Op};
use crate::passes::analysis::{Dominators, Evolution, Evolutions, LoopTree, Point, SparseFacts};
use crate::passes::domain::{Interval, IntervalDomain, JoinSemiLattice, Polynomial, Symbol};
use crate::ty::{CastTy, IntTy, Ty, TyKind};

/// What an expansion reads. `dominators` is the manager's for the body the
/// editor holds, computed before the edits made since: an inserted
/// instruction and a replaced loop head keep every edge, so it still
/// answers for the body. `intervals` holds the facts of every value that
/// existed when it was computed, which are the values an expansion reads.
pub(in crate::passes) struct Expander<'f> {
    pub evolutions: &'f Evolutions,
    pub loops: &'f LoopTree,
    pub intervals: &'f SparseFacts<IntervalDomain>,
    pub dominators: &'f Dominators,
    /// `u64`, the type of a slice's length.
    pub len_ty: Ty,
}

/// A value an expansion reads as it is.
#[derive(Debug, Clone, Copy)]
enum Leaf {
    Value(ValueId),
    Len(ValueId),
}

/// One term: its factors, multiplied in order, scaled by `magnitude`.
struct Term {
    factors: Vec<Leaf>,
    magnitude: i128,
    negative: bool,
}

impl Expander<'_> {
    /// `evolution`'s value at `at`, made in `ty`, an integer type; none where
    /// it cannot be made there, as the module doc states.
    ///
    /// # Panics
    /// `ty` is not an integer type.
    pub fn expand<'m, S>(&self, editor: &mut Editor<'m, S>, evolution: &Evolution, ty: Ty, at: Position, span: Span) -> Option<ValueId>
    where
        S: ReplacesValues,
        S::Needs<'m>: DominanceFacts,
    {
        let TyKind::Int(width) = *editor.types().kind(ty) else {
            panic!("an evolution is made in an integer type, not {:?}", editor.types().kind(ty))
        };
        let terms = self.plan(editor.body(), evolution, width, at)?;
        Some(Made { editor, ty, width, at, span, len_ty: self.len_ty }.terms(&terms))
    }

    /// The terms `evolution` is made of at `at`, each value they make proven
    /// within `width`.
    fn plan(&self, body: &Body, evolution: &Evolution, width: IntTy, at: Position) -> Option<Vec<Term>> {
        let terms = match evolution {
            Evolution::Invariant(polynomial) => self.polynomial_terms(body, polynomial, at)?,
            Evolution::Recurrence(_) => {
                let param = self.recurrence_param(body, evolution, at)?;
                vec![Term { factors: vec![Leaf::Value(param)], magnitude: 1, negative: false }]
            }
            Evolution::Unknown => return None,
        };
        self.proven_within(&terms, width).then_some(terms)
    }

    fn polynomial_terms(&self, body: &Body, polynomial: &Polynomial<Symbol>, at: Position) -> Option<Vec<Term>> {
        let mut terms = Vec::new();
        for (monomial, coefficient) in polynomial.terms() {
            let mut factors = Vec::new();
            for (atom, power) in monomial.factors() {
                let leaf = match atom {
                    Symbol::Value(value) => Leaf::Value(*value),
                    Symbol::Len(slice) => Leaf::Len(*slice),
                    Symbol::Remaining(_) => return None,
                };
                let (Leaf::Value(read) | Leaf::Len(read)) = leaf;
                if !self.dominates(body, read, at) {
                    return None;
                }
                factors.extend(std::iter::repeat_n(leaf, power as usize));
            }
            terms.push(Term { factors, magnitude: coefficient.checked_abs()?, negative: coefficient < 0 });
        }
        Some(terms)
    }

    /// The header parameter of the recurrence's loop whose evolution is
    /// `evolution`, where `at` is inside that loop.
    fn recurrence_param(&self, body: &Body, evolution: &Evolution, at: Position) -> Option<ValueId> {
        let Evolution::Recurrence(recurrence) = evolution else {
            return None;
        };
        let at_block = point(at).block(body);
        if !self.loops.contains(recurrence.over, at_block) {
            return None;
        }
        let header = self.loops.header(recurrence.over);
        body.block(header).params().iter().copied().find(|param| self.evolutions.of(*param) == evolution)
    }

    /// Whether `value`'s definition dominates `at`.
    fn dominates(&self, body: &Body, value: ValueId, at: Position) -> bool {
        let defined = match body.definition(value) {
            Definition::Param(_) | Definition::Capture(_) | Definition::OrderParam | Definition::Slot(_) => return true,
            Definition::BlockParam(block, _) => Point::start(body, block),
            Definition::Inst(inst) => Point::after(body, inst),
        };
        self.dominators.point_dominates(body, defined, point(at))
    }

    /// Whether every value `terms` make, in the order `Made` makes them,
    /// lies within `width`.
    fn proven_within(&self, terms: &[Term], width: IntTy) -> bool {
        let range = Interval::of_width(width);
        let within = |interval: &Interval| interval.le(&range);
        let mut sum = Interval::exactly(0);
        for term in terms.iter().filter(|term| !term.negative) {
            let Some(made) = self.term_interval(term, &within) else {
                return false;
            };
            sum = sum.sum(&made);
            if !within(&sum) {
                return false;
            }
        }
        for term in terms.iter().filter(|term| term.negative) {
            let Some(made) = self.term_interval(term, &within) else {
                return false;
            };
            sum = sum.difference(&made);
            if !within(&sum) {
                return false;
            }
        }
        true
    }

    /// The interval of the value `term` makes, where it and every value
    /// made on the way lie `within`.
    fn term_interval<W>(&self, term: &Term, within: &W) -> Option<Interval>
    where
        W: Fn(&Interval) -> bool,
    {
        let mut product: Option<Interval> = None;
        for leaf in &term.factors {
            let factor = self.leaf_interval(*leaf);
            let next = product.map_or(factor.clone(), |product| product.product(&factor));
            if !within(&factor) || !within(&next) {
                return None;
            }
            product = Some(next);
        }
        let magnitude = Interval::exactly(term.magnitude);
        let scaled = product.map_or(magnitude.clone(), |product| product.product(&magnitude));
        (within(&magnitude) && within(&scaled)).then_some(scaled)
    }

    /// The integers a leaf may be: a value's interval, and a length's `u64`
    /// range.
    fn leaf_interval(&self, leaf: Leaf) -> Interval {
        match leaf {
            Leaf::Value(value) => self.intervals.of(value).clone(),
            Leaf::Len(_) => Interval::of_width(IntTy::U64),
        }
    }
}

/// The making of planned terms at one position.
struct Made<'e, 'm, S>
where
    S: ReplacesValues,
{
    editor: &'e mut Editor<'m, S>,
    ty: Ty,
    width: IntTy,
    at: Position,
    span: Span,
    len_ty: Ty,
}

impl<'m, S> Made<'_, 'm, S>
where
    S: ReplacesValues,
    S::Needs<'m>: DominanceFacts,
{
    /// The positive terms summed, then the negative ones subtracted; zero
    /// where there is no positive term.
    fn terms(&mut self, terms: &[Term]) -> ValueId {
        let mut sum: Option<ValueId> = None;
        for term in terms.iter().filter(|term| !term.negative) {
            let made = self.term(term);
            sum = Some(match sum {
                Some(sum) => self.binary(crate::ir::BinOp::Add, sum, made),
                None => made,
            });
        }
        let mut sum = sum.unwrap_or_else(|| self.constant(0));
        for term in terms.iter().filter(|term| term.negative) {
            let made = self.term(term);
            sum = self.binary(crate::ir::BinOp::Sub, sum, made);
        }
        sum
    }

    fn term(&mut self, term: &Term) -> ValueId {
        let mut product: Option<ValueId> = None;
        for leaf in &term.factors {
            let factor = self.leaf(*leaf);
            product = Some(match product {
                Some(product) => self.binary(crate::ir::BinOp::Mul, product, factor),
                None => factor,
            });
        }
        match (product, term.magnitude) {
            (Some(product), 1) => product,
            (Some(product), magnitude) => {
                let scale = self.constant(magnitude);
                self.binary(crate::ir::BinOp::Mul, product, scale)
            }
            (None, magnitude) => self.constant(magnitude),
        }
    }

    fn leaf(&mut self, leaf: Leaf) -> ValueId {
        let value = match leaf {
            Leaf::Value(value) => value,
            Leaf::Len(slice) => self.editor.insert(self.at, Op::SliceLen { src: slice }, self.len_ty, self.span),
        };
        match self.editor.body().ty_of(value) == self.ty {
            true => value,
            false => self.editor.insert(self.at, Op::Cast { src: value, to: CastTy::Int(self.width) }, self.ty, self.span),
        }
    }

    fn constant(&mut self, value: i128) -> ValueId {
        self.editor.insert(self.at, Op::Const { value: Literal::Int(value) }, self.ty, self.span)
    }

    fn binary(&mut self, op: crate::ir::BinOp, left: ValueId, right: ValueId) -> ValueId {
        self.editor.insert(self.at, Op::BinOp { op, left, right }, self.ty, self.span)
    }
}

fn point(at: Position) -> Point {
    match at {
        Position::Before(inst) => Point::Before(inst.id()),
        Position::End(block) => Point::End(block),
    }
}
