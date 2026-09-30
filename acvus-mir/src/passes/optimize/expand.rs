//! The expander (LLVM's SCEVExpander): the one place an evolution becomes
//! values, made with the body's edits at a position the caller names.
//!
//! An invariant ring element is made from its atoms: an atom `Value(v)` is
//! `v`, an atom `Len(s)` is the length of `s`, each cast to the width asked
//! for where its own differs. The terms with a positive coefficient are
//! added first and the others subtracted after them, each term its factors
//! multiplied and then scaled by its coefficient's magnitude.
//!
//! A recurrence `{S', +, D}` of a loop `L` is made only at a position inside
//! `L`, by the recurrence algebra. Where `L` is a `For` whose counter `k`,
//! the canonical induction variable, is `{A, +, 1}` and dominates the
//! position, it is `S' + (k - A)·D`; otherwise, for a header parameter `p` of
//! `L` whose evolution is `{S, +, D}` with the same step, it is
//! `p + (S' - S)`. The starts are made as evolutions in turn.
//!
//! Every value made is one the run as written did not compute, so each,
//! every partial result included, is made only where it is proven within the
//! width asked for (undefined behaviour §2): by the interval domain, or, for
//! a value that is a recurrence `{S, +, D}` of a loop with a count `n`, exact
//! or at most, by its own range there, `S + [0, n - 1]·D` where the loop's
//! body dominates the position and `S + [0, n]·D` elsewhere inside the loop
//! (ScalarEvolution's range of an add recurrence).
//!
//! An expansion made before, of the same evolution in the same type, whose
//! value dominates the new position is reused (SCEVExpander's inserted
//! expressions), so one value serves every use it reaches. The
//! expansion answers none where a value it needs does not dominate the
//! position, where neither a counter nor a same-step parameter makes a
//! recurrence, where a value is not proven within the width, or where the
//! evolution is unknown; it then edits nothing.
//!
//! Cost: O(the evolution's terms · MAX_DEGREE · d) planned and made values
//! for a loop nesting depth d, and O(the header's parameters) per recurrence
//! to find a same-step parameter.

use std::cell::RefCell;

use acvus_ast::{Literal, Span};
use rustc_hash::FxHashMap;

use crate::ir::cfg::edit::{DominanceFacts, Editor, Position, ReplacesValues};
use crate::ir::cfg::{Body, Definition, Op, Terminator};
use crate::ir::{BinOp, ValueId};
use crate::passes::analysis::{Dominators, Evolution, Evolutions, LoopTree, Point, Recurrence, SparseFacts, TripCount};
use crate::passes::domain::{Interval, IntervalDomain, JoinSemiLattice, Polynomial, Symbol};
use crate::ty::{CastTy, IntTy, Ty, TyKind};

/// What an expansion reads, and what it has made. `dominators` is the
/// manager's for the body the editor holds, computed before the edits made
/// since: an inserted instruction, a replaced use, a replaced loop head and a
/// removed parameter keep every edge, so it still answers for the body.
/// `evolutions` and `intervals` hold the facts of every value that existed
/// when they were computed, which are the values an expansion reads.
pub(in crate::passes) struct Expander<'f> {
    evolutions: &'f Evolutions,
    loops: &'f LoopTree,
    intervals: &'f SparseFacts<IntervalDomain>,
    dominators: &'f Dominators,
    /// `u64`, the type of a slice's length.
    len_ty: Ty,
    /// Every value made so far, by what it is: an expansion whose value
    /// dominates the new position is reused (SCEVExpander's inserted
    /// expressions), so one value serves every use it reaches.
    made: RefCell<FxHashMap<Expansion, Vec<ValueId>>>,
}

/// An evolution made in a type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Expansion {
    evolution: Evolution,
    ty: Ty,
}

/// A value an expansion reads as it is, cast to the width where its own
/// differs.
#[derive(Debug, Clone, Copy)]
enum Leaf {
    Value(ValueId),
    Len(ValueId),
}

/// How a value is made.
enum Plan {
    Leaf(Leaf),
    Constant(i128),
    Binary(BinOp, Box<Planned>, Box<Planned>),
}

/// A plan and the integers its value may be, proven within the width.
struct Planned {
    plan: Plan,
    interval: Interval,
}

impl Planned {
    fn is_constant(&self, value: i128) -> bool {
        matches!(self.plan, Plan::Constant(held) if held == value)
    }
}

impl<'f> Expander<'f> {
    pub fn new(
        evolutions: &'f Evolutions,
        loops: &'f LoopTree,
        intervals: &'f SparseFacts<IntervalDomain>,
        dominators: &'f Dominators,
        len_ty: Ty,
    ) -> Expander<'f> {
        Expander { evolutions, loops, intervals, dominators, len_ty, made: RefCell::default() }
    }

    /// `evolution`'s value at `at`, made in `ty`, an integer type, or one
    /// made before that dominates `at`; none where it cannot be made there,
    /// as the module doc states.
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
        let planner = Planner { expander: self, body: editor.body(), range: Interval::of_width(width), at };
        let expansion = Expansion { evolution: evolution.clone(), ty };
        let reused = self.made.borrow().get(&expansion).and_then(|made| made.iter().copied().find(|value| planner.dominates(*value)));
        if reused.is_some() {
            return reused;
        }
        let planned = planner.evolution(evolution)?;
        let value = Made { editor, ty, width, at, span, len_ty: self.len_ty }.make(planned);
        self.made.borrow_mut().entry(expansion).or_default().push(value);
        Some(value)
    }
}

/// The planning of one expansion.
struct Planner<'p> {
    expander: &'p Expander<'p>,
    body: &'p Body,
    range: Interval,
    at: Position,
}

impl Planner<'_> {
    fn evolution(&self, evolution: &Evolution) -> Option<Planned> {
        match evolution {
            Evolution::Invariant(polynomial) => self.polynomial(polynomial),
            Evolution::Recurrence(recurrence) => self.recurrence(recurrence),
            Evolution::Unknown => None,
        }
    }

    fn polynomial(&self, polynomial: &Polynomial<Symbol>) -> Option<Planned> {
        let mut positive = Vec::new();
        let mut negative = Vec::new();
        for (monomial, coefficient) in polynomial.terms() {
            let mut product: Option<Planned> = None;
            for (atom, power) in monomial.factors() {
                let leaf = match atom {
                    Symbol::Value(value) => Leaf::Value(*value),
                    Symbol::Len(slice) => Leaf::Len(*slice),
                    Symbol::Remaining(_) => return None,
                };
                for _ in 0..power {
                    let factor = self.leaf(leaf)?;
                    product = Some(match product {
                        Some(product) => self.binary(BinOp::Mul, product, factor)?,
                        None => factor,
                    });
                }
            }
            let magnitude = self.constant(coefficient.checked_abs()?)?;
            let term = match product {
                Some(product) => self.product(product, magnitude)?,
                None => magnitude,
            };
            match coefficient < 0 {
                true => negative.push(term),
                false => positive.push(term),
            }
        }
        let mut sum: Option<Planned> = None;
        for term in positive {
            sum = Some(match sum {
                Some(sum) => self.binary(BinOp::Add, sum, term)?,
                None => term,
            });
        }
        let mut sum = match sum {
            Some(sum) => sum,
            None => self.constant(0)?,
        };
        for term in negative {
            sum = self.binary(BinOp::Sub, sum, term)?;
        }
        Some(sum)
    }

    /// `{S', +, D}` of its loop, from the counter where it exists, else
    /// from a same-step header parameter.
    fn recurrence(&self, recurrence: &Recurrence) -> Option<Planned> {
        if !self.expander.loops.contains(recurrence.over, point(self.at).block(self.body)) {
            return None;
        }
        self.from_counter(recurrence).or_else(|| self.from_same_step(recurrence))
    }

    /// `S' + (k - A)·D`, `k` the counter of the `For` heading the loop,
    /// whose evolution is `{A, +, 1}`.
    fn from_counter(&self, recurrence: &Recurrence) -> Option<Planned> {
        let header = self.expander.loops.header(recurrence.over);
        let Terminator::For { source, body, .. } = self.body.block(header).terminator() else {
            return None;
        };
        let counter = self.body.block(*body).params()[source.counter_param()];
        let Evolution::Recurrence(counted) = self.expander.evolutions.of(counter) else {
            return None;
        };
        if counted.over != recurrence.over || counted.step.as_constant() != Some(1) {
            return None;
        }
        let k = self.leaf(Leaf::Value(counter))?;
        let position = match counted.start == Evolution::Invariant(Polynomial::zero()) {
            true => k,
            false => self.binary(BinOp::Sub, k, self.evolution(&counted.start)?)?,
        };
        let risen = self.product(position, self.polynomial(&recurrence.step)?)?;
        self.sum(self.evolution(&recurrence.start)?, risen)
    }

    /// `p + (S' - S)` for the first header parameter `p` of the loop whose
    /// evolution `{S, +, D}` has the recurrence's step and makes it.
    fn from_same_step(&self, recurrence: &Recurrence) -> Option<Planned> {
        let header = self.expander.loops.header(recurrence.over);
        self.body.block(header).params().iter().find_map(|param| {
            let Evolution::Recurrence(own) = self.expander.evolutions.of(*param) else {
                return None;
            };
            if own.over != recurrence.over || own.step != recurrence.step {
                return None;
            }
            let offset = self.expander.evolutions.difference(&recurrence.start, &own.start);
            self.sum(self.leaf(Leaf::Value(*param))?, self.evolution(&offset)?)
        })
    }

    fn leaf(&self, leaf: Leaf) -> Option<Planned> {
        let (Leaf::Value(read) | Leaf::Len(read)) = leaf;
        if !self.dominates(read) {
            return None;
        }
        let interval = match leaf {
            Leaf::Value(value) => {
                let domain = self.expander.intervals.of(value).clone();
                match self.expander.evolutions.of(value) {
                    Evolution::Recurrence(recurrence) => self.recurrence_range(recurrence).map_or(domain.clone(), |range| range.meet(&domain)),
                    Evolution::Invariant(_) | Evolution::Unknown => domain,
                }
            }
            Leaf::Len(_) => Interval::of_width(IntTy::U64),
        };
        self.within(Plan::Leaf(leaf), interval)
    }

    /// The range of `{S, +, D}` over a loop with a count `n`, exact or at
    /// most, at the position, inside the loop: `S + [0, n - 1]·D` where the
    /// loop's body dominates the position, `S + [0, n]·D` elsewhere in it,
    /// the header's last visit included; the start and step from their
    /// intervals. None where the position is outside the loop or no count is
    /// named.
    fn recurrence_range(&self, recurrence: &Recurrence) -> Option<Interval> {
        let block = point(self.at).block(self.body);
        let loops = self.expander.loops;
        if !loops.contains(recurrence.over, block) {
            return None;
        }
        let (TripCount::Exact(count) | TripCount::AtMost(count)) = self.expander.evolutions.trip_count(recurrence.over)?;
        let count = self.evolution_interval(count)?;
        let loop_body = self.body.block(loops.header(recurrence.over)).terminator().as_loop()?.body;
        let last = match self.expander.dominators.dominates(loop_body, block) {
            true => count.difference(&Interval::exactly(1)),
            false => count,
        };
        let iterations = Interval::exactly(0).at_least_lo().meet(&last.at_most_hi());
        let start = self.evolution_interval(&recurrence.start)?;
        Some(start.sum(&iterations.product(&self.polynomial_interval(&recurrence.step))))
    }

    /// The integers `evolution` may be at the position.
    fn evolution_interval(&self, evolution: &Evolution) -> Option<Interval> {
        match evolution {
            Evolution::Invariant(polynomial) => Some(self.polynomial_interval(polynomial)),
            Evolution::Recurrence(recurrence) => self.recurrence_range(recurrence),
            Evolution::Unknown => None,
        }
    }

    fn polynomial_interval(&self, polynomial: &Polynomial<Symbol>) -> Interval {
        polynomial.terms().fold(Interval::exactly(0), |sum, (monomial, coefficient)| {
            let term = monomial.factors().fold(Interval::exactly(coefficient), |product, (atom, power)| {
                let atom = match atom {
                    Symbol::Value(value) => self.expander.intervals.of(*value).clone(),
                    Symbol::Len(_) => Interval::of_width(IntTy::U64),
                    Symbol::Remaining(_) => Interval::exactly(0).at_least_lo(),
                };
                (0..power).fold(product, |product, _| product.product(&atom))
            });
            sum.sum(&term)
        })
    }

    fn constant(&self, value: i128) -> Option<Planned> {
        self.within(Plan::Constant(value), Interval::exactly(value))
    }

    fn binary(&self, op: BinOp, left: Planned, right: Planned) -> Option<Planned> {
        let interval = match op {
            BinOp::Add => left.interval.sum(&right.interval),
            BinOp::Sub => left.interval.difference(&right.interval),
            BinOp::Mul => left.interval.product(&right.interval),
            other => panic!("an expansion adds, subtracts and multiplies, not {other:?}"),
        };
        self.within(Plan::Binary(op, Box::new(left), Box::new(right)), interval)
    }

    /// `left + right`, a zero operand left out.
    fn sum(&self, left: Planned, right: Planned) -> Option<Planned> {
        match (left.is_constant(0), right.is_constant(0)) {
            (true, _) => Some(right),
            (false, true) => Some(left),
            (false, false) => self.binary(BinOp::Add, left, right),
        }
    }

    /// `left · right`, a unit operand left out.
    fn product(&self, left: Planned, right: Planned) -> Option<Planned> {
        match (left.is_constant(1), right.is_constant(1)) {
            (true, _) => Some(right),
            (false, true) => Some(left),
            (false, false) => self.binary(BinOp::Mul, left, right),
        }
    }

    fn within(&self, plan: Plan, interval: Interval) -> Option<Planned> {
        interval.le(&self.range).then_some(Planned { plan, interval })
    }

    /// Whether `value`'s definition dominates the position.
    fn dominates(&self, value: ValueId) -> bool {
        let defined = match self.body.definition(value) {
            Definition::Param(_) | Definition::Capture(_) | Definition::OrderParam | Definition::Slot(_) => return true,
            Definition::BlockParam(block, _) => Point::start(self.body, block),
            Definition::Inst(inst) => Point::after(self.body, inst),
        };
        self.expander.dominators.point_dominates(self.body, defined, point(self.at))
    }
}

/// The making of a plan at one position.
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
    fn make(&mut self, planned: Planned) -> ValueId {
        match planned.plan {
            Plan::Leaf(leaf) => self.leaf(leaf),
            Plan::Constant(value) => self.insert(Op::Const { value: Literal::Int(value) }),
            Plan::Binary(op, left, right) => {
                let (left, right) = (self.make(*left), self.make(*right));
                self.insert(Op::BinOp { op, left, right })
            }
        }
    }

    fn leaf(&mut self, leaf: Leaf) -> ValueId {
        let value = match leaf {
            Leaf::Value(value) => value,
            Leaf::Len(slice) => self.editor.insert(self.at, Op::SliceLen { src: slice }, self.len_ty, self.span),
        };
        match self.editor.body().ty_of(value) == self.ty {
            true => value,
            false => self.insert(Op::Cast { src: value, to: CastTy::Int(self.width) }),
        }
    }

    fn insert(&mut self, op: Op) -> ValueId {
        self.editor.insert(self.at, op, self.ty, self.span)
    }
}

fn point(at: Position) -> Point {
    match at {
        Position::Before(inst) => Point::Before(inst.id()),
        Position::End(block) => Point::End(block),
    }
}
