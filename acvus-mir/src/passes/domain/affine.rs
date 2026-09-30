//! The affine domain (D83): a value is known as an affine combination of
//! symbolic atoms with integer coefficients, `c₁·a₁ + … + cₙ·aₙ + k`, equal
//! to it on every run that defines it; `top` where no combination is known,
//! `bottom` where no run defines it. Two different combinations join to
//! `top`, so the lattice has height three and its widening is its join.
//!
//! It is the one reader of declared Relations (extern-fn §3, D82): each is
//! read as an affine constraint over a call's atoms and solved for the atom
//! a reader asks about.

use std::collections::BTreeMap;

use acvus_ast::Literal;

use crate::decl::declared::{Atom, Comparison, Relation, Term};
use crate::ir::{BinOp, Overflow, UnaryOp, ValueId};
use crate::passes::domain::{Arithmetic, JoinSemiLattice, Operation, Scalar, ValueDomain, Widen, not_an_operation};
use crate::ty::{CastTy, IntTy};

/// An integer a body determines, which the domain does not look into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Symbol {
    /// The integer value.
    Value(ValueId),
    /// The element count of a slice, which no run changes while the slice
    /// exists.
    Len(ValueId),
    /// The steps a `For` has left after the one whose counter is this
    /// value: a range's `hi - 1 - counter`, a slice's `len - 1 - counter`.
    /// It is never negative.
    Remaining(ValueId),
}

/// `Σ coefficient·atom + constant` over the integers, with no zero
/// coefficient, so equal combinations are equal values of this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Combination<A>
where
    A: Ord,
{
    constant: i128,
    terms: BTreeMap<A, i128>,
}

impl<A> Combination<A>
where
    A: Ord + Clone,
{
    pub fn constant(constant: i128) -> Self {
        Combination { constant, terms: BTreeMap::new() }
    }

    pub fn atom(atom: A) -> Self {
        Combination { constant: 0, terms: BTreeMap::from([(atom, 1)]) }
    }

    /// The constant alone, where the combination has no atom.
    pub fn as_constant(&self) -> Option<i128> {
        self.terms.is_empty().then_some(self.constant)
    }

    pub fn constant_term(&self) -> i128 {
        self.constant
    }

    /// Each atom with its coefficient, none of them zero.
    pub fn terms(&self) -> impl Iterator<Item = (&A, i128)> {
        self.terms.iter().map(|(atom, coefficient)| (atom, *coefficient))
    }

    pub fn coefficient(&self, atom: &A) -> i128 {
        self.terms.get(atom).copied().unwrap_or(0)
    }

    /// `self + other`; `None` where a coefficient or the constant leaves
    /// `i128`.
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        let mut sum = self.clone();
        sum.constant = sum.constant.checked_add(other.constant)?;
        for (atom, coefficient) in &other.terms {
            let next = sum.coefficient(atom).checked_add(*coefficient)?;
            match next {
                0 => sum.terms.remove(atom),
                next => sum.terms.insert(atom.clone(), next),
            };
        }
        Some(sum)
    }

    /// `factor · self`; `None` where a coefficient or the constant leaves
    /// `i128`.
    pub fn checked_scale(&self, factor: i128) -> Option<Self> {
        if factor == 0 {
            return Some(Combination::constant(0));
        }
        let terms = self
            .terms
            .iter()
            .map(|(atom, coefficient)| Some((atom.clone(), coefficient.checked_mul(factor)?)))
            .collect::<Option<BTreeMap<A, i128>>>()?;
        Some(Combination { constant: self.constant.checked_mul(factor)?, terms })
    }

    /// `self - other`.
    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        self.checked_add(&other.checked_scale(-1)?)
    }

    /// The combination with each atom replaced by the combination `by`
    /// gives it; `None` where `by` gives none for an atom, or arithmetic
    /// leaves `i128`.
    pub fn substitute<B, F>(&self, mut by: F) -> Option<Combination<B>>
    where
        B: Ord + Clone,
        F: FnMut(&A) -> Option<Combination<B>>,
    {
        let mut out = Combination::constant(self.constant);
        for (atom, coefficient) in &self.terms {
            out = out.checked_add(&by(atom)?.checked_scale(*coefficient)?)?;
        }
        Some(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Affine {
    Bottom,
    Exactly(Combination<Symbol>),
    Top,
}

impl JoinSemiLattice for Affine {
    fn bottom() -> Self {
        Affine::Bottom
    }

    fn join(&mut self, other: &Self) -> bool {
        let next = match (&*self, other) {
            (_, Affine::Bottom) => return false,
            (Affine::Bottom, _) => other.clone(),
            (Affine::Exactly(mine), Affine::Exactly(theirs)) if mine == theirs => return false,
            (Affine::Top, _) => return false,
            (Affine::Exactly(_), Affine::Exactly(_) | Affine::Top) => Affine::Top,
        };
        *self = next;
        true
    }

    fn le(&self, other: &Self) -> bool {
        match (self, other) {
            (Affine::Bottom, _) | (_, Affine::Top) => true,
            (Affine::Exactly(mine), Affine::Exactly(theirs)) => mine == theirs,
            (Affine::Exactly(_) | Affine::Top, _) => false,
        }
    }
}

/// The lattice has height three, so its widening is its join.
impl Widen for Affine {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

impl Affine {
    fn constant(n: i128) -> Affine {
        Affine::Exactly(Combination::constant(n))
    }

    /// An exact integer result at `width`: a trapping operation passes only
    /// a result that fits, so one that never fits leaves no value; a
    /// wrapping one keeps its exact result only where it is a constant that
    /// fits, since a combination may wrap.
    fn at_width(exact: Option<Combination<Symbol>>, overflow: Overflow, width: IntTy) -> Affine {
        let Some(exact) = exact else {
            return Affine::Top;
        };
        match (exact.as_constant(), overflow) {
            (Some(n), Overflow::Trap) if !width.holds(n) => Affine::Bottom,
            (Some(n), Overflow::Wrap) if !width.holds(n) => Affine::Top,
            (Some(_), _) | (None, Overflow::Trap) => Affine::Exactly(exact),
            (None, Overflow::Wrap) => Affine::Top,
        }
    }
}

pub struct AffineDomain;

impl ValueDomain for AffineDomain {
    type Value = Affine;

    fn top(&self) -> Affine {
        Affine::Top
    }

    fn constant(&self, literal: &Literal, ty: CastTy) -> Affine {
        match Scalar::of(literal, ty) {
            Scalar::Int(n) => Affine::constant(n),
            Scalar::F64(_) | Scalar::Char(_) | Scalar::Bool(_) => Affine::Top,
        }
    }

    fn binary(&self, op: BinOp, ty: CastTy, left: &Affine, right: &Affine) -> Affine {
        let arithmetic = match Operation::of(op) {
            Operation::Compare(_) => return Affine::Top,
            Operation::Compute(arithmetic) => arithmetic,
        };
        let CastTy::Int(width) = ty else {
            return Affine::Top;
        };
        let (left, right) = match (left, right) {
            (Affine::Bottom, _) | (_, Affine::Bottom) => return Affine::Bottom,
            (Affine::Top, _) | (_, Affine::Top) => return Affine::Top,
            (Affine::Exactly(left), Affine::Exactly(right)) => (left, right),
        };
        match arithmetic {
            Arithmetic::Add(overflow) => Affine::at_width(left.checked_add(right), overflow, width),
            Arithmetic::Sub(overflow) => Affine::at_width(left.checked_sub(right), overflow, width),
            Arithmetic::Mul(overflow) => {
                let product = match (left.as_constant(), right.as_constant()) {
                    (Some(factor), _) => right.checked_scale(factor),
                    (None, Some(factor)) => left.checked_scale(factor),
                    (None, None) => return Affine::Top,
                };
                Affine::at_width(product, overflow, width)
            }
            Arithmetic::And | Arithmetic::Or => not_an_operation(op, ty),
            Arithmetic::Div
            | Arithmetic::Mod
            | Arithmetic::Shl(_)
            | Arithmetic::Shr(_)
            | Arithmetic::BitAnd
            | Arithmetic::BitOr
            | Arithmetic::Xor
            | Arithmetic::Min
            | Arithmetic::Max => Affine::Top,
        }
    }

    fn unary(&self, op: UnaryOp, ty: CastTy, operand: &Affine) -> Affine {
        match (op, ty) {
            (UnaryOp::Neg(overflow), CastTy::Int(width)) if width.signed() => match operand {
                Affine::Bottom => Affine::Bottom,
                Affine::Top => Affine::Top,
                Affine::Exactly(operand) => Affine::at_width(operand.checked_scale(-1), overflow, width),
            },
            (UnaryOp::Neg(_), CastTy::F64) | (UnaryOp::Not, CastTy::Bool) => Affine::Top,
            _ => not_an_operation(op, ty),
        }
    }

    /// A cast keeps a constant that fits the target width; any other
    /// value may wrap.
    fn cast(&self, from: CastTy, to: CastTy, operand: &Affine) -> Affine {
        assert!(from.admits(to), "`{} as {}` is not a cast the language admits", from.name(), to.name());
        match (from, to, operand) {
            (_, _, Affine::Bottom) => Affine::Bottom,
            (CastTy::Int(_), CastTy::Int(width), Affine::Exactly(exact)) => {
                Affine::at_width(Some(exact.clone()), Overflow::Wrap, width)
            }
            _ => Affine::Top,
        }
    }
}

/// What a declared relation says of one of its atoms: an affine bound over
/// the others.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Limit<A>
where
    A: Ord,
{
    Exactly(Combination<A>),
    AtMost(Combination<A>),
    AtLeast(Combination<A>),
}

impl<A> Limit<A>
where
    A: Ord + Clone,
{
    /// The bound with each atom replaced as `Combination::substitute`
    /// replaces it.
    pub fn substitute<B, F>(&self, by: F) -> Option<Limit<B>>
    where
        B: Ord + Clone,
        F: FnMut(&A) -> Option<Combination<B>>,
    {
        Some(match self {
            Limit::Exactly(at) => Limit::Exactly(at.substitute(by)?),
            Limit::AtMost(at) => Limit::AtMost(at.substitute(by)?),
            Limit::AtLeast(at) => Limit::AtLeast(at.substitute(by)?),
        })
    }
}

/// The term as a combination of its atoms; `None` where arithmetic leaves
/// `i128`.
fn combination(term: &Term) -> Option<Combination<Atom>> {
    Some(match term {
        Term::Const(n) => Combination::constant(*n),
        Term::Atom(atom) => Combination::atom(*atom),
        Term::Add(left, right) => combination(left)?.checked_add(&combination(right)?)?,
        Term::Sub(left, right) => combination(left)?.checked_sub(&combination(right)?)?,
        Term::Scale(factor, inner) => combination(inner)?.checked_scale(*factor)?,
    })
}

/// Every bound `relations` put on `subject`. A relation reads as
/// `left - right` compared with zero; it bounds `subject` where `subject`
/// stands in it with coefficient `1` or `-1`, and `<` is `≤` one less,
/// since its terms are integers. A relation in which `subject` has another
/// coefficient bounds a multiple of it, which is not read.
pub fn bounds_on(relations: &[Relation], subject: Atom) -> Vec<Limit<Atom>> {
    relations
        .iter()
        .filter_map(|Relation { left, comparison, right }| {
            let difference = combination(left)?.checked_sub(&combination(right)?)?;
            let coefficient = difference.coefficient(&subject);
            let others = difference.checked_sub(&Combination::atom(subject).checked_scale(coefficient)?)?;
            let one = Combination::constant(1);
            Some(match (coefficient, comparison) {
                (1, Comparison::Eq) => Limit::Exactly(others.checked_scale(-1)?),
                (1, Comparison::Le) => Limit::AtMost(others.checked_scale(-1)?),
                (1, Comparison::Lt) => Limit::AtMost(others.checked_scale(-1)?.checked_sub(&one)?),
                (-1, Comparison::Eq) => Limit::Exactly(others),
                (-1, Comparison::Le) => Limit::AtLeast(others),
                (-1, Comparison::Lt) => Limit::AtLeast(others.checked_add(&one)?),
                _ => return None,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use acvus_utils::LocalIdOps;

    use super::*;
    use crate::decl::declared::Subject;

    fn value(at: usize) -> Symbol {
        Symbol::Value(ValueId::from_raw(at))
    }

    /// `Σ coefficient·value(atom) + constant`.
    fn exactly(constant: i128, terms: &[(usize, i128)]) -> Affine {
        let mut out = Combination::constant(constant);
        for (atom, coefficient) in terms {
            out = out.checked_add(&Combination::atom(value(*atom)).checked_scale(*coefficient).unwrap()).unwrap();
        }
        Affine::Exactly(out)
    }

    fn samples() -> Vec<Affine> {
        vec![
            Affine::Bottom,
            Affine::Top,
            exactly(0, &[]),
            exactly(5, &[]),
            exactly(0, &[(0, 1)]),
            exactly(3, &[(0, 2)]),
            exactly(-1, &[(1, 1)]),
            exactly(0, &[(0, -1), (1, 1)]),
        ]
    }

    fn joined(a: &Affine, b: &Affine) -> Affine {
        let mut out = a.clone();
        out.join(b);
        out
    }

    #[test]
    fn join_is_associative_commutative_and_idempotent() {
        let all = samples();
        for a in &all {
            assert_eq!(joined(a, a), *a);
            for b in &all {
                assert_eq!(joined(a, b), joined(b, a));
                for c in &all {
                    assert_eq!(joined(&joined(a, b), c), joined(a, &joined(b, c)));
                }
            }
        }
    }

    #[test]
    fn bottom_is_the_unit_of_join() {
        for a in &samples() {
            assert_eq!(joined(&Affine::bottom(), a), *a);
            assert_eq!(joined(a, &Affine::bottom()), *a);
        }
    }

    #[test]
    fn le_agrees_with_join() {
        let all = samples();
        for a in &all {
            for b in &all {
                assert_eq!(a.le(b), joined(a, b) == *b, "{a:?} ≤ {b:?}");
            }
        }
    }

    #[test]
    fn join_reports_growth_exactly() {
        let all = samples();
        for a in &all {
            for b in &all {
                let mut grown = a.clone();
                assert_eq!(grown.join(b), grown != *a);
            }
        }
    }

    #[test]
    fn widening_covers_the_join_and_rises_at_most_twice() {
        let all = samples();
        for a in &all {
            for b in &all {
                let mut widened = a.clone();
                widened.widen(b);
                assert!(joined(a, b).le(&widened), "{a:?} ∇ {b:?} = {widened:?}");
            }
        }
        let mut fact = Affine::bottom();
        let rises = (0..100).filter(|n| fact.widen(&exactly(*n, &[]))).count();
        assert_eq!((fact, rises), (Affine::Top, 2));
    }

    // -- Value operations over-approximate the language's ------------------

    /// The value of `affine` where `value(0)` is `a` and `value(1)` is
    /// `b`; `None` for `top`.
    fn evaluate(affine: &Affine, a: i128, b: i128) -> Option<i128> {
        let Affine::Exactly(combination) = affine else {
            return None;
        };
        let at = |symbol: &Symbol| match symbol {
            Symbol::Value(id) if *id == ValueId::from_raw(0) => a,
            Symbol::Value(id) if *id == ValueId::from_raw(1) => b,
            other => panic!("the samples name value(0) and value(1), not {other:?}"),
        };
        Some(combination.terms().map(|(symbol, coefficient)| coefficient * at(symbol)).sum::<i128>() + combination.constant_term())
    }

    /// The language's `x op y` at `width`, by Rust's own operations at a
    /// width of eight bits; `None` is a trap.
    fn concrete(op: BinOp, width: IntTy, x: i128, y: i128) -> Option<i128> {
        use Overflow::{Trap, Wrap};
        macro_rules! at {
            ($t:ty) => {{
                let (x, y) = (x as $t, y as $t);
                let result: Option<$t> = match op {
                    BinOp::Add(Trap) => x.checked_add(y),
                    BinOp::Add(Wrap) => Some(x.wrapping_add(y)),
                    BinOp::Sub(Trap) => x.checked_sub(y),
                    BinOp::Sub(Wrap) => Some(x.wrapping_sub(y)),
                    BinOp::Mul(Trap) => x.checked_mul(y),
                    BinOp::Mul(Wrap) => Some(x.wrapping_mul(y)),
                    BinOp::Div => x.checked_div(y),
                    BinOp::Mod => x.checked_rem(y),
                    BinOp::Min => Some(x.min(y)),
                    BinOp::Max => Some(x.max(y)),
                    other => panic!("{other:?} is not in the table"),
                };
                result.map(i128::from)
            }};
        }
        match width {
            IntTy::I8 => at!(i8),
            IntTy::U8 => at!(u8),
            other => panic!("the table runs at eight bits, not {}", other.name()),
        }
    }

    /// For every choice of `a` and `b` under which both operands are values
    /// of the width, a result the language computes is the abstract
    /// result's value there, and a result it never computes is `bottom`.
    #[test]
    fn binary_operations_over_approximate_the_language_at_eight_bits() {
        use Overflow::{Trap, Wrap};
        let ops = [
            BinOp::Add(Trap),
            BinOp::Add(Wrap),
            BinOp::Sub(Trap),
            BinOp::Sub(Wrap),
            BinOp::Mul(Trap),
            BinOp::Mul(Wrap),
            BinOp::Div,
            BinOp::Mod,
            BinOp::Min,
            BinOp::Max,
        ];
        let operands: Vec<Affine> = samples().into_iter().filter(|sample| *sample != Affine::Bottom).collect();
        let choices: Vec<i128> = (-130..=260).step_by(7).collect();
        for width in [IntTy::I8, IntTy::U8] {
            for op in ops {
                for left in &operands {
                    for right in &operands {
                        let result = AffineDomain.binary(op, CastTy::Int(width), left, right);
                        for &a in &choices {
                            for &b in &choices {
                                let (Some(x), Some(y)) = (evaluate(left, a, b), evaluate(right, a, b)) else {
                                    continue;
                                };
                                if !width.holds(x) || !width.holds(y) {
                                    continue;
                                }
                                let Some(z) = concrete(op, width, x, y) else {
                                    continue;
                                };
                                assert_ne!(result, Affine::Bottom, "{x} {op:?} {y} = {z} passes");
                                if let Some(abstract_z) = evaluate(&result, a, b) {
                                    assert_eq!(abstract_z, z, "{left:?} {op:?} {right:?} = {result:?} at a = {a}, b = {b}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn negation_and_casts_over_approximate_the_language() {
        let i8_ty = CastTy::Int(IntTy::I8);
        let u8_ty = CastTy::Int(IntTy::U8);
        for operand in samples().iter().filter(|sample| **sample != Affine::Bottom) {
            let negated = AffineDomain.unary(UnaryOp::Neg(Overflow::Trap), i8_ty, operand);
            let cast = AffineDomain.cast(i8_ty, u8_ty, operand);
            for a in -128..=127 {
                for b in [-3, 0, 9] {
                    let Some(x) = evaluate(operand, a, b).filter(|x| IntTy::I8.holds(*x)) else {
                        continue;
                    };
                    if let Some(z) = (x as i8).checked_neg() {
                        assert!(evaluate(&negated, a, b).is_none_or(|abstract_z| abstract_z == i128::from(z)));
                    }
                    let z = i128::from(x as i8 as u8);
                    assert!(evaluate(&cast, a, b).is_none_or(|abstract_z| abstract_z == z), "{operand:?} as u8");
                }
            }
        }
    }

    #[test]
    fn a_trapping_constant_past_its_width_leaves_no_value_and_a_combination_stays_exact() {
        let i8_ty = CastTy::Int(IntTy::I8);
        let add = BinOp::Add(Overflow::Trap);
        assert_eq!(AffineDomain.binary(add, i8_ty, &exactly(100, &[]), &exactly(100, &[])), Affine::Bottom);
        assert_eq!(AffineDomain.binary(BinOp::Add(Overflow::Wrap), i8_ty, &exactly(100, &[]), &exactly(100, &[])), Affine::Top);
        assert_eq!(AffineDomain.binary(add, i8_ty, &exactly(3, &[(0, 2)]), &exactly(-1, &[(1, 1)])), exactly(2, &[(0, 2), (1, 1)]));
        assert_eq!(AffineDomain.binary(BinOp::Sub(Overflow::Trap), i8_ty, &exactly(0, &[(0, 1)]), &exactly(0, &[(0, 1)])), exactly(0, &[]));
        assert_eq!(AffineDomain.binary(BinOp::Mul(Overflow::Trap), i8_ty, &exactly(0, &[(0, 1)]), &exactly(0, &[(1, 1)])), Affine::Top);
    }

    // -- Declared relations --------------------------------------------------

    fn relation(left: Term, comparison: Comparison, right: Term) -> Relation {
        Relation { left, comparison, right }
    }

    fn atom(atom: Atom) -> Term {
        Term::Atom(atom)
    }

    #[test]
    fn a_relation_bounds_an_atom_of_unit_coefficient() {
        let relations = [
            relation(Term::Const(0), Comparison::Le, atom(Atom::Result)),
            relation(atom(Atom::Result), Comparison::Lt, atom(Atom::Arg(0))),
            relation(atom(Atom::Len(Subject::Result)), Comparison::Eq, Term::Add(Box::new(atom(Atom::Arg(1))), Box::new(Term::Const(2)))),
            relation(Term::Scale(2, Box::new(atom(Atom::Result))), Comparison::Le, atom(Atom::Arg(0))),
        ];
        let arg = |at| Combination::atom(Atom::Arg(at));
        assert_eq!(
            bounds_on(&relations, Atom::Result),
            vec![
                Limit::AtLeast(Combination::constant(0)),
                Limit::AtMost(arg(0).checked_sub(&Combination::constant(1)).unwrap()),
            ],
            "2·ret ≤ p0 bounds a multiple of the result, which is not read"
        );
        assert_eq!(
            bounds_on(&relations, Atom::Len(Subject::Result)),
            vec![Limit::Exactly(arg(1).checked_add(&Combination::constant(2)).unwrap())]
        );
    }
}
