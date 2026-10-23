//! The interval domain over the integers: `[lo, hi]` with infinite ends,
//! bottom the empty interval; widening at loop headers sends a growing end to
//! infinity, or, with thresholds, to the nearest threshold past it
//! (compilation §2: the fixpoint is bounded). Abstract operations
//! over-approximate the language's (operators §1).

use acvus_ast::Literal;

use crate::ir::{BinOp, UnaryOp};
use crate::passes::domain::{Arithmetic, JoinSemiLattice, Operation, Scalar, ValueDomain, Widen, not_an_operation};
use crate::ty::{CastTy, IntTy};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Interval {
    Empty,
    Range { lo: Bound, hi: Bound },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Bound {
    NegInfinity,
    Finite(i128),
    PosInfinity,
}

impl JoinSemiLattice for Interval {
    fn bottom() -> Self {
        Interval::Empty
    }

    fn join(&mut self, other: &Self) -> bool {
        let next = match (&*self, other) {
            (_, Interval::Empty) => return false,
            (Interval::Empty, _) => other.clone(),
            (Interval::Range { lo, hi }, Interval::Range { lo: other_lo, hi: other_hi }) => {
                Interval::Range { lo: *lo.min(other_lo), hi: *hi.max(other_hi) }
            }
        };
        let grew = *self != next;
        *self = next;
        grew
    }

    fn le(&self, other: &Self) -> bool {
        match (self, other) {
            (Interval::Empty, _) => true,
            (Interval::Range { .. }, Interval::Empty) => false,
            (Interval::Range { lo, hi }, Interval::Range { lo: other_lo, hi: other_hi }) => {
                other_lo <= lo && hi <= other_hi
            }
        }
    }
}

/// The standard interval widening (Cousot and Cousot, 1977): an end that
/// `next` moves outward goes to its infinity, so each end changes at most
/// once after the first value and every ascending chain is finite.
impl Widen for Interval {
    fn widen(&mut self, next: &Self) -> bool {
        self.widen_to(next, Bound::NegInfinity, Bound::PosInfinity)
    }
}

pub struct IntervalDomain;

impl ValueDomain for IntervalDomain {
    type Value = Interval;

    fn top(&self) -> Interval {
        Interval::full()
    }

    fn constant(&self, literal: &Literal, ty: CastTy) -> Interval {
        match Scalar::of(literal, ty) {
            Scalar::Int(n) => Interval::exactly(n),
            Scalar::F64(_) | Scalar::Char(_) | Scalar::Bool(_) => Interval::full(),
        }
    }

    fn binary(&self, op: BinOp, ty: CastTy, left: &Interval, right: &Interval) -> Interval {
        let arithmetic = match Operation::of(op) {
            Operation::Compare(_) => return Interval::full(),
            Operation::Compute(arithmetic) => arithmetic,
        };
        let CastTy::Int(width) = ty else {
            return Interval::full();
        };
        if *left == Interval::Empty || *right == Interval::Empty {
            return Interval::Empty;
        }
        let range = Interval::of_width(width);
        match arithmetic {
            Arithmetic::Add => left.sum(right).meet(&range),
            Arithmetic::Sub => left.difference(right).meet(&range),
            Arithmetic::Mul => left.product(right).meet(&range),
            Arithmetic::Div => left.quotient(right).meet(&range),
            Arithmetic::Mod => left.remainder(right).meet(&range),
            Arithmetic::Shl => {
                let amount = shift_amount(right, width);
                let shifted = amount.bounds().map_or(Interval::Empty, |(lo, hi)| {
                    left.product(&Interval::Range { lo: power_of_two(lo), hi: power_of_two(hi) })
                });
                truncated_to_width(shifted, width)
            }
            Arithmetic::Shr => {
                let amount = shift_amount(right, width);
                left.shifted_right(&amount).meet(&range)
            }
            Arithmetic::BitAnd => left.bit_and(right, width).meet(&range),
            Arithmetic::BitOr | Arithmetic::Xor => left.bit_or(right, width).meet(&range),
            Arithmetic::Min => left.least(right),
            Arithmetic::Max => left.greatest(right),
            Arithmetic::And | Arithmetic::Or => not_an_operation(op, ty),
        }
    }

    fn unary(&self, op: UnaryOp, ty: CastTy, operand: &Interval) -> Interval {
        match (op, ty) {
            (UnaryOp::Neg, CastTy::Int(width)) if width.signed() => {
                Interval::exactly(0).difference(operand).meet(&Interval::of_width(width))
            }
            (UnaryOp::Neg, CastTy::F64) | (UnaryOp::Not, CastTy::Bool) => Interval::full(),
            _ => not_an_operation(op, ty),
        }
    }

    fn cast(&self, from: CastTy, to: CastTy, operand: &Interval) -> Interval {
        assert!(from.admits(to), "`{} as {}` is not a cast the language admits", from.name(), to.name());
        let CastTy::Int(width) = to else {
            return Interval::full();
        };
        match from {
            CastTy::Int(_) => truncated_to_width(operand.clone(), width),
            CastTy::Bool => Interval::finite(0, 1),
            CastTy::Char => truncated_to_width(Interval::finite(0, i128::from(u32::from(char::MAX))), width),
            CastTy::F64 => Interval::of_width(width),
        }
    }
}

/// The result at `width` of a cast or a left shift whose exact result lies
/// in `exact`: both discard the bits past the width, as Rust's `as` and
/// `<<` do, so the result is `exact` where it fits and the whole width
/// where bits may be discarded.
fn truncated_to_width(exact: Interval, width: IntTy) -> Interval {
    let range = Interval::of_width(width);
    if exact.le(&range) { exact } else { range }
}

/// The amounts a shift runs with: the ones below the width, since any
/// other overflows (operators §2).
fn shift_amount(amount: &Interval, width: IntTy) -> Interval {
    amount.meet(&Interval::finite(0, i128::from(width.bits()) - 1))
}

/// `2^amount` for an amount below 64.
fn power_of_two(amount: i128) -> Bound {
    Bound::Finite(1i128 << amount)
}

/// An end of an interval, or the exact value an operation on ends computes,
/// which may lie beyond `i128`. Rounding it back to a `Bound` moves it
/// outward: a lower end down, an upper end up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Exact {
    NegInfinity,
    BelowMin,
    Finite(i128),
    AboveMax,
    PosInfinity,
}

impl Exact {
    fn of(bound: Bound) -> Exact {
        match bound {
            Bound::NegInfinity => Exact::NegInfinity,
            Bound::Finite(n) => Exact::Finite(n),
            Bound::PosInfinity => Exact::PosInfinity,
        }
    }

    fn lower(self) -> Bound {
        match self {
            Exact::NegInfinity | Exact::BelowMin => Bound::NegInfinity,
            Exact::Finite(n) => Bound::Finite(n),
            Exact::AboveMax => Bound::Finite(i128::MAX),
            Exact::PosInfinity => Bound::PosInfinity,
        }
    }

    fn upper(self) -> Bound {
        match self {
            Exact::NegInfinity => Bound::NegInfinity,
            Exact::BelowMin => Bound::Finite(i128::MIN),
            Exact::Finite(n) => Bound::Finite(n),
            Exact::AboveMax | Exact::PosInfinity => Bound::PosInfinity,
        }
    }

    /// Past `i128` on the side of `positive`.
    fn overflowed(positive: bool) -> Exact {
        match positive {
            true => Exact::AboveMax,
            false => Exact::BelowMin,
        }
    }

    fn infinity(positive: bool) -> Exact {
        match positive {
            true => Exact::PosInfinity,
            false => Exact::NegInfinity,
        }
    }
}

fn is_positive(bound: Bound) -> bool {
    bound > Bound::Finite(0)
}

/// `a + b`; `None` where the ends are opposite infinities, which only an
/// end of no interval pairs.
fn sum(a: Bound, b: Bound) -> Option<Exact> {
    match (a, b) {
        (Bound::NegInfinity, Bound::PosInfinity) | (Bound::PosInfinity, Bound::NegInfinity) => None,
        (Bound::NegInfinity, _) | (_, Bound::NegInfinity) => Some(Exact::NegInfinity),
        (Bound::PosInfinity, _) | (_, Bound::PosInfinity) => Some(Exact::PosInfinity),
        (Bound::Finite(a), Bound::Finite(b)) => {
            Some(a.checked_add(b).map_or(Exact::overflowed(a > 0), Exact::Finite))
        }
    }
}

fn negation(a: Bound) -> Exact {
    match a {
        Bound::NegInfinity => Exact::PosInfinity,
        Bound::Finite(n) => n.checked_neg().map_or(Exact::AboveMax, Exact::Finite),
        Bound::PosInfinity => Exact::NegInfinity,
    }
}

/// `a · b`, with `0 · ∞ = 0`: an infinite end stands for values without
/// bound, and each of them times zero is zero.
fn product(a: Bound, b: Bound) -> Exact {
    let positive = is_positive(a) == is_positive(b);
    match (a, b) {
        (Bound::Finite(0), _) | (_, Bound::Finite(0)) => Exact::Finite(0),
        (Bound::Finite(a), Bound::Finite(b)) => a.checked_mul(b).map_or(Exact::overflowed(positive), Exact::Finite),
        (Bound::NegInfinity | Bound::PosInfinity, _) | (_, Bound::NegInfinity | Bound::PosInfinity) => {
            Exact::infinity(positive)
        }
    }
}

/// `a / b` rounded toward zero, as the language divides, for a `b` that is
/// not zero; a finite `a` over an infinite `b` is zero.
fn quotient(a: Bound, b: Bound) -> Exact {
    assert!(b != Bound::Finite(0), "a divisor's part excludes zero");
    let positive = is_positive(a) == is_positive(b);
    match (a, b) {
        (Bound::Finite(0), _) => Exact::Finite(0),
        (Bound::Finite(a), Bound::Finite(b)) => a.checked_div(b).map_or(Exact::overflowed(positive), Exact::Finite),
        (Bound::Finite(_), Bound::NegInfinity | Bound::PosInfinity) => Exact::Finite(0),
        (Bound::NegInfinity | Bound::PosInfinity, _) => Exact::infinity(positive),
    }
}

fn shifted_right(a: Bound, amount: i128) -> Bound {
    match a {
        Bound::Finite(n) => Bound::Finite(n >> amount),
        infinite @ (Bound::NegInfinity | Bound::PosInfinity) => infinite,
    }
}

impl Interval {
    pub(crate) fn full() -> Interval {
        Interval::Range { lo: Bound::NegInfinity, hi: Bound::PosInfinity }
    }

    pub(crate) fn exactly(n: i128) -> Interval {
        Interval::finite(n, n)
    }

    fn finite(lo: i128, hi: i128) -> Interval {
        Interval::between(Bound::Finite(lo), Bound::Finite(hi))
    }

    /// Every value of `width`.
    pub(crate) fn of_width(width: IntTy) -> Interval {
        Interval::finite(width.min(), width.max())
    }

    /// Widening with thresholds (Cousot and Cousot, 1992), the thresholds
    /// being `least`, `greatest` and the infinities: an end that `next`
    /// moves outward goes to the nearest threshold at or past `next`'s end.
    /// Each end changes at most twice after the first value, and at most
    /// once while `next` lies within `[least, greatest]`, so every ascending
    /// chain is finite. Returns whether `self` grew.
    pub(crate) fn widen_to(&mut self, next: &Interval, least: Bound, greatest: Bound) -> bool {
        let (Interval::Range { lo, hi }, Interval::Range { lo: next_lo, hi: next_hi }) = (&*self, next) else {
            return self.join(next);
        };
        let lo = if next_lo >= lo {
            *lo
        } else if *next_lo >= least {
            least
        } else {
            Bound::NegInfinity
        };
        let hi = if next_hi <= hi {
            *hi
        } else if *next_hi <= greatest {
            greatest
        } else {
            Bound::PosInfinity
        };
        let widened = Interval::Range { lo, hi };
        let grew = *self != widened;
        *self = widened;
        grew
    }

    /// `[lo, hi]`, or `Empty` where it holds no integer.
    fn between(lo: Bound, hi: Bound) -> Interval {
        match lo <= hi && lo != Bound::PosInfinity && hi != Bound::NegInfinity {
            true => Interval::Range { lo, hi },
            false => Interval::Empty,
        }
    }

    /// The smallest interval holding every one of `exacts`, each rounded
    /// outward; `Empty` for none.
    fn hull(exacts: impl IntoIterator<Item = Exact>) -> Interval {
        let mut exacts = exacts.into_iter();
        let Some(first) = exacts.next() else {
            return Interval::Empty;
        };
        let (lo, hi) = exacts.fold((first, first), |(lo, hi), exact| (lo.min(exact), hi.max(exact)));
        Interval::between(lo.lower(), hi.upper())
    }

    fn bounds(&self) -> Option<(i128, i128)> {
        match self {
            Interval::Range { lo: Bound::Finite(lo), hi: Bound::Finite(hi) } => Some((*lo, *hi)),
            Interval::Empty | Interval::Range { .. } => None,
        }
    }

    fn ends(&self) -> Option<(Bound, Bound)> {
        match self {
            Interval::Empty => None,
            Interval::Range { lo, hi } => Some((*lo, *hi)),
        }
    }

    /// The four products of an end of `self` with an end of `other`, by
    /// `op`, whose hull holds `op` of every pair where `op` is monotone in
    /// each argument over each interval.
    fn corners(&self, other: &Interval, op: impl Fn(Bound, Bound) -> Exact) -> Interval {
        let (Some((a_lo, a_hi)), Some((b_lo, b_hi))) = (self.ends(), other.ends()) else {
            return Interval::Empty;
        };
        Interval::hull([op(a_lo, b_lo), op(a_lo, b_hi), op(a_hi, b_lo), op(a_hi, b_hi)])
    }

    /// Every integer at least this interval's least.
    pub(crate) fn at_least_lo(&self) -> Interval {
        match self.ends() {
            None => Interval::Empty,
            Some((lo, _)) => Interval::between(lo, Bound::PosInfinity),
        }
    }

    /// Every integer at most this interval's greatest.
    pub(crate) fn at_most_hi(&self) -> Interval {
        match self.ends() {
            None => Interval::Empty,
            Some((_, hi)) => Interval::between(Bound::NegInfinity, hi),
        }
    }

    /// The integers both hold.
    pub(crate) fn meet(&self, other: &Interval) -> Interval {
        let (Some((a_lo, a_hi)), Some((b_lo, b_hi))) = (self.ends(), other.ends()) else {
            return Interval::Empty;
        };
        Interval::between(a_lo.max(b_lo), a_hi.min(b_hi))
    }

    /// `{a + b}` over the integers, no width.
    pub(crate) fn sum(&self, other: &Interval) -> Interval {
        let (Some((a_lo, a_hi)), Some((b_lo, b_hi))) = (self.ends(), other.ends()) else {
            return Interval::Empty;
        };
        let lo = sum(a_lo, b_lo).map_or(Bound::NegInfinity, Exact::lower);
        let hi = sum(a_hi, b_hi).map_or(Bound::PosInfinity, Exact::upper);
        Interval::between(lo, hi)
    }

    /// `{-a}` over the integers.
    fn negated(&self) -> Interval {
        match self.ends() {
            None => Interval::Empty,
            Some((lo, hi)) => Interval::between(negation(hi).lower(), negation(lo).upper()),
        }
    }

    /// `{a - b}` over the integers, no width.
    pub(crate) fn difference(&self, other: &Interval) -> Interval {
        self.sum(&other.negated())
    }

    /// `{a · b}` over the integers, no width.
    pub(crate) fn product(&self, other: &Interval) -> Interval {
        self.corners(other, product)
    }

    /// `{max(a, b)}`.
    pub(crate) fn greatest(&self, other: &Interval) -> Interval {
        let (Some((a_lo, a_hi)), Some((b_lo, b_hi))) = (self.ends(), other.ends()) else {
            return Interval::Empty;
        };
        Interval::between(a_lo.max(b_lo), a_hi.max(b_hi))
    }

    fn least(&self, other: &Interval) -> Interval {
        let (Some((a_lo, a_hi)), Some((b_lo, b_hi))) = (self.ends(), other.ends()) else {
            return Interval::Empty;
        };
        Interval::between(a_lo.min(b_lo), a_hi.min(b_hi))
    }

    /// The divisor's negative and positive parts, zero left out, since a
    /// division by zero traps.
    fn divisor_parts(&self) -> [Interval; 2] {
        [
            self.meet(&Interval::between(Bound::NegInfinity, Bound::Finite(-1))),
            self.meet(&Interval::between(Bound::Finite(1), Bound::PosInfinity)),
        ]
    }

    /// `{a / b}` for every `b` but zero. On each part of the divisor, the
    /// truncated quotient is monotone in each argument, so the corners
    /// bound it.
    fn quotient(&self, divisor: &Interval) -> Interval {
        let mut out = Interval::Empty;
        for part in divisor.divisor_parts() {
            out.join(&self.corners(&part, quotient));
        }
        out
    }

    /// `{a % b}` for every `b` but zero: its sign is the dividend's, and
    /// its magnitude is below the divisor's.
    fn remainder(&self, divisor: &Interval) -> Interval {
        let [negative, positive] = divisor.divisor_parts();
        let mut magnitudes = negative.negated();
        magnitudes.join(&positive);
        let (Some((a_lo, a_hi)), Some((_, largest))) = (self.ends(), magnitudes.ends()) else {
            return Interval::Empty;
        };
        let below = sum(largest, Bound::Finite(-1)).map_or(Bound::PosInfinity, Exact::upper);
        let within = Interval::between(negation(below).lower(), below);
        let signed = Interval::between(a_lo.min(Bound::Finite(0)), a_hi.max(Bound::Finite(0)));
        signed.meet(&within)
    }

    /// `{a >> s}` for amounts `s` below 128: for each amount monotone in
    /// `a`, and for each `a` monotone in the amount.
    fn shifted_right(&self, amount: &Interval) -> Interval {
        let (Some((lo, hi)), Some((s_lo, s_hi))) = (self.ends(), amount.bounds()) else {
            return Interval::Empty;
        };
        Interval::hull(
            [(lo, s_lo), (lo, s_hi), (hi, s_lo), (hi, s_hi)]
                .into_iter()
                .map(|(end, s)| Exact::of(shifted_right(end, s))),
        )
    }

    /// `{a & b}`: at most either operand that is not negative, and at
    /// least zero where one is not.
    fn bit_and(&self, other: &Interval, width: IntTy) -> Interval {
        let (Some((a_lo, a_hi)), Some((b_lo, b_hi))) = (self.ends(), other.ends()) else {
            return Interval::Empty;
        };
        let zero = Bound::Finite(0);
        match (a_lo >= zero, b_lo >= zero) {
            (true, true) => Interval::between(zero, a_hi.min(b_hi)),
            (true, false) => Interval::between(zero, a_hi),
            (false, true) => Interval::between(zero, b_hi),
            (false, false) => Interval::of_width(width),
        }
    }

    /// `{a | b}` and `{a ^ b}`: for operands not negative, below the power
    /// of two above both.
    fn bit_or(&self, other: &Interval, width: IntTy) -> Interval {
        let (Some((a_lo, a_hi)), Some((b_lo, b_hi))) = (self.ends(), other.ends()) else {
            return Interval::Empty;
        };
        match (a_lo, b_lo, a_hi.max(b_hi)) {
            (Bound::Finite(a_lo), Bound::Finite(b_lo), Bound::Finite(hi)) if a_lo >= 0 && b_lo >= 0 => {
                let bits = 128 - hi.leading_zeros();
                let below = match bits {
                    0..=126 => Bound::Finite((1i128 << bits) - 1),
                    _ => Bound::PosInfinity,
                };
                Interval::between(Bound::Finite(0), below)
            }
            _ => Interval::of_width(width),
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(lo: Bound, hi: Bound) -> Interval {
        Interval::between(lo, hi)
    }

    /// D79: a shift's amount keeps its own width: an amount of a wider type
    /// at or past the value's width overflows, which leaves no value.
    #[test]
    fn a_shift_amount_past_the_width_of_a_wider_type_leaves_no_value() {
        let u8_ty = CastTy::Int(IntTy::U8);
        let wide = Interval::exactly(1i128 << 40);
        let one = Interval::exactly(1);
        assert_eq!(IntervalDomain.binary(BinOp::Shl, u8_ty, &one, &wide), Interval::Empty);
        assert_eq!(IntervalDomain.binary(BinOp::Shl, u8_ty, &one, &Interval::exactly(3)), Interval::exactly(8));
    }

    fn finite(lo: i128, hi: i128) -> Interval {
        range(Bound::Finite(lo), Bound::Finite(hi))
    }

    fn samples() -> Vec<Interval> {
        vec![
            Interval::Empty,
            finite(0, 0),
            finite(-3, 5),
            finite(4, 10),
            finite(i128::MIN, i128::MAX),
            range(Bound::NegInfinity, Bound::Finite(0)),
            range(Bound::Finite(1), Bound::PosInfinity),
            range(Bound::NegInfinity, Bound::PosInfinity),
        ]
    }

    fn joined(a: &Interval, b: &Interval) -> Interval {
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
            assert_eq!(joined(&Interval::bottom(), a), *a);
            assert_eq!(joined(a, &Interval::bottom()), *a);
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
    fn widening_covers_the_join() {
        let all = samples();
        for a in &all {
            for b in &all {
                let mut widened = a.clone();
                widened.widen(b);
                assert!(joined(a, b).le(&widened), "{a:?} ∇ {b:?} = {widened:?}");
            }
        }
    }

    #[test]
    fn widening_terminates_on_an_ascending_chain() {
        let mut fact = Interval::bottom();
        let mut rises = 0;
        for n in 0..1000 {
            if fact.widen(&finite(-n, n)) {
                rises += 1;
            }
        }
        assert_eq!(fact, range(Bound::NegInfinity, Bound::PosInfinity));
        assert!(rises <= 3, "each end reaches its infinity in one widening step, rose {rises}");
    }

    #[test]
    fn widening_to_thresholds_covers_the_join() {
        let all = samples();
        for a in &all {
            for b in &all {
                let mut widened = a.clone();
                widened.widen_to(b, Bound::Finite(-128), Bound::Finite(127));
                assert!(joined(a, b).le(&widened), "{a:?} ∇ {b:?} = {widened:?}");
            }
        }
    }

    /// A growing end within the thresholds jumps to the threshold once; one
    /// past them goes on to its infinity, so each end changes at most twice.
    #[test]
    fn widening_to_thresholds_terminates_on_an_ascending_chain() {
        let (least, greatest) = (Bound::Finite(-128), Bound::Finite(127));
        let mut within = Interval::bottom();
        let within_rises = (0..=127).filter(|n| within.widen_to(&finite(-n, *n), least, greatest)).count();
        assert_eq!(within, finite(-128, 127));
        assert!(within_rises <= 3, "each end reaches its threshold in one widening step, rose {within_rises}");

        let mut past = Interval::bottom();
        let past_rises = (0..1000).filter(|n| past.widen_to(&finite(-n, *n), least, greatest)).count();
        assert_eq!(past, Interval::full());
        assert!(past_rises <= 5, "each end reaches its threshold, then its infinity, rose {past_rises}");
    }

    #[test]
    fn widening_keeps_a_stable_end() {
        let mut fact = finite(0, 1);
        fact.widen(&finite(0, 2));
        assert_eq!(fact, range(Bound::Finite(0), Bound::PosInfinity));
    }

    // -- Value operations over-approximate the language's ------------------

    use crate::ir::{BinOp, UnaryOp};
    use crate::ty::{CastTy, IntTy};

    fn contains(interval: &Interval, n: i128) -> bool {
        Interval::exactly(n).le(interval)
    }

    /// The integers of `interval` that are values of `width`.
    fn members(interval: &Interval, width: IntTy) -> Vec<i128> {
        match interval.meet(&Interval::of_width(width)).bounds() {
            Some((lo, hi)) => (lo..=hi).collect(),
            None => Vec::new(),
        }
    }

    /// The language's `a op b` at `width`, by Rust's own operations at a
    /// width of eight bits; `None` is a trap or an overflow.
    fn concrete(op: BinOp, width: IntTy, a: i128, b: i128) -> Option<i128> {
        macro_rules! at {
            ($t:ty) => {{
                let (a, b) = (a as $t, b as $t);
                let amount = u32::try_from(b).ok();
                let result: Option<$t> = match op {
                    BinOp::Add => a.checked_add(b),
                    BinOp::Sub => a.checked_sub(b),
                    BinOp::Mul => a.checked_mul(b),
                    BinOp::Div => a.checked_div(b),
                    BinOp::Mod => a.checked_rem(b),
                    BinOp::Shl => amount.and_then(|amount| a.checked_shl(amount)),
                    BinOp::Shr => amount.and_then(|amount| a.checked_shr(amount)),
                    BinOp::BitAnd => Some(a & b),
                    BinOp::BitOr => Some(a | b),
                    BinOp::Xor => Some(a ^ b),
                    BinOp::Min => Some(a.min(b)),
                    BinOp::Max => Some(a.max(b)),
                    other => panic!("{other:?} is not an integer operation"),
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

    fn operands() -> Vec<Interval> {
        vec![
            Interval::Empty,
            finite(0, 0),
            finite(-1, -1),
            finite(1, 3),
            finite(-3, -1),
            finite(-5, 5),
            finite(0, 7),
            finite(7, 9),
            finite(100, 127),
            finite(200, 255),
            finite(-128, -120),
            range(Bound::NegInfinity, Bound::PosInfinity),
        ]
    }

    #[test]
    fn integer_operations_over_approximate_the_language_at_eight_bits() {
        let ops = [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::Div,
            BinOp::Mod,
            BinOp::Shl,
            BinOp::Shr,
            BinOp::BitAnd,
            BinOp::BitOr,
            BinOp::Xor,
            BinOp::Min,
            BinOp::Max,
        ];
        for width in [IntTy::I8, IntTy::U8] {
            for op in ops {
                for a in &operands() {
                    for b in &operands() {
                        let abstract_result = IntervalDomain.binary(op, CastTy::Int(width), a, b);
                        for x in members(a, width) {
                            for y in members(b, width) {
                                if let Some(value) = concrete(op, width, x, y) {
                                    assert!(
                                        contains(&abstract_result, value),
                                        "{x} {op:?} {y} = {value} at {}, outside {a:?} {op:?} {b:?} = {abstract_result:?}",
                                        width.name()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn negation_over_approximates_the_language_at_eight_bits() {
        for a in &operands() {
            let abstract_result = IntervalDomain.unary(UnaryOp::Neg, CastTy::Int(IntTy::I8), a);
            for x in members(a, IntTy::I8) {
                let x = x as i8;
                if let Some(value) = x.checked_neg() {
                    assert!(contains(&abstract_result, i128::from(value)), "-{x} outside {abstract_result:?}");
                }
            }
        }
    }

    #[test]
    fn an_overflowing_result_is_clamped_to_its_width() {
        let i8_ty = CastTy::Int(IntTy::I8);
        let u64_ty = CastTy::Int(IntTy::U64);
        let max = i128::from(u64::MAX);
        assert_eq!(IntervalDomain.binary(BinOp::Add, i8_ty, &finite(100, 120), &finite(10, 20)), finite(110, 127));
        assert_eq!(IntervalDomain.binary(BinOp::Add, i8_ty, &finite(1, 2), &finite(10, 20)), finite(11, 22));
        assert_eq!(IntervalDomain.binary(BinOp::Add, u64_ty, &finite(max - 1, max), &finite(0, 5)), finite(max - 1, max));
        assert_eq!(
            IntervalDomain.binary(BinOp::Mul, u64_ty, &finite(max, max), &finite(max, max)),
            Interval::Empty,
            "every run overflows, so no value passes"
        );
        assert_eq!(IntervalDomain.binary(BinOp::Div, i8_ty, &finite(-5, 5), &finite(0, 0)), Interval::Empty);
        assert_eq!(IntervalDomain.binary(BinOp::Div, i8_ty, &finite(-128, -128), &finite(-1, -1)), Interval::Empty);
        assert_eq!(IntervalDomain.binary(BinOp::Shl, u64_ty, &finite(1, 1), &finite(64, 70)), Interval::Empty);
        assert_eq!(
            IntervalDomain.binary(BinOp::Shr, u64_ty, &finite(max, max), &finite(63, 63)),
            finite(1, 1)
        );
    }

    #[test]
    fn comparisons_and_non_integers_are_top() {
        let top = IntervalDomain.top();
        let i8_ty = CastTy::Int(IntTy::I8);
        assert_eq!(IntervalDomain.binary(BinOp::Lt, i8_ty, &finite(0, 1), &finite(5, 6)), top);
        assert_eq!(IntervalDomain.binary(BinOp::Add, CastTy::F64, &top, &top), top);
        assert_eq!(IntervalDomain.constant(&acvus_ast::Literal::Float(1.5), CastTy::F64), top);
        assert_eq!(IntervalDomain.constant(&acvus_ast::Literal::Int(-7), i8_ty), finite(-7, -7));
    }

    #[test]
    fn casts_keep_what_fits_the_target_width() {
        let u8_ty = CastTy::Int(IntTy::U8);
        let i64_ty = CastTy::Int(IntTy::I64);
        let top = IntervalDomain.top();
        assert_eq!(IntervalDomain.cast(i64_ty, u8_ty, &finite(3, 5)), finite(3, 5));
        assert_eq!(IntervalDomain.cast(i64_ty, u8_ty, &finite(250, 260)), finite(0, 255));
        assert_eq!(IntervalDomain.cast(CastTy::F64, CastTy::Int(IntTy::I8), &top), finite(-128, 127));
        assert_eq!(IntervalDomain.cast(CastTy::Bool, i64_ty, &top), finite(0, 1));
        assert_eq!(IntervalDomain.cast(i64_ty, CastTy::F64, &finite(3, 5)), top);
    }

}
