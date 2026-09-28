//! The interval domain over the integers: `[lo, hi]` with infinite ends,
//! bottom the empty interval; widening at loop headers sends a growing end to
//! infinity (compilation §2: the fixpoint is bounded). Abstract operations
//! over-approximate the language's (operators §1).

use crate::passes::domain::{JoinSemiLattice, ValueDomain, Widen};

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
/// twice after the first value and every ascending chain is finite.
impl Widen for Interval {
    fn widen(&mut self, next: &Self) -> bool {
        let (Interval::Range { lo, hi }, Interval::Range { lo: next_lo, hi: next_hi }) = (&*self, next) else {
            return self.join(next);
        };
        let widened = Interval::Range {
            lo: if next_lo < lo { Bound::NegInfinity } else { *lo },
            hi: if next_hi > hi { Bound::PosInfinity } else { *hi },
        };
        let grew = *self != widened;
        *self = widened;
        grew
    }
}

pub struct IntervalDomain;

#[cfg(test)]
mod tests {
    use super::*;

    /// `[lo, hi]`, or `Empty` where it holds no integer.
    fn range(lo: Bound, hi: Bound) -> Interval {
        match lo <= hi && lo != Bound::PosInfinity && hi != Bound::NegInfinity {
            true => Interval::Range { lo, hi },
            false => Interval::Empty,
        }
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
    fn widening_keeps_a_stable_end() {
        let mut fact = finite(0, 1);
        fact.widen(&finite(0, 2));
        assert_eq!(fact, range(Bound::Finite(0), Bound::PosInfinity));
    }
}
