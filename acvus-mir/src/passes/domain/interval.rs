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

pub struct IntervalDomain;
