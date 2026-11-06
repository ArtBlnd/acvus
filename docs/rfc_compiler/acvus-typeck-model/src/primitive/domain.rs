//! Domains.

use std::collections::BTreeSet;

use super::ty::Scalar;

/// A row of the Domains table, which gives a fallback type (O20).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Family {
    /// Every type. Falls back to `!`.
    Every,
    /// The integers of 8 to 64 bits, and `usize`. Falls back to `i32`.
    Integer,
    /// `f32` and `f64`. Falls back to `f64`.
    Float,
}

/// The domain a literal with no suffix registers: the types that hold its
/// value (O8).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Domain {
    Every,
    Of(BTreeSet<Scalar>),
}
