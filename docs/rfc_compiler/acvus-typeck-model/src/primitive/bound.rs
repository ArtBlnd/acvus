//! Bounds: what relates two classes and leaves them two.
//!
//! A union makes two classes one, at every depth. A bound keeps them apart at
//! the top layer, where what it means differs by its kind, and every part below
//! that layer is one type with the corresponding part. It gives heads while it
//! waits: a class may take its head from either end of a bound.

use std::fmt::Debug;

use super::class::ClassKind;

/// A bound as a store of the kind `K` sees it.
///
/// These methods are all a store reads of a bound, so what a bound means stays
/// with whoever defines it.
pub trait BoundOf<K>: Clone + Ord + Debug
where
    K: ClassKind,
{
    fn lower(&self) -> K::Id;
    fn upper(&self) -> K::Id;
}

/// A bound of the kind `K` written as one of the kind `J`, as a copy of a
/// template needs it.
pub trait BoundInto<K, J>
where
    K: ClassKind,
    J: ClassKind,
{
    fn translate<F>(self, f: F) -> J::Bound
    where
        F: FnMut(K::Id) -> J::Id;
}

/// The bounds of the language. They differ only at the top layer.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Bound<K>
where
    K: ClassKind,
{
    /// `upper`'s head is the join of the heads that enter it (O1, O2, O6). The
    /// one operation `lower` may take to reach it is the reborrow. Classes
    /// that enter one another are one class.
    Enter { lower: K::Id, upper: K::Id },
    /// `lower` is passed to `upper`: by subtyping the language gives, or else
    /// through declared functions. It decides neither head.
    Change { lower: K::Id, upper: K::Id },
    /// The two heads are equal (O9). It decides neither head and makes no
    /// change, so it can be judged one layer at a time.
    Equal { lower: K::Id, upper: K::Id },
}

impl<K> BoundOf<K> for Bound<K>
where
    K: ClassKind,
{
    fn lower(&self) -> K::Id {
        match self {
            Bound::Enter { lower, .. }
            | Bound::Change { lower, .. }
            | Bound::Equal { lower, .. } => *lower,
        }
    }

    fn upper(&self) -> K::Id {
        match self {
            Bound::Enter { upper, .. }
            | Bound::Change { upper, .. }
            | Bound::Equal { upper, .. } => *upper,
        }
    }
}

impl<K, J> BoundInto<K, J> for Bound<K>
where
    K: ClassKind,
    J: ClassKind<Bound = Bound<J>>,
{
    fn translate<F>(self, mut f: F) -> Bound<J>
    where
        F: FnMut(K::Id) -> J::Id,
    {
        match self {
            Bound::Enter { lower, upper } => Bound::Enter {
                lower: f(lower),
                upper: f(upper),
            },
            Bound::Change { lower, upper } => Bound::Change {
                lower: f(lower),
                upper: f(upper),
            },
            Bound::Equal { lower, upper } => Bound::Equal {
                lower: f(lower),
                upper: f(upper),
            },
        }
    }
}
