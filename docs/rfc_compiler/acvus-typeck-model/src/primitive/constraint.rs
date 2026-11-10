//! Constraints: what solving records, and holds until it is finished.
//!
//! A constraint is a bound, which relates two classes and leaves them two, or a
//! suspended constraint, which waits for a class's head and then adds constraints.
//!
//! A union makes two classes one, at every depth. A bound keeps them apart at
//! the top layer, where what it means differs by its kind, and every part below
//! that layer is one type with the corresponding part. It gives heads while it
//! waits: a class may take its head from either end of a bound.

use std::fmt::Debug;

use super::class::ClassKind;

/// A constraint as a store of the kind `K` sees it.
///
/// These methods are all a store reads of a constraint, so what a constraint
/// means stays with whoever defines it.
pub trait ConstraintOf<K>: Clone + Ord + Debug
where
    K: ClassKind,
{
    fn lower(&self) -> K::Id;
    fn upper(&self) -> K::Id;
}

/// A constraint of the kind `K` written as one of the kind `J`, as a copy of a
/// template needs it.
pub trait ConstraintInto<K, J>
where
    K: ClassKind,
    J: ClassKind,
{
    fn translate<F>(self, f: F) -> J::Constraint
    where
        F: FnMut(K::Id) -> J::Id;
}

/// What solving records between two classes, and holds until it is finished: a
/// bound, or a suspended constraint a store holds as it holds a bound.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Constraint<K>
where
    K: ClassKind,
{
    Bound(Bound<K>),
    Suspended(Suspended<K>),
}

/// The bounds of the language, which differ only at the top layer.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Bound<K>
where
    K: ClassKind,
{
    /// `upper`'s head is the join of the heads that enter it. The
    /// one operation `lower` may take to reach it is the reborrow. Classes
    /// that enter one another are one class.
    Enter { lower: K::Id, upper: K::Id },
    /// `lower` is passed to `upper`: by subtyping the language gives, or else
    /// through declared functions. It decides neither head.
    Change { lower: K::Id, upper: K::Id },
    /// The two heads are equal. It decides neither head and makes no
    /// change, so it can be judged one layer at a time.
    Equal { lower: K::Id, upper: K::Id },
}

/// The suspended constraints of the language: [`Suspended::Alone`] and
/// [`Suspended::Exit`] are recorded, looked at and finished as a bound is.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Suspended<K>
where
    K: ClassKind,
{
    /// `lower as _`: the type `lower` comes to by itself enters `upper`. No head
    /// is asked of `lower`, and it gives `upper` none until that type has a head
    /// at every depth.
    Alone { lower: K::Id, upper: K::Id },
    /// The exit of `e?`: `lower` is what `e` exits with, and `upper` the error of
    /// the body's result. What enters `upper` is told by the head of `lower`, and
    /// nothing enters it until that head is there.
    Exit { lower: K::Id, upper: K::Id },
}

impl<K> From<Bound<K>> for Constraint<K>
where
    K: ClassKind,
{
    fn from(bound: Bound<K>) -> Self {
        Constraint::Bound(bound)
    }
}

impl<K> From<Suspended<K>> for Constraint<K>
where
    K: ClassKind,
{
    fn from(suspended: Suspended<K>) -> Self {
        Constraint::Suspended(suspended)
    }
}

impl<K> ConstraintOf<K> for Bound<K>
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

impl<K> ConstraintOf<K> for Suspended<K>
where
    K: ClassKind,
{
    fn lower(&self) -> K::Id {
        match self {
            Suspended::Alone { lower, .. } | Suspended::Exit { lower, .. } => *lower,
        }
    }

    fn upper(&self) -> K::Id {
        match self {
            Suspended::Alone { upper, .. } | Suspended::Exit { upper, .. } => *upper,
        }
    }
}

impl<K> ConstraintOf<K> for Constraint<K>
where
    K: ClassKind,
{
    fn lower(&self) -> K::Id {
        match self {
            Constraint::Bound(bound) => bound.lower(),
            Constraint::Suspended(suspended) => suspended.lower(),
        }
    }

    fn upper(&self) -> K::Id {
        match self {
            Constraint::Bound(bound) => bound.upper(),
            Constraint::Suspended(suspended) => suspended.upper(),
        }
    }
}

impl<K, J> ConstraintInto<K, J> for Constraint<K>
where
    K: ClassKind,
    J: ClassKind<Constraint = Constraint<J>>,
{
    fn translate<F>(self, mut f: F) -> Constraint<J>
    where
        F: FnMut(K::Id) -> J::Id,
    {
        match self {
            Constraint::Bound(Bound::Enter { lower, upper }) => Bound::Enter {
                lower: f(lower),
                upper: f(upper),
            }
            .into(),
            Constraint::Bound(Bound::Change { lower, upper }) => Bound::Change {
                lower: f(lower),
                upper: f(upper),
            }
            .into(),
            Constraint::Bound(Bound::Equal { lower, upper }) => Bound::Equal {
                lower: f(lower),
                upper: f(upper),
            }
            .into(),
            Constraint::Suspended(Suspended::Alone { lower, upper }) => Suspended::Alone {
                lower: f(lower),
                upper: f(upper),
            }
            .into(),
            Constraint::Suspended(Suspended::Exit { lower, upper }) => Suspended::Exit {
                lower: f(lower),
                upper: f(upper),
            }
            .into(),
        }
    }
}
