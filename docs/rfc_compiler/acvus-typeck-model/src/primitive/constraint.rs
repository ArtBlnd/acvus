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

use super::{ClassId, NameId};

/// A constraint as a store sees it.
///
/// These methods are all a store reads of a constraint, so what a constraint
/// means stays with whoever defines it.
pub trait ConstraintOf: Clone + Ord + Debug {
    fn lower(&self) -> ClassId;
    fn upper(&self) -> ClassId;

    /// The same constraint, with every id it holds given by `f`, as a body made
    /// anew needs it.
    fn renamed<F>(self, f: F) -> Self
    where
        F: FnMut(ClassId) -> ClassId;
}

/// What solving records between two classes, and holds until it is finished: a
/// bound, or a suspended constraint a store holds as it holds a bound.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Constraint {
    Bound(Bound),
    Suspended(Suspended),
}

/// The bounds of the language, which differ only at the top layer.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Bound {
    /// `upper`'s head is the join of the heads that enter it. The
    /// one operation `lower` may take to reach it is the reborrow. Classes
    /// that enter one another are one class.
    Enter { lower: ClassId, upper: ClassId },
    /// `lower` is passed to `upper`: by subtyping the language gives, or else
    /// through declared functions. It decides neither head.
    Change { lower: ClassId, upper: ClassId },
    /// The two heads are equal. It decides neither head and makes no
    /// change, so it can be judged one layer at a time.
    Equal { lower: ClassId, upper: ClassId },
}

/// The suspended constraints of the language: [`Suspended::Alone`] and
/// [`Suspended::Exit`] are recorded, looked at and finished as a bound is.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Suspended {
    /// `lower as _`: the type `lower` comes to by itself enters `upper`. No head
    /// is asked of `lower`, and it gives `upper` none until that type has a head
    /// at every depth.
    Alone { lower: ClassId, upper: ClassId },
    /// The exit of `e?`: `lower` is what `e` exits with, and `upper` the error of
    /// the body's result. What enters `upper` is told by the head of `lower`, and
    /// nothing enters it until that head is there.
    Exit { lower: ClassId, upper: ClassId },
    /// `lower::name(ā)`: `lower` is the receiver, and `upper` the function type
    /// `Fn(ā) → r` the call writes. Which function of that name is called is told
    /// by the head of the receiver, and waits for it.
    Named { lower: ClassId, upper: ClassId, name: NameId },
}

impl From<Bound> for Constraint {
    fn from(bound: Bound) -> Self {
        Constraint::Bound(bound)
    }
}

impl From<Suspended> for Constraint {
    fn from(suspended: Suspended) -> Self {
        Constraint::Suspended(suspended)
    }
}

impl ConstraintOf for Bound {
    fn renamed<F>(self, mut f: F) -> Self
    where
        F: FnMut(ClassId) -> ClassId,
    {
        match self {
            Bound::Enter { lower, upper } => Bound::Enter { lower: f(lower), upper: f(upper) },
            Bound::Equal { lower, upper } => Bound::Equal { lower: f(lower), upper: f(upper) },
            Bound::Change { lower, upper } => Bound::Change { lower: f(lower), upper: f(upper) },
        }
    }

    fn lower(&self) -> ClassId {
        match self {
            Bound::Enter { lower, .. }
            | Bound::Change { lower, .. }
            | Bound::Equal { lower, .. } => *lower,
        }
    }

    fn upper(&self) -> ClassId {
        match self {
            Bound::Enter { upper, .. }
            | Bound::Change { upper, .. }
            | Bound::Equal { upper, .. } => *upper,
        }
    }
}

impl ConstraintOf for Suspended {
    fn renamed<F>(self, mut f: F) -> Self
    where
        F: FnMut(ClassId) -> ClassId,
    {
        match self {
            Suspended::Alone { lower, upper } => Suspended::Alone { lower: f(lower), upper: f(upper) },
            Suspended::Exit { lower, upper } => Suspended::Exit { lower: f(lower), upper: f(upper) },
            Suspended::Named { lower, upper, name } => Suspended::Named { lower: f(lower), upper: f(upper), name },
        }
    }

    fn lower(&self) -> ClassId {
        match self {
            Suspended::Alone { lower, .. } | Suspended::Exit { lower, .. } | Suspended::Named { lower, .. } => *lower,
        }
    }

    fn upper(&self) -> ClassId {
        match self {
            Suspended::Alone { upper, .. } | Suspended::Exit { upper, .. } | Suspended::Named { upper, .. } => *upper,
        }
    }
}

impl ConstraintOf for Constraint {
    fn renamed<F>(self, f: F) -> Self
    where
        F: FnMut(ClassId) -> ClassId,
    {
        match self {
            Constraint::Bound(bound) => Constraint::Bound(bound.renamed(f)),
            Constraint::Suspended(suspended) => Constraint::Suspended(suspended.renamed(f)),
        }
    }

    fn lower(&self) -> ClassId {
        match self {
            Constraint::Bound(bound) => bound.lower(),
            Constraint::Suspended(suspended) => suspended.lower(),
        }
    }

    fn upper(&self) -> ClassId {
        match self {
            Constraint::Bound(bound) => bound.upper(),
            Constraint::Suspended(suspended) => suspended.upper(),
        }
    }
}
