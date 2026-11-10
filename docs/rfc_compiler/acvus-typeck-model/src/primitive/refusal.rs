//! Refusals: what solving refuses of a program, and the items that met there.
//!
//! A rule says what it refuses, and of which classes ([`Refused`]). What settled
//! those classes is gathered once, when solving ends, and by no rule
//! ([`Refusal`]): it is an output, and no rule reads it.

use std::collections::BTreeSet;

use super::ItemId;
use super::class::ClassKind;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Cause {
    /// Two heads that do not join, or that are not of one layer where they are
    /// compared.
    Mismatch,
    /// An argument does not go to the type it is passed to: the language gives
    /// nothing, and not exactly one way through declared functions goes.
    NoChange,
    /// The domains registered on a class admit no head, or not the head it has.
    EmptyDomain,
    /// A class is a part of itself.
    Occurs,
    /// A part named on a class has no place in the head the class is decided as.
    NotHeld,
    /// A class is still undecided when solving ends.
    UndecidedClass,
}

/// What is refused, as whoever refuses it says it: of these classes, by these
/// items.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Refused<K>
where
    K: ClassKind,
{
    pub cause: Cause,
    /// What was judged: the bounds, the projections, the literals.
    pub items: BTreeSet<ItemId>,
    /// The classes it was judged of.
    pub of: Vec<K::Id>,
}

/// A refusal, with what settled the classes it is of.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Refusal {
    pub cause: Cause,
    /// What was judged, and what decided the head of each class it was judged of.
    pub items: BTreeSet<ItemId>,
    /// Every connection that made each of those classes one class, or made one
    /// a class that holds it as a part.
    pub connections: BTreeSet<ItemId>,
}
