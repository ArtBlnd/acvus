//! Refusals: what solving refuses of a program, and the items that met there.

use std::collections::BTreeSet;

use super::ItemId;

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

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Refusal {
    pub cause: Cause,
    pub items: BTreeSet<ItemId>,
}
