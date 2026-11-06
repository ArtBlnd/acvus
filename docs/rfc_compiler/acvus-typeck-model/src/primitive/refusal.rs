//! Refusals (O22).

use std::collections::BTreeSet;

use super::ItemId;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Cause {
    /// Two shapes of constructors that do not join (Joins, O6).
    Mismatch,
    /// No change goes at a boundary (O11).
    NoChange,
    /// O8
    EmptyDomain,
    /// O5
    Occurs,
    /// A part named on a class has no place in the shape the class is
    /// decided as (O7).
    NotHeld,
    /// A closure has no instance at a use (O30).
    NoInstance,
    /// A brand type leaves its scope (O28).
    Escape,
    /// A class is still undecided when solving ends (O21).
    UndecidedClass,
    /// A use is still undecided when solving ends (O21).
    UndecidedUse,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Refusal {
    pub cause: Cause,
    pub items: BTreeSet<ItemId>,
}
