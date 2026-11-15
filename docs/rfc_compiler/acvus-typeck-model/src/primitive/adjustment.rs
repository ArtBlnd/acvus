//! Adjustments: what solving writes back to the text.
//!
//! An adjustment is an operation the text does not write and the program
//! performs, for a later stage to make explicit. It is an output: no constraint
//! reads one, and it decides no head.

use std::collections::BTreeMap;

use super::ItemId;

/// Subtyping the language gives: one index of a head loosened, or `!` going to
/// any type.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Subtyping {
    /// `&mut T` to `&T`.
    Reborrow,
    /// `!` to any type.
    Never,
    /// An object to one with fewer fields.
    DropFields,
    /// An enum or an `Error` to one with more variants.
    AddVariants,
    /// An array behind a reference to a slice.
    Unsize,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Adjustment {
    /// A reference is read through: `p.f` is `(*p).f`, and `&&U` goes to `&U`.
    /// One is recorded for each reference read through.
    Deref,
    /// Subtyping the language gives was taken: where an argument is passed, any
    /// of them, and where a value enters a class, the reborrow alone.
    Subtype(Subtyping),
}

/// The adjustments of each item, in the order they apply.
pub type Adjustments = BTreeMap<ItemId, Vec<Adjustment>>;
