//! Adjustments: what solving writes back to the text.
//!
//! An adjustment is an operation the text does not write and the program
//! performs, for a later stage to make explicit. It is an output: no constraint
//! reads one, and it decides no head.

use std::collections::BTreeMap;

use super::{ItemId, Name};
use super::record::Coercion;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Adjustment {
    /// A reference is read through: `p.f` is `(*p).f`, and `&a` is `&*a`. One
    /// is enough, as no reference has a reference as its target (O23).
    Deref,
    /// A coercion the language gives: the reborrow where a lower bound reaches a
    /// shared head (O10), and any of them at a boundary (O11).
    Coerce(Coercion),
    /// A declared conversion is called, in the order of the chain (O12).
    Convert(Name),
}

/// The adjustments of each item, in the order they apply.
///
/// Every copy of a template keeps the items of its text, so all instances of
/// one text share an entry here, though each may take its own adjustment.
pub type Adjustments = BTreeMap<ItemId, Vec<Adjustment>>;
