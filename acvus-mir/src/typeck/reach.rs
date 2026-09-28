//! The one reach rule (types §6), pure.
//!
//! Arguments are read in order, a tuple's parts in order: at each one,
//! stepping up from its exact type along the cast tree, the candidates that
//! match at the first step where any match are kept. Exactly one must be
//! left. Cost: positions × tree height × candidates.

use crate::decl::{CastTree, Head};

/// A candidate's parameter heads, in argument order.
pub(super) struct Candidate<'c> {
    pub params: &'c [Head],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum NoReach {
    None,
    Several(Vec<usize>),
}

/// The index of the one candidate reached by `args`, the closed heads of a
/// call's arguments.
pub(super) fn reach(candidates: &[Candidate<'_>], args: &[Head], casts: &CastTree) -> Result<usize, NoReach> {
    todo!("typeck::reach")
}
