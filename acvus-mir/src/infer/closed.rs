//! Whether a term can still change (types §6): a decision reads only a
//! closed term.

use super::{Infer, Term};
use crate::decl::{CastTree, Head};

/// What the decisions still pending may do: typeck answers from its
/// decision graph; B only asks.
pub trait OpenSources {
    /// Whether some pending decision outside the cycles through `term` may
    /// still enter a value into it or bind it.
    fn may_change(&self, term: Term) -> bool;
}

/// A closed term's head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Closed {
    pub head: Head,
    pub term: Term,
}

impl Infer {
    /// `term`'s head, when nothing can change it: its head is the top of its
    /// cast tree, or no open source may change it.
    pub fn closed(&self, casts: &CastTree, open: &impl OpenSources, term: Term) -> Option<Closed> {
        todo!("infer::closed")
    }
}
