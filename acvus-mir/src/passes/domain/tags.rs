//! The tag domain (matchck, patterns §3): which variants a value of an
//! enum, `Option` or `Result` may hold. A value the body makes has its
//! variant; a join is the union; a value from outside the body holds any
//! variant its type has, `Any`.

use std::collections::BTreeSet;

use acvus_utils::Astr;

/// The variants a value may hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tags {
    /// Reached by no value yet: the bottom.
    None,
    Of(BTreeSet<Astr>),
    /// Any variant the type has: a value from outside the body.
    Any,
}

impl Tags {
    /// The variants either may hold: `Any` where either may hold any.
    pub fn join(&self, other: &Tags) -> Tags {
        match (self, other) {
            (Tags::Any, _) | (_, Tags::Any) => Tags::Any,
            (Tags::None, one) | (one, Tags::None) => one.clone(),
            (Tags::Of(a), Tags::Of(b)) => Tags::Of(a.union(b).copied().collect()),
        }
    }
}
