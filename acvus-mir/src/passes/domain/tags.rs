//! The tag domain (matchck, patterns §3): which variants a value of an
//! enum, `Option` or `Result` may hold. A value the body makes has its
//! variant; a join is the union; a value from outside the body holds any
//! variant its type has, `Any`.

use std::collections::BTreeSet;

use acvus_utils::Astr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tags {
    /// Reached by no value yet: the bottom.
    None,
    Of(BTreeSet<Astr>),
    /// Any variant the type has: a value from outside the body.
    Any,
}
