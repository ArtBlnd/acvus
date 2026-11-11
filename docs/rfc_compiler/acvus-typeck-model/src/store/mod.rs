//! The store of classes.
//!
//! It knows identity and structure alone: which ids are one class, which
//! parts have one name, which classes a bound relates, and whether two shapes
//! have one constructor and the same members. Every other judgment, what a
//! bound means, a join, a change, a domain, an instance, a part with no place,
//! is its caller's, and the store returns what needs one.

pub mod classes;
pub mod closures;
pub mod template;
pub mod verify;

use crate::primitive::class::ClassKind;
use crate::primitive::scope::Scopes;

/// What is solved: the classes, the scopes they lie in, and the closures named
/// over them. Solving takes the three together, as each names the others: a
/// class lies in a scope, and a closure captures classes and is copied into a
/// scope.
#[derive(Clone, Debug, Default)]
pub struct Store<K>
where
    K: ClassKind,
{
    pub classes: classes::Classes<K>,
    pub scopes: Scopes,
    pub closures: closures::Closures<K>,
}
