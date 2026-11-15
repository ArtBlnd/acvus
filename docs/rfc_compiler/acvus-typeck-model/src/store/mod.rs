//! The store of classes.
//!
//! It knows identity and structure alone: which ids are one class, which
//! parts have one name, which classes a bound relates, and whether two shapes
//! have one constructor and the same members. Every other judgment, what a
//! bound means, a join, a change, a domain, a part with no place,
//! is its caller's, and the store returns what needs one.

pub mod classes;
pub mod ledger;
mod owners;
pub mod verify;

use crate::primitive::constraint::ConstraintOf;
use crate::primitive::scope::Scopes;

/// What solving reads and writes: the classes, and the scopes they lie in.
#[derive(Clone, Debug)]
pub struct Store<C>
where
    C: ConstraintOf,
{
    pub classes: classes::Classes<C>,
    pub scopes: Scopes,
}

// Written out, as a derive would ask a constraint to have a default, and none
// has.
impl<C> Default for Store<C>
where
    C: ConstraintOf,
{
    fn default() -> Self {
        Self {
            classes: classes::Classes::default(),
            scopes: Scopes::default(),
        }
    }
}
