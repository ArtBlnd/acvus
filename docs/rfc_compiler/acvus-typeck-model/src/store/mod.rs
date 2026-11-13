//! The store of classes.
//!
//! It knows identity and structure alone: which ids are one class, which
//! parts have one name, which classes a bound relates, and whether two shapes
//! have one constructor and the same members. Every other judgment, what a
//! bound means, a join, a change, a domain, an instance, a part with no place,
//! is its caller's, and the store returns what needs one.

pub mod classes;
pub mod closures;
pub mod ledger;
mod owners;
pub mod template;
pub mod verify;

use crate::primitive::constraint::ConstraintOf;
use crate::primitive::id::Arena;
use crate::primitive::scope::Scopes;
use crate::primitive::{BrandId, ItemId};

/// What solving reads and writes: the classes, the scopes they lie in, and the
/// closures named of those scopes. Solving takes the three together, as each
/// names the others: a class lies in a scope, and a closure is a scope with
/// classes of it, and is made anew into a scope. Not all of it is solved: what a
/// declaration writes is read and made anew, and never changed.
#[derive(Clone, Debug)]
pub struct Store<C>
where
    C: ConstraintOf,
{
    pub classes: classes::Classes<C>,
    pub scopes: Scopes,
    pub closures: closures::Closures,
    /// Every brand made, with the item it was made by.
    pub brands: Arena<BrandId, ItemId>,
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
            closures: closures::Closures::default(),
            brands: Arena::default(),
        }
    }
}
