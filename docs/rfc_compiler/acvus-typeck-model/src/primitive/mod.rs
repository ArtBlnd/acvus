//! The domain primitives of type checking: types, classes and their
//! constraints, declarations, uses, bodies and refusals. Operations on them
//! and the solvers over those operations live outside this module.

pub mod body;
pub mod class;
pub mod decl;
pub mod domain;
pub mod id;
pub mod record;
pub mod refusal;
pub mod scope;
pub mod ty;

pub use id::{BodyId, ClassId, ClosureId, GroupId, InstanceId, ItemId, ScopeId, TemplateId, UseId};

/// The name of a field, a variant, an enum, a module, an extern declaration,
/// a hole, or a place in the text.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Name(pub String);

impl Name {
    pub fn new<S>(name: S) -> Self
    where
        S: Into<String>,
    {
        Self(name.into())
    }
}
