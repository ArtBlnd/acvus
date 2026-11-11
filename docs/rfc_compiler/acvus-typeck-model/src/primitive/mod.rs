//! The domain primitives of type checking: types, classes and what they hold,
//! constraints, adjustments and refusals. The store of classes, and the operations
//! that solve over it, live outside this module.

pub mod adjustment;
pub mod components;
pub mod constraint;
pub mod class;
pub mod domain;
pub mod id;
pub mod refusal;
pub mod scope;
pub mod ty;

pub use id::{ClassId, ClosureId, ItemId, LambdaId, NameId, ScopeId, TemplateId};

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
