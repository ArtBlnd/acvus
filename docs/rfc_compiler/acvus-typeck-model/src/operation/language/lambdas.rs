//! The lambdas of the text: a table, each row a template and its openings.

use crate::primitive::class::Template;
use crate::primitive::id::Arena;
use crate::primitive::{ItemId, LambdaId, TemplateId};
use crate::store::classes::Classes;

/// One lambda: the classes and constraints its body writes, and the classes of
/// them that open to what is outside.
#[derive(Clone, Debug)]
pub struct Lambda {
    pub template: Classes<Template>,
    pub parameters: Vec<TemplateId>,
    pub result: TemplateId,
    /// The classes that stand for what the body captures. In a copy each is
    /// one type with the class the closure captures there.
    pub captures: Vec<TemplateId>,
    /// The lambdas the body defines.
    pub defines: Vec<Defined>,
}

/// A lambda defined in a body: each copy of the body names a closure of it
/// anew, over the copies of the classes written where it captures.
#[derive(Clone, Debug)]
pub struct Defined {
    pub lambda: LambdaId,
    pub captures: Vec<TemplateId>,
    /// The class of the lambda as an expression, which its function value enters.
    pub value: TemplateId,
    /// The item of the definition.
    pub by: ItemId,
}

/// Every lambda of the text, in the order written.
#[derive(Clone, Debug, Default)]
pub struct Lambdas {
    lambdas: Arena<LambdaId, Lambda>,
}

impl Lambdas {
    pub fn write(&mut self, lambda: Lambda) -> LambdaId {
        self.lambdas.push(lambda)
    }

    pub fn get(&self, lambda: LambdaId) -> &Lambda {
        self.lambdas.get(lambda)
    }
}
