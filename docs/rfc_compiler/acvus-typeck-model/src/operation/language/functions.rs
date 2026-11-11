//! The functions declared from outside: a table, each row a name and a signature.
//!
//! A function of a module that takes a receiver is of the table too, by the type
//! its first parameter writes ([`Method`]): a call by name finds it as it finds a
//! row.
//!
//! A row adds no rule: it is read as any type given is read, as far as it is
//! written. So what tells when a row can be told to go is the row itself, by how
//! deep its parameters write. A value is passed through a row of one parameter
//! ([`super::passing`]).

use std::collections::BTreeMap;

use crate::primitive::class::Template;
use crate::primitive::scope::Scopes;
use crate::primitive::ty::Ty;
use crate::primitive::id::Arena;
use crate::primitive::{ItemId, LambdaId, Name, NameId, TemplateId};
use crate::store::classes::Classes;

/// One declared function, `parameters -> result`, written once as a template
/// with one class for each type it writes: a hole is one class wherever the
/// signature writes it.
#[derive(Clone, Debug)]
pub struct Row {
    pub name: Name,
    pub(super) template: Classes<Template>,
    pub(super) parameters: Vec<TemplateId>,
    pub(super) result: TemplateId,
}

impl Row {
    /// The parameter of a row of one parameter: what a value is passed to, to be
    /// passed through the row. A row of another count takes no value so.
    pub(super) fn only_parameter(&self) -> Option<TemplateId> {
        match self.parameters[..] {
            [parameter] => Some(parameter),
            _ => None,
        }
    }
}

/// A function of a module whose first parameter writes a type: one a value of
/// that type is the receiver of.
#[derive(Clone, Debug)]
pub struct Method {
    pub name: Name,
    /// The lambda of the module, which gives the function when it is called.
    pub(super) module: LambdaId,
    /// The template of the function, read for the type its first parameter
    /// writes and never copied from here.
    pub(super) template: Classes<Template>,
    pub(super) receiver: TemplateId,
}

/// Every function declared, in the order declared, every function of a module
/// that takes a receiver, and every name a function is called by.
#[derive(Clone, Debug, Default)]
pub struct Functions {
    pub(super) rows: Vec<Row>,
    pub(super) methods: Vec<Method>,
    called: Arena<NameId, Name>,
}

impl Functions {
    /// The id of `name` as a name a function is called by: one id for one name.
    pub fn called(&mut self, name: &Name) -> NameId {
        let known = self.called.iter().find(|(_, known)| *known == name).map(|(id, _)| id);
        known.unwrap_or_else(|| self.called.push(name.clone()))
    }

    /// The name `id` is the id of.
    pub fn name(&self, id: NameId) -> &Name {
        self.called.get(id)
    }

    /// Says that the function `name` of the module whose lambda is `module` takes
    /// a receiver: `receiver`, of `template`, is its first parameter.
    pub fn method(&mut self, name: Name, module: LambdaId, template: Classes<Template>, receiver: TemplateId) {
        self.methods.push(Method {
            name,
            module,
            template,
            receiver,
        });
    }

    /// Declares `name: parameters -> result`, by `by`.
    pub fn declare(&mut self, name: Name, parameters: &[Ty], result: &Ty, scopes: &Scopes, by: ItemId) {
        let mut template = Classes::default();
        let mut written = BTreeMap::new();
        let parameters = parameters
            .iter()
            .map(|parameter| template.write(scopes.root(), parameter, by, &mut written))
            .collect();
        let result = template.write(scopes.root(), result, by, &mut written);
        self.rows.push(Row {
            name,
            template,
            parameters,
            result,
        });
    }
}
