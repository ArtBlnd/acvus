//! The functions declared from outside: a table, each row a name and a template.
//!
//! A function no declaration writes is of the table too where the table is told
//! of it, by the type its first parameter writes ([`Told`]): a call by name
//! finds it as it finds a row, and what calling it comes to is said by whoever
//! told of it.
//!
//! A row adds no rule: it is read as any type given is read, as far as it is
//! written. So what tells when a row can be told to go is the row itself, by how
//! deep its parameters write. A value is passed through a row of one parameter
//! ([`super::passing`]).

use std::collections::BTreeMap;

use crate::primitive::class::Template as OfATemplate;
use crate::primitive::id::Arena;
use crate::primitive::scope::Scopes;
use crate::primitive::ty::Ty;
use crate::primitive::{ItemId, Name, NameId, TemplateId};
use crate::store::classes::Classes;

use super::templates::Template;

/// One declared function, `parameters -> result`: its name, and the template
/// its signature writes, with one class for each type it writes. A hole is one
/// class wherever the signature writes it.
#[derive(Clone, Debug)]
pub struct Row {
    pub name: Name,
    pub(super) template: Template,
}

impl Row {
    /// The parameter of a row of one parameter: what a value is passed to, to be
    /// passed through the row. A row of another count takes no value so.
    pub(super) fn only_parameter(&self) -> Option<TemplateId> {
        match self.template.parameters[..] {
            [parameter] => Some(parameter),
            _ => None,
        }
    }
}

/// A function called by name that no declaration writes, as the table was told
/// of it: its name, and the type its first parameter writes, which a value of
/// that type is the receiver of. What it is besides is not told here.
#[derive(Clone, Debug)]
pub struct Told {
    pub name: Name,
    /// The classes that write the type of its first parameter, read for that
    /// type and never copied from here.
    pub(super) classes: Classes<OfATemplate>,
    pub(super) receiver: TemplateId,
}

/// A function called by name: a row of the table, or one the table was told of,
/// each by its place.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::operation) enum Named {
    Row(usize),
    Told(usize),
}

/// Every function declared, in the order declared, every function the table was
/// told of, and every name a function is called by.
#[derive(Clone, Debug, Default)]
pub struct Functions {
    pub(super) rows: Vec<Row>,
    pub(super) told: Vec<Told>,
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

    /// Each function of the name `name` that may take a receiver, with the
    /// classes that write its first parameter and that parameter there: a row
    /// that has a parameter, and each function the table was told of.
    pub(super) fn taking_a_receiver<'a>(&'a self, name: &'a Name) -> impl Iterator<Item = (Named, &'a Classes<OfATemplate>, TemplateId)> {
        let rows = self.rows.iter().enumerate().filter(move |(_, row)| row.name == *name).filter_map(|(index, row)| {
            let first = row.template.parameters.first()?;
            Some((Named::Row(index), &row.template.classes, *first))
        });
        let told = self.told.iter().enumerate().filter(move |(_, told)| told.name == *name);
        let told = told.map(|(index, told)| (Named::Told(index), &told.classes, told.receiver));
        rows.chain(told)
    }

    /// Tells the table of a function `name` no declaration writes, which takes a
    /// receiver: `receiver`, of `classes`, is its first parameter. It gives the
    /// place the table names the function by.
    pub fn told(&mut self, name: Name, classes: Classes<OfATemplate>, receiver: TemplateId) -> usize {
        self.told.push(Told { name, classes, receiver });
        self.told.len() - 1
    }

    /// Declares `name: parameters -> result`, by `by`.
    pub fn declare(&mut self, name: Name, parameters: &[Ty], result: &Ty, scopes: &Scopes, by: ItemId) {
        let mut classes = Classes::default();
        let mut written = BTreeMap::new();
        let parameters: Vec<TemplateId> = parameters
            .iter()
            .map(|parameter| classes.write(scopes.root(), parameter, by, &mut written))
            .collect();
        let result = classes.write(scopes.root(), result, by, &mut written);
        self.rows.push(Row {
            name,
            template: Template::of_a_signature(classes, parameters, result),
        });
    }
}
