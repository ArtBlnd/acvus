//! The functions declared from outside: a table, each row a name and a signature.
//!
//! The language states one rule of them ([`super::passing`]), and a row adds no
//! rule: it is read as any type given is read, as far as it is written. So what
//! tells when a row can be told to go is the row itself, by how deep its
//! parameter writes.

use std::collections::BTreeMap;

use crate::primitive::class::Template;
use crate::primitive::scope::Scopes;
use crate::primitive::ty::Ty;
use crate::primitive::{ItemId, Name, TemplateId};
use crate::store::classes::Classes;

/// One declared function, `parameter -> result`, written once as a template with
/// one class for each type it writes: a hole is one class wherever the signature
/// writes it.
#[derive(Clone, Debug)]
pub struct Row {
    pub name: Name,
    pub(super) template: Classes<Template>,
    pub(super) parameter: TemplateId,
    pub(super) result: TemplateId,
}

/// Every function declared, in the order declared.
#[derive(Clone, Debug, Default)]
pub struct Functions {
    pub(super) rows: Vec<Row>,
}

impl Functions {
    /// Declares `name: parameter -> result`, by `by`.
    pub fn declare(&mut self, name: Name, parameter: &Ty, result: &Ty, scopes: &Scopes, by: ItemId) {
        let mut template = Classes::default();
        let mut written = BTreeMap::new();
        let parameter = template.write(scopes.root(), parameter, by, &mut written);
        let result = template.write(scopes.root(), result, by, &mut written);
        self.rows.push(Row {
            name,
            template,
            parameter,
            result,
        });
    }
}
