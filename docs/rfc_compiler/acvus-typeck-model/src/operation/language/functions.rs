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

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::class::{ClassKind, State, Template};
use crate::primitive::constraint::Constraint;
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{Shape, Ty};
use crate::primitive::id::Arena;
use crate::primitive::{ItemId, LambdaId, Name, NameId, ScopeId, TemplateId};
use crate::store::classes::Classes;

use super::super::law::{Pair, Scribe};

/// One declared function, `parameters -> result`, written once as a template
/// with one class for each type it writes: a hole is one class wherever the
/// signature writes it.
#[derive(Clone, Debug)]
pub struct Row {
    pub name: Name,
    pub(super) template: Classes<Template>,
    pub(super) parameters: Vec<TemplateId>,
    pub(super) result: TemplateId,
    /// The brands the function makes, each the class that stands for it in the
    /// template: see [`made_by`].
    makes: BTreeSet<TemplateId>,
}

/// The brands a function of this signature makes: what the signature writes at
/// a brand, but for what a parameter writes.
///
/// A brand a parameter writes is the brand of what is passed. Nothing passed can
/// be any other brand the signature writes, so the function makes that one, and
/// each use of it makes one of its own.
fn made_by(template: &Classes<Template>, parameters: &[TemplateId]) -> BTreeSet<TemplateId> {
    let at_a_brand = written_at_a_brand(template);
    let by_a_parameter = written_by(template, parameters);
    &at_a_brand - &by_a_parameter
}

/// Each class that stands at the brand of a type `template` writes.
fn written_at_a_brand(template: &Classes<Template>) -> BTreeSet<TemplateId> {
    let brand_of = |class| {
        let State::Decided(decided) = &template.get(class).state else {
            return None;
        };
        let Shape::Extern(_, _, brand) = &decided.shape else {
            return None;
        };
        Some(template.find(*brand).id())
    };
    template.roots().filter_map(brand_of).collect()
}

/// Each class the types of `parameters` write, at every depth.
fn written_by(template: &Classes<Template>, parameters: &[TemplateId]) -> BTreeSet<TemplateId> {
    let themselves = parameters.iter().map(|parameter| template.find(*parameter).id());
    let below_them = template.below(parameters.iter().copied());
    themselves.chain(below_them).collect()
}

/// A row's signature, made anew for one use of the function.
pub(super) struct Used<K>
where
    K: ClassKind,
{
    /// Each class of the row's template, with the class it became.
    pub(super) classes: BTreeMap<TemplateId, K::Id>,
    /// Each brand the function makes, made for this use: the class that stands
    /// for it in the signature and the brand are one type.
    pub(super) brands: Vec<Pair<K>>,
}

impl Row {
    /// Makes the signature anew in `scope`, by `at`, and with it a brand for each
    /// the function makes.
    pub(super) fn used<K>(&self, scribe: &mut Scribe<'_, K>, scope: ScopeId, at: ItemId) -> Used<K>
    where
        K: ClassKind<Constraint = Constraint<K>>,
    {
        let classes = scribe.anew(&self.template, scope, at);
        let brands = self.makes.iter().map(|stands_for| Pair::Classes(classes[stands_for], scribe.brand(scope, at))).collect();
        Used { classes, brands }
    }

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

/// A function called by name: a row of the table, or a function of a module.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::operation) enum Named {
    Row(usize),
    Method(usize),
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

    /// Each function of the name `name` that may take a receiver, with the
    /// template that writes its signature and its first parameter there: a row
    /// that has a parameter, and a function of a module whose first parameter
    /// writes a type.
    pub(super) fn taking_a_receiver<'a>(&'a self, name: &'a Name) -> impl Iterator<Item = (Named, &'a Classes<Template>, TemplateId)> {
        let rows = self.rows.iter().enumerate().filter(move |(_, row)| row.name == *name).filter_map(|(index, row)| {
            let first = row.parameters.first()?;
            Some((Named::Row(index), &row.template, *first))
        });
        let methods = self.methods.iter().enumerate().filter(move |(_, method)| method.name == *name);
        let methods = methods.map(|(index, method)| (Named::Method(index), &method.template, method.receiver));
        rows.chain(methods)
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
        let parameters: Vec<TemplateId> = parameters
            .iter()
            .map(|parameter| template.write(scopes.root(), parameter, by, &mut written))
            .collect();
        let result = template.write(scopes.root(), result, by, &mut written);
        let makes = made_by(&template, &parameters);
        self.rows.push(Row {
            name,
            template,
            parameters,
            result,
            makes,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::ty::{Mutability, Scalar};

    fn hole(name: &str) -> Ty {
        Ty::Hole(Name::new(name))
    }

    fn branded(name: &str, brand: Ty) -> Ty {
        Ty::Node(Box::new(Shape::Extern(Name::new(name), Vec::new(), brand)))
    }

    /// The holes that stand for the brands a function `parameters -> result`
    /// makes.
    fn made(parameters: &[Ty], result: &Ty) -> Vec<Ty> {
        let scopes = Scopes::new();
        let mut items: Arena<ItemId, ()> = Arena::default();
        let by = items.push(());
        let mut template = Classes::default();
        let mut written = BTreeMap::new();
        let parameters: Vec<TemplateId> = parameters.iter().map(|parameter| template.write(scopes.root(), parameter, by, &mut written)).collect();
        template.write(scopes.root(), result, by, &mut written);
        let makes = made_by(&template, &parameters);
        written.into_iter().filter(|(_, class)| makes.contains(class)).map(|(ty, _)| ty).collect()
    }

    #[test]
    fn a_function_makes_the_brand_its_result_writes_and_no_parameter_does() {
        let u8_ = || Ty::Node(Box::new(Shape::Scalar(Scalar::U8)));
        assert_eq!(made(&[u8_()], &branded("F", hole("B"))), [hole("B")]);
        assert_eq!(made(&[], &branded("F", hole("B"))), [hole("B")]);
        // Two types of the result, two brands.
        let pair = Ty::Node(Box::new(Shape::Tuple(vec![branded("F", hole("A")), branded("G", hole("B"))])));
        assert_eq!(made(&[u8_()], &pair), [hole("A"), hole("B")]);
    }

    #[test]
    fn a_brand_a_parameter_writes_is_not_made() {
        let lent = || Ty::Node(Box::new(Shape::Ref(Mutability::Shared, branded("F", hole("B")))));
        assert_eq!(made(&[lent()], &branded("Idx", hole("B"))), []);
        // The parameter itself being the hole.
        assert_eq!(made(&[hole("B")], &branded("F", hole("B"))), []);
        // One of two.
        let pair = Ty::Node(Box::new(Shape::Tuple(vec![branded("F", hole("A")), branded("G", hole("B"))])));
        assert_eq!(made(&[lent()], &pair), [hole("A")]);
    }

    #[test]
    fn a_hole_that_stands_at_no_brand_is_not_made() {
        let vec = Ty::Node(Box::new(Shape::Struct(Name::new("Vec"), vec![hole("A")])));
        assert_eq!(made(&[], &vec), []);
    }
}

