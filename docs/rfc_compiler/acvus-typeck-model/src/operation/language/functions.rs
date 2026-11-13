//! The functions called by name: a table, each row a name and a closure.
//!
//! A declared function is a closure whose body holds its signature and nothing
//! besides, with one class for each type the signature writes: its parameters
//! are written exactly. A function no declaration writes is of the table too
//! where the table is told of it ([`Told`]). A call by name finds either by the
//! type its first parameter writes, read where it lies and never connected
//! there: what is connected is the body made anew for the call.
//!
//! A row adds no rule: it is read as any type given is read, as far as it is
//! written. So what tells when a row can be told to go is the row itself, by how
//! deep its parameters write. A value is passed through a row of one parameter
//! ([`super::passing`]).

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::class::{ClassKind, State};
use crate::primitive::id::Arena;
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{Shape, Ty};
use crate::primitive::{ClosureId, ItemId, Name, NameId};
use crate::store::classes::Classes;
use crate::store::closures::{Closure, Closures};

/// One declared function, `parameters -> result`: its name, and the closure
/// whose body is its signature.
#[derive(Clone, Debug)]
pub struct Row {
    pub name: Name,
    pub(super) closure: ClosureId,
}

/// A function called by name that no declaration writes, as the table was told
/// of it: its name and its closure. What it is besides is not told here.
#[derive(Clone, Debug)]
pub struct Told {
    pub name: Name,
    pub(super) closure: ClosureId,
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

    /// Each function of the name `name`, with its closure: every row, and every
    /// function the table was told of.
    pub(super) fn named<'a>(&'a self, name: &'a Name) -> impl Iterator<Item = (Named, ClosureId)> + 'a {
        let rows = self.rows.iter().enumerate().filter(move |(_, row)| row.name == *name);
        let rows = rows.map(|(index, row)| (Named::Row(index), row.closure));
        let told = self.told.iter().enumerate().filter(move |(_, told)| told.name == *name);
        let told = told.map(|(index, told)| (Named::Told(index), told.closure));
        rows.chain(told)
    }

    /// Tells the table of a function `name` no declaration writes, whose closure
    /// is `closure`. It gives the place the table names the function by.
    pub fn told(&mut self, name: Name, closure: ClosureId) -> usize {
        self.told.push(Told { name, closure });
        self.told.len() - 1
    }

    /// Declares `name: parameters -> result`, by `by`: its signature is written
    /// as a body of its own, within `within`, and a closure is named of it.
    #[allow(clippy::too_many_arguments)]
    pub fn declare<K>(&mut self, classes: &mut Classes<K>, scopes: &mut Scopes, closures: &mut Closures<K>, name: Name, parameters: &[Ty], result: &Ty, by: ItemId)
    where
        K: ClassKind,
    {
        let body = scopes.child(scopes.root());
        let mut written = BTreeMap::new();
        let parameters: Vec<K::Id> = parameters.iter().map(|parameter| classes.write(body, parameter, by, &mut written)).collect();
        let result = classes.write(body, result, by, &mut written);
        let makes = made_by(classes, &written, &parameters);
        let closure = closures.name(Closure {
            body,
            parameters,
            result,
            makes,
        });
        self.rows.push(Row { name, closure });
    }
}

/// The brands a function of this signature makes: what the signature writes at
/// a brand, but for what a parameter writes. `written` holds each class the
/// signature wrote.
///
/// A brand a parameter writes is the brand of what is passed. Nothing passed can
/// be any other brand the signature writes, so the function makes that one, and
/// each use of it makes one of its own.
fn made_by<K>(classes: &Classes<K>, written: &BTreeMap<Ty, K::Id>, parameters: &[K::Id]) -> BTreeSet<K::Id>
where
    K: ClassKind,
{
    let brand_of = |class: &K::Id| {
        let State::Decided(decided) = &classes.get(classes.find(*class)).state else {
            return None;
        };
        let Shape::Extern(_, _, brand) = &decided.shape else {
            return None;
        };
        Some(classes.find(*brand).id())
    };
    let at_a_brand: BTreeSet<K::Id> = written.values().filter_map(brand_of).collect();
    let themselves = parameters.iter().map(|parameter| classes.find(*parameter).id());
    let by_a_parameter: BTreeSet<K::Id> = themselves.chain(classes.below(parameters.iter().copied())).collect();
    &at_a_brand - &by_a_parameter
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::ClassId;
    use crate::primitive::class::Live;
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
        let mut classes = Classes::<Live>::default();
        let mut written = BTreeMap::new();
        let parameters: Vec<ClassId> = parameters.iter().map(|parameter| classes.write(scopes.root(), parameter, by, &mut written)).collect();
        classes.write(scopes.root(), result, by, &mut written);
        let makes = made_by(&classes, &written, &parameters);
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
