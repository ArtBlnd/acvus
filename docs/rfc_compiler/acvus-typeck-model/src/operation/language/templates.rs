//! Templates: what a use makes anew.
//!
//! A declared function is a template whose parameters are written exactly: it
//! holds its signature and nothing besides, in a store of its own, and a use
//! makes the whole of it anew (`Template::used`). A body is a template too, and
//! is written where it lies, in the store being solved (`store::template::anew`).

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::class::{self, ClassKind, State};
use crate::primitive::constraint::Constraint;
use crate::primitive::ty::Shape;
use crate::primitive::{ItemId, ScopeId, TemplateId};
use crate::store::classes::Classes;

use super::super::law::{Pair, Scribe};

/// The classes a signature writes, and the classes of them that open to what is
/// outside.
#[derive(Clone, Debug)]
pub struct Template {
    pub classes: Classes<class::Template>,
    pub parameters: Vec<TemplateId>,
    pub result: TemplateId,
    /// The brands a use makes, each the class that stands for it in the
    /// template: see [`made_by`].
    makes: BTreeSet<TemplateId>,
}

/// A template made anew for one use of it.
pub(super) struct Used<K>
where
    K: ClassKind,
{
    /// Each class of the template, with the class it became.
    pub(super) classes: BTreeMap<TemplateId, K::Id>,
    /// What is one type by the use: each class that stands for a brand the
    /// template makes with the brand made for this use.
    pub(super) pairs: Vec<Pair<K>>,
}

impl Template {
    /// The template of a signature `parameters -> result` that `classes` writes:
    /// a use makes each brand the signature writes in no parameter.
    pub fn of_a_signature(classes: Classes<class::Template>, parameters: Vec<TemplateId>, result: TemplateId) -> Self {
        let makes = made_by(&classes, &parameters);
        Self {
            classes,
            parameters,
            result,
            makes,
        }
    }

    /// Makes the template anew in `scope`, by `at`: every class of it, and a
    /// brand for each it makes, which the class that stands for it is one type
    /// with.
    ///
    /// This is the only place a template is made anew for a use.
    pub(super) fn used<K>(&self, scribe: &mut Scribe<'_, K>, scope: ScopeId, at: ItemId) -> Used<K>
    where
        K: ClassKind<Constraint = Constraint<K>>,
    {
        let classes = scribe.anew(&self.classes, scope);
        let pairs = self.makes.iter().map(|stands_for| Pair::Classes(classes[stands_for], scribe.brand(scope, at))).collect();
        Used { classes, pairs }
    }
}

/// The brands a function of this signature makes: what the signature writes at
/// a brand, but for what a parameter writes.
///
/// A brand a parameter writes is the brand of what is passed. Nothing passed can
/// be any other brand the signature writes, so the function makes that one, and
/// each use of it makes one of its own.
fn made_by(template: &Classes<class::Template>, parameters: &[TemplateId]) -> BTreeSet<TemplateId> {
    let at_a_brand = written_at_a_brand(template);
    let by_a_parameter = written_by(template, parameters);
    &at_a_brand - &by_a_parameter
}

/// Each class that stands at the brand of a type `template` writes.
fn written_at_a_brand(template: &Classes<class::Template>) -> BTreeSet<TemplateId> {
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
fn written_by(template: &Classes<class::Template>, parameters: &[TemplateId]) -> BTreeSet<TemplateId> {
    let themselves = parameters.iter().map(|parameter| template.find(*parameter).id());
    let below_them = template.below(parameters.iter().copied());
    themselves.chain(below_them).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::Name;
    use crate::primitive::id::Arena;
    use crate::primitive::scope::Scopes;
    use crate::primitive::ty::{Mutability, Scalar, Ty};

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
