//! The language's law: what `TERMS.md` rules, one rule in one place.
//!
//! | The law is asked | The rule | Where |
//! |---|---|---|
//! | what a constraint comes to | by the relation the constraint is: entering, being equal, being passed, `as _`, the exit of `?` | [`enter`], [`equal`], [`passing`], [`alone`], [`exit`] |
//! | which functions are declared | a table, and no rule | [`functions`] |
//! | which lambdas the text writes | a table, and no rule | [`lambdas`] |
//! | what a function value passed to a function type comes to | a copy of each closure | [`calling`] |
//! | which function a call by name calls | the one row of that name the receiver goes to | [`naming`] |
//! | which classes are one type by the constraints as a whole | classes that enter one another | [`enter::entering_one_another`] |
//! | where a named part lies | read through a reference | [`Language::place`] |
//! | what head a class may take | the rungs, in order | [`rungs`] |
//! | what is refused at the end | a class still undecided, a class that is its own part | [`rungs::end`] |

pub mod alone;
pub mod calling;
pub mod enter;
pub mod equal;
pub mod exit;
pub mod functions;
pub mod lambdas;
pub mod naming;
pub mod passing;
pub mod rungs;

use std::collections::BTreeMap;

use crate::primitive::{ItemId, Name};
use crate::primitive::adjustment::Adjustment;
use crate::primitive::constraint::{Bound, Constraint, ConstraintOf, Suspended};
use crate::primitive::class::{ClassKind, State};
use crate::primitive::domain::Domain;
use crate::primitive::refusal::Refused;
use crate::primitive::ty::{PartKey, Shape};
use crate::store::classes::Classes;

use super::law::{Heads, Law, Looked, Pair, Place, Scribe};

/// The rules of the language, with the functions declared from outside, and the
/// lambdas and the structs the text writes.
#[derive(Clone, Debug, Default)]
pub struct Language {
    pub functions: functions::Functions,
    pub lambdas: lambdas::Lambdas,
    /// The fields of each struct, in the order it declares them: the type of the
    /// field at a place is the struct's type parameter at that place.
    pub structs: BTreeMap<Name, Vec<Name>>,
}

impl<K> Law<K> for Language
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    fn look(&self, scribe: &mut Scribe<'_, K>, constraint: &Constraint<K>, at: ItemId) -> Looked<K> {
        let told = tell(&View::of(scribe.classes()), &self.functions, constraint, at);
        match told.then {
            Some(Then::Way(way)) => passing::take(scribe, &self.functions, way, constraint.lower(), at),
            Some(Then::Calls) => calling::take(scribe, &self.lambdas, constraint.lower(), constraint.upper(), at),
            Some(Then::Named(received)) => naming::take(scribe, &self.functions, received, constraint.lower(), constraint.upper(), at),
            Some(Then::Enters(shape)) => {
                let upper = constraint.upper();
                let scope = scribe.classes().get(scribe.classes().find(upper)).scope;
                let written = scribe.written(scope, shape, [at].into_iter().collect());
                Looked::Holds {
                    pairs: Vec::new(),
                    adjustments: Vec::new(),
                    constraints: vec![Bound::Enter { lower: written, upper }.into()],
                }
            }
            None => told.looked,
        }
    }

    fn same(&self, part: K::Id, declared: K::Id) -> Constraint<K> {
        Bound::Equal {
            lower: part,
            upper: declared,
        }
        .into()
    }

    fn one(&self, classes: &Classes<K>) -> Vec<(K::Id, K::Id, ItemId)> {
        enter::entering_one_another(classes)
    }

    /// A part named on a reference is the part of that name of what it refers to,
    /// read through it. The target is the one part a reference has a place for.
    ///
    /// A field named on a struct is the struct's type parameter at the place its
    /// declaration gives the field.
    fn place(&self, head: &Shape<K::Id>, key: &PartKey) -> Place {
        match (head, key) {
            (Shape::Ref(..), key) if *key != PartKey::Target => Place::Within {
                through: PartKey::Target,
                adjustment: Adjustment::Deref,
            },
            (Shape::Struct(name, _), PartKey::Field(field)) => {
                let declared = self.structs.get(name).and_then(|fields| fields.iter().position(|of| of == field));
                declared.map_or(Place::Nowhere, |at| Place::At(PartKey::Arg(at)))
            }
            _ => Place::Nowhere,
        }
    }

    /// A literal's class is a scalar of the literal's domain.
    fn admits(&self, head: &Shape<K::Id>, domain: &Domain) -> bool {
        matches!(head, Shape::Scalar(scalar) if domain.0.contains(scalar))
    }

    fn heads(&self, classes: &Classes<K>, asked: &[K::Id], held: &dyn Fn(K::Id) -> bool) -> Heads<K> {
        rungs::heads(classes, &self.functions, asked, held)
    }

    fn end(&self, classes: &Classes<K>, class: K::Id) -> Vec<Refused<K>> {
        rungs::end(classes, &self.functions, class)
    }
}

/// The store as a relation reads it: as it is, or with heads supposed of some
/// undecided classes, to ask what a constraint would come to if they took them.
#[derive(Clone, Debug)]
pub(super) struct View<'a, K>
where
    K: ClassKind,
{
    pub(super) classes: &'a Classes<K>,
    supposed: BTreeMap<K::Id, Shape<K::Id>>,
}

impl<'a, K> View<'a, K>
where
    K: ClassKind,
{
    pub(super) fn of(classes: &'a Classes<K>) -> Self {
        Self {
            classes,
            supposed: BTreeMap::new(),
        }
    }

    /// This view, with `class` supposed to have the head `head`.
    pub(super) fn supposing(mut self, class: K::Id, head: Shape<K::Id>) -> Self {
        self.supposed.insert(self.classes.find(class).id(), head);
        self
    }

    /// The head of `class`: its own where it is decided, or the one supposed.
    pub(super) fn head(&self, class: K::Id) -> Option<&Shape<K::Id>> {
        let root = self.classes.find(class);
        match &self.classes.get(root).state {
            State::Decided(decided) => Some(&decided.shape),
            State::Undecided(_) => self.supposed.get(&root.id()),
        }
    }
}

/// What a relation tells of a constraint: what it comes to, and, where it waits for a
/// class's head, the head it will hold that class's head against.
pub(super) struct Told<K>
where
    K: ClassKind,
{
    pub(super) looked: Looked<K>,
    pub(super) asks: Vec<(K::Id, Shape<K::Id>)>,
    /// What is still to be made in the store for the constraint to come to what it
    /// does. Telling only reads, so `looked` then says only that the constraint holds.
    pub(super) then: Option<Then<K>>,
}

/// What a constraint that holds has still to make in the store.
pub(super) enum Then<K>
where
    K: ClassKind,
{
    /// The signatures of the one way through declared functions it goes by.
    Way(passing::Way<K>),
    /// A class with this head, written by the bound, which enters its upper end.
    Enters(Shape<K::Id>),
    /// A copy of the template of each closure of its lower end, a function value.
    Calls,
    /// The signature of the row its receiver goes to.
    Named(passing::Received<K>),
}

impl<K> Told<K>
where
    K: ClassKind,
{
    pub(super) fn done(looked: Looked<K>) -> Self {
        Self {
            looked,
            asks: Vec::new(),
            then: None,
        }
    }
}

/// What the constraint comes to in `view`, by the relation it is.
pub(super) fn tell<K>(view: &View<'_, K>, functions: &functions::Functions, constraint: &Constraint<K>, at: ItemId) -> Told<K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let (lower, upper) = (constraint.lower(), constraint.upper());
    match constraint {
        Constraint::Bound(Bound::Enter { .. }) => enter::look(view, lower, upper, at),
        Constraint::Bound(Bound::Equal { .. }) => equal::look(view, lower, upper, at),
        Constraint::Bound(Bound::Change { .. }) => passing::look(view, functions, lower, upper, at),
        Constraint::Suspended(Suspended::Alone { .. }) => alone::look(view, lower, upper),
        Constraint::Suspended(Suspended::Exit { .. }) => exit::look(view, lower, upper),
        Constraint::Suspended(Suspended::Named { name, .. }) => naming::look(view, functions, *name, lower, upper, at),
    }
}

/// Each part known on `lower`, as one type with the part of its name on `upper`:
/// what holds of two classes one of which enters, or is equal to, the other,
/// whatever heads the two take.
pub(super) fn parts_by_name<K>(classes: &Classes<K>, lower: K::Id, upper: K::Id) -> Vec<Pair<K>>
where
    K: ClassKind,
{
    let known: Vec<(PartKey, K::Id)> = match &classes.get(classes.find(lower)).state {
        State::Decided(decided) => decided.shape.places().into_iter().map(|(key, part)| (key, *part)).collect(),
        State::Undecided(undecided) => undecided.parts.iter().map(|(key, part)| (key.clone(), part.class)).collect(),
    };
    known
        .into_iter()
        .map(|(key, part)| Pair::Part { part, of: upper, key })
        .collect()
}
