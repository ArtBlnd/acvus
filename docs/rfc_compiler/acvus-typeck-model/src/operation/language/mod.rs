//! The language's law: what `TERMS.md` rules, one rule in one place.
//!
//! | The law is asked | The rule | Where |
//! |---|---|---|
//! | what a constraint comes to | by the relation the constraint is: entering, being equal, being passed, `as _`, the exit of `?` | [`enter`], [`equal`], [`passing`], [`alone`], [`exit`] |
//! | which functions are called by name | a table of closures, and no rule | [`functions`] |
//! | what a function value passed to a function type comes to | the body of each closure made anew | [`calling`] |
//! | which bodies await a use | the body of each closure | [`Language::awaiting`] |
//! | which function a call by name calls | the one function of that name the receiver goes to | [`naming`] |
//! | what calling a function of a module comes to | the module used, and the function called | [`units`] |
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
pub mod naming;
pub mod passing;
pub mod rungs;
pub mod units;

use std::collections::BTreeMap;

use crate::primitive::{ItemId, Name, ScopeId};
use crate::primitive::adjustment::Adjustment;
use crate::primitive::constraint::{Bound, Constraint, ConstraintOf, Suspended};
use crate::primitive::class::{ClassKind, State};
use crate::primitive::domain::Domain;
use crate::primitive::refusal::Refused;
use crate::primitive::ty::{PartKey, Shape};
use crate::store::Store;
use crate::store::classes::Classes;
use crate::store::closures::Closures;

use super::law::{Bodies, Heads, Law, Looked, Pair, Place, Scribe};

/// The rules of the language, with the functions declared from outside, and the
/// structs the text writes.
#[derive(Clone, Debug, Default)]
pub struct Language {
    pub functions: functions::Functions,
    pub units: units::Units,
    /// The fields of each struct, in the order it declares them: the type of the
    /// field at a place is the struct's type parameter at that place.
    pub structs: BTreeMap<Name, Vec<Name>>,
}

impl<K> Law<K> for Language
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    fn look(&self, scribe: &mut Scribe<'_, K>, constraint: &Constraint<K>, at: ItemId) -> Looked<K> {
        let told = tell(&View::of(scribe.classes(), scribe.closures()), &self.functions, constraint, at);
        match told.then {
            Some(Then::Way(way)) => passing::take(scribe, &self.functions, way, constraint.lower(), at),
            Some(Then::Calls(wanted)) => calling::take(scribe, constraint.lower(), wanted, at),
            Some(Then::Named(received)) => naming::take(scribe, &self.functions, &self.units, received, constraint.lower(), constraint.upper(), at),
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

    fn admits(&self, head: &Shape<K::Id>, domain: &Domain) -> bool {
        admits(head, domain)
    }

    fn heads(&self, classes: &Classes<K>, bodies: &Bodies<'_, K>, asked: &[K::Id], held: &dyn Fn(K::Id) -> bool) -> Heads<K> {
        rungs::heads(classes, bodies, &self.functions, asked, held)
    }

    fn fixpoint(&self, classes: &Classes<K>, bodies: &Bodies<'_, K>, asked: &[K::Id], held: &dyn Fn(K::Id) -> bool) -> Heads<K> {
        rungs::fixpoint(classes, bodies, &self.functions, asked, held)
    }

    fn end(&self, classes: &Classes<K>, bodies: &Bodies<'_, K>, class: K::Id) -> Vec<Refused<K>> {
        rungs::end(classes, bodies, &self.functions, class)
    }

    /// The body of each closure: a call gives its parameters what they lack, to
    /// a body made anew.
    fn awaiting(&self, store: &Store<K>) -> Vec<ScopeId> {
        store.closures.iter().map(|(_, closure)| closure.body).collect()
    }
}

/// Whether a class a literal registered `domain` in may have the head `head`: a
/// literal's class is a scalar of the literal's domain.
pub(super) fn admits<P>(head: &Shape<P>, domain: &Domain) -> bool {
    matches!(head, Shape::Scalar(scalar) if domain.0.contains(scalar))
}

/// The store as a relation reads it: its classes and the closures named of its
/// bodies, as they are, or with heads supposed of some undecided classes, to
/// ask what a constraint would come to if they took them.
#[derive(Clone, Debug)]
pub(super) struct View<'a, K>
where
    K: ClassKind,
{
    pub(super) classes: &'a Classes<K>,
    pub(super) closures: &'a Closures<K>,
    supposed: BTreeMap<K::Id, Shape<K::Id>>,
}

impl<'a, K> View<'a, K>
where
    K: ClassKind,
{
    pub(super) fn of(classes: &'a Classes<K>, closures: &'a Closures<K>) -> Self {
        Self {
            classes,
            closures,
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

impl<K> View<'_, K>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    /// The class a declaration writes that `class` is bound to be one type with,
    /// by a bound not finished yet.
    ///
    /// Nothing is merged with a declared class: what is one type with it is
    /// bound to it instead, and takes its head where classes are next decided.
    /// Until then it has no head, and is not a class nothing is said of. An
    /// `Equal` is recorded by [`Law::same`] alone, with the declared class as its
    /// upper end; the declared class is an end of it too, and is bound to none.
    pub(super) fn declared_for(&self, class: K::Id) -> Option<K::Id> {
        let root = self.classes.find(class).id();
        self.classes.constraints_of(class).find_map(|(constraint, _)| match constraint {
            Constraint::Bound(Bound::Equal { lower, upper }) if self.classes.find(*lower).id() == root => Some(*upper),
            _ => None,
        })
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
    /// The call its lower end, a function, is used by: it is passed to this
    /// class, a function type.
    Calls(K::Id),
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
