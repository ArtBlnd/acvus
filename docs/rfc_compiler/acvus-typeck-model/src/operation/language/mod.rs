//! The language's law: what `TERMS.md` rules, one rule in one place.
//!
//! | The law is asked | The rule | Where |
//! |---|---|---|
//! | what a bound comes to | by the relation the bound is: entering, being equal, being passed | [`enter`], [`equal`], [`passing`] |
//! | which classes are one type by the bounds as a whole | classes that enter one another | [`enter::entering_one_another`] |
//! | where a named part lies | read through a reference | [`Language::place`] |
//! | what head a class may take | the rungs, in order | [`rungs`] |
//! | what is refused at the end | a class still undecided, a class that is its own part | [`rungs::end`] |

pub mod enter;
pub mod equal;
pub mod passing;
pub mod rungs;

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::ItemId;
use crate::primitive::adjustment::Adjustment;
use crate::primitive::bound::{Bound, BoundOf};
use crate::primitive::class::{ClassKind, State};
use crate::primitive::domain::Domain;
use crate::primitive::refusal::Refusal;
use crate::primitive::ty::{PartKey, Shape};
use crate::store::classes::Classes;

use super::law::{Law, Looked, Offer, Pair, Place, Scribe};

/// The rules of the language.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Language;

impl<K> Law<K> for Language
where
    K: ClassKind<Bound = Bound<K>>,
{
    fn look(&self, scribe: &mut Scribe<'_, K>, bound: &Bound<K>, at: ItemId) -> Looked<K> {
        tell(&View::of(scribe.classes()), bound, at).looked
    }

    fn same(&self, part: K::Id, declared: K::Id) -> Bound<K> {
        Bound::Equal {
            lower: part,
            upper: declared,
        }
    }

    fn one(&self, classes: &Classes<K>) -> Vec<(K::Id, K::Id, ItemId)> {
        enter::entering_one_another(classes)
    }

    /// A part named on a reference is the part of that name of what it refers to,
    /// read through it. The target is the one part a reference has a place for.
    fn place(&self, head: &Shape<K::Id>, key: &PartKey) -> Place {
        match head {
            Shape::Ref(..) if *key != PartKey::Target => Place::Within {
                through: PartKey::Target,
                adjustment: Adjustment::Deref,
            },
            _ => Place::Nowhere,
        }
    }

    /// A literal's class is a scalar of the literal's domain.
    fn admits(&self, head: &Shape<K::Id>, domain: &Domain) -> bool {
        matches!(head, Shape::Scalar(scalar) if domain.0.contains(scalar))
    }

    fn rungs(&self) -> usize {
        rungs::COUNT
    }

    fn offer(&self, classes: &Classes<K>, rung: usize, class: K::Id) -> Offer<K> {
        rungs::offer(classes, rung, class)
    }

    fn end(&self, classes: &Classes<K>, class: K::Id) -> Vec<Refusal> {
        rungs::end(classes, class)
    }
}

/// The store as a relation reads it: as it is, or with heads supposed of some
/// undecided classes, to ask what a bound would come to if they took them.
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

/// What a relation tells of a bound: what it comes to, and, where it waits for a
/// class's head, the head it will hold that class's head against.
pub(super) struct Told<K>
where
    K: ClassKind,
{
    pub(super) looked: Looked<K>,
    pub(super) asks: Vec<(K::Id, Shape<K::Id>)>,
}

impl<K> Told<K>
where
    K: ClassKind,
{
    pub(super) fn done(looked: Looked<K>) -> Self {
        Self {
            looked,
            asks: Vec::new(),
        }
    }
}

/// What the bound comes to in `view`, by the relation it is.
pub(super) fn tell<K>(view: &View<'_, K>, bound: &Bound<K>, at: ItemId) -> Told<K>
where
    K: ClassKind<Bound = Bound<K>>,
{
    let (lower, upper) = (bound.lower(), bound.upper());
    match bound {
        Bound::Enter { .. } => enter::look(view, lower, upper, at),
        Bound::Equal { .. } => equal::look(view, lower, upper, at),
        Bound::Change { .. } => passing::look(view, lower, upper, at),
    }
}

/// The items that decided `class`, where it is decided.
pub(super) fn decided_by<K>(classes: &Classes<K>, class: K::Id) -> BTreeSet<ItemId>
where
    K: ClassKind,
{
    match &classes.get(classes.find(class)).state {
        State::Decided(decided) => decided.by.clone(),
        State::Undecided(_) => BTreeSet::new(),
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
        State::Undecided(constraints) => constraints.parts.iter().map(|(key, part)| (key.clone(), part.class)).collect(),
    };
    known
        .into_iter()
        .map(|(key, part)| Pair::Part { part, of: upper, key })
        .collect()
}
