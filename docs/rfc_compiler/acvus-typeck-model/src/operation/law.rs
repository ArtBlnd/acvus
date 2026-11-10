//! The contracts between the three operations and the rules they ask.
//!
//! A law reads the store and answers. It merges nothing, decides nothing and
//! finishes nothing: the operations do those, from what it answers. So a rule
//! cannot leave a bound half done, and every rule is one answer in one place.

use std::collections::BTreeSet;

use crate::primitive::ItemId;
use crate::primitive::adjustment::Adjustment;
use crate::primitive::class::ClassKind;
use crate::primitive::domain::Domain;
use crate::primitive::refusal::Refusal;
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{PartKey, Shape};
use crate::primitive::ScopeId;
use crate::store::classes::Classes;

/// Two classes that are one type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Pair<K>
where
    K: ClassKind,
{
    Classes(K::Id, K::Id),
    /// `part` is one type with the part `key` of `of`. Where `of` is undecided it
    /// is given the name, and where it is decided with no place for it there is
    /// nothing to connect.
    Part { part: K::Id, of: K::Id, key: PartKey },
}

/// What a bound comes to, as its two ends stand.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Looked<K>
where
    K: ClassKind,
{
    /// It holds, and is finished. These pairs are one type by it, and these
    /// adjustments are what it takes.
    Holds { pairs: Vec<Pair<K>>, adjustments: Vec<Adjustment> },
    /// It does not hold, and is finished.
    Refused(Refusal),
    /// It cannot be told yet. It is looked at again when a class of `on` is
    /// merged or decided. Until it is finished it may still merge what lies below
    /// each class of `apart`, though not those classes themselves, so none of
    /// what lies there is decided. `pairs` are one type whatever it comes to.
    Waits {
        on: BTreeSet<K::Id>,
        apart: BTreeSet<K::Id>,
        pairs: Vec<Pair<K>>,
    },
}

/// What a rung offers an undecided class.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Offer<K>
where
    K: ClassKind,
{
    Nothing,
    /// This head. Only its constructor and members are taken: the parts of the
    /// head decided are the class's own.
    Head { shape: Shape<K::Id>, by: BTreeSet<ItemId> },
    /// The class cannot take a head by this rung, and the program is refused.
    Refused(Refusal),
}

/// Where a part named on a class lies, once the class is decided as a head that
/// has no place of that name.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Place {
    /// It is the part of that name of the head's part `through`, and each
    /// projection that named it takes `adjustment`.
    Within { through: PartKey, adjustment: Adjustment },
    Nowhere,
}

/// The store as a law may use it while it looks at a bound: all of it to read,
/// and classes to make anew, as a signature made at a boundary needs. It merges
/// and decides nothing.
#[derive(Debug)]
pub struct Scribe<'a, K>
where
    K: ClassKind,
{
    classes: &'a mut Classes<K>,
    scopes: &'a Scopes,
}

impl<'a, K> Scribe<'a, K>
where
    K: ClassKind,
{
    pub(super) fn new(classes: &'a mut Classes<K>, scopes: &'a Scopes) -> Self {
        Self { classes, scopes }
    }

    pub fn classes(&self) -> &Classes<K> {
        self.classes
    }

    pub fn scopes(&self) -> &Scopes {
        self.scopes
    }

    pub fn undecided(&mut self, scope: ScopeId) -> K::Id {
        self.classes.undecided(scope)
    }
}

/// The rules solving asks.
pub trait Law<K>
where
    K: ClassKind,
{
    /// What `bound`, recorded by `at`, comes to as its ends stand.
    fn look(&self, scribe: &mut Scribe<'_, K>, bound: &K::Bound, at: ItemId) -> Looked<K>;

    /// The bound that says `part` has the head of `declared`, a class a
    /// declaration writes. Such a class never changes, so nothing is merged with
    /// it, and what is one type with it is compared with it by this bound.
    fn same(&self, part: K::Id, declared: K::Id) -> K::Bound;

    /// The classes that are one type by how the bounds stand as a whole, and by
    /// no one bound, each pair with the item it goes by.
    fn one(&self, classes: &Classes<K>) -> Vec<(K::Id, K::Id, ItemId)>;

    /// Where the part `key` lies on a class decided as `head`, which has no place
    /// of that name.
    fn place(&self, head: &Shape<K::Id>, key: &PartKey) -> Place;

    /// Whether a class a literal registered `domain` in may be decided as `head`.
    fn admits(&self, head: &Shape<K::Id>, domain: &Domain) -> bool;

    /// How many rungs deciding takes, the first being rung 0. A rung is taken
    /// only where no rung before it decides anything.
    fn rungs(&self) -> usize;

    /// What the rung `rung` offers the undecided class `class`.
    fn offer(&self, classes: &Classes<K>, rung: usize, class: K::Id) -> Offer<K>;

    /// What is refused of `class` when nothing more can be done.
    fn end(&self, classes: &Classes<K>, class: K::Id) -> Vec<Refusal>;
}
