//! Equivalence classes.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::hash::Hash;

use super::constraint::{Constraint, ConstraintOf};
use super::domain::Domain;
use super::id::Id;
use super::refusal::{Cause, Refused};
use super::scope::Scopes;
use super::ty::{PartKey, Shape};
use super::{ClassId, ItemId, ScopeId};

/// A kind of class, which gives its classes their ids.
///
/// A kind is a marker with no value of its own; it asks for the traits the
/// records of its classes derive, as a derive asks them of the kind.
pub trait ClassKind: Copy + Ord + Hash + Debug + Default {
    type Id: Id + Ord + Hash + Debug;
    /// What relates two classes of this kind. A store holds it until it is
    /// finished, without reading what it means.
    type Constraint: ConstraintOf<Self>;
}

/// The classes that are solved.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Live;

impl ClassKind for Live {
    type Id = ClassId;
    type Constraint = Constraint<Self>;
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Class<K>
where
    K: ClassKind,
{
    /// Where two classes are merged, the outer of their scopes.
    pub scope: ScopeId,
    pub state: State<K>,
    /// Every connection that made this class of the classes it was: the two ids
    /// connected, as they were given, and the item it went by. Where two classes
    /// are merged, those of both. Nothing is taken out, and no rule reads it: it
    /// is kept to say, of a refusal, what made the classes meet.
    pub connections: Connections<K>,
}

/// Connections, each the two ids connected and the item it went by.
pub type Connections<K> = BTreeSet<(<K as ClassKind>::Id, <K as ClassKind>::Id, ItemId)>;

/// A class is undecided, holding what has been recorded on it, or decided,
/// holding its shape alone.
///
/// A decided class never changes: a constraint that reaches it afterwards is
/// checked against its shape at once and is not kept.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum State<K>
where
    K: ClassKind,
{
    Undecided(Undecided<K>),
    Decided(Decided<K>),
}

/// What has been recorded on an undecided class, each with what recorded it.
/// Deciding the class consumes them.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Undecided<K>
where
    K: ClassKind,
{
    /// The parts projections have named, each waiting for the class's head.
    /// They give the class no head by themselves. A name has one class, however many
    /// projections name it on either side.
    pub parts: BTreeMap<PartKey, Part<K>>,
    /// Every domain registered. Their intersection is computed.
    pub domains: BTreeSet<(Domain, ItemId)>,
}

/// Constraints, each with the item that recorded it.
///
/// A constraint keeps the ids it was recorded with: they may name classes since
/// merged into others, and both may name one class, as a class can be its own
/// lower bound.
pub type Constraints<K> = BTreeSet<(<K as ClassKind>::Constraint, ItemId)>;

/// How a part came to be named on a class.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Side {
    /// A projection writes into it.
    Lower,
    /// A projection reads it.
    Upper,
    /// A bound carried it here from what enters the class. No projection named
    /// it on this class, so none takes an adjustment for it here.
    Carried,
}

/// A part of an undecided class, and the items that named it.
///
/// A part is added only together with the item that names it, so `lower`,
/// `upper` and `carried` are never all empty.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Part<K>
where
    K: ClassKind,
{
    pub class: K::Id,
    /// Items that named the part as a lower bound: a write into it.
    pub lower: BTreeSet<ItemId>,
    /// Items that named the part as an upper bound: a read of it.
    pub upper: BTreeSet<ItemId>,
    /// Bounds that carried the name here.
    pub carried: BTreeSet<ItemId>,
}

/// A decided class: its constructor and members, and a class at each part.
///
/// A part is a class of its own, decided or not, so a decided class may hold
/// undecided ones: a signature's hole is an undecided class its decided
/// classes hold as a part.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Decided<K>
where
    K: ClassKind,
{
    pub shape: Shape<K::Id>,
    /// The items that decided it, for a refusal to name.
    pub by: BTreeSet<ItemId>,
    pub origin: Origin,
}

/// Where a decided class's head comes from. It outlasts the decision, so
/// a hole once decided stays apart from what a declaration writes. An undecided
/// class is free, so only a decided class holds this.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Origin {
    /// Solving or the text decided it. A part that corresponds to it is
    /// unified with it.
    Free,
    /// A declaration or the host writes it. It never changes, so a part that
    /// corresponds to it is compared with it.
    External,
}

/// The domains an undecided class held when it became decided, or one
/// registered on a class already decided. Nothing in the store judges these; its
/// caller checks each against the class's shape.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Pending<K>
where
    K: ClassKind,
{
    pub class: K::Id,
    pub domains: BTreeSet<(Domain, ItemId)>,
}

/// A part named on a class whose shape has no place for it. Nothing in the
/// store judges this: it is its caller's to refuse, or to take elsewhere.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Unplaced<K>
where
    K: ClassKind,
{
    pub class: K::Id,
    pub key: PartKey,
    pub part: Part<K>,
    /// The item that decided the class.
    pub at: ItemId,
}

/// Two classes made one, and what that leaves to whoever merged them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Merged<K>
where
    K: ClassKind,
{
    pub class: Class<K>,
    /// Pairs of parts of one name, each of which must be one class in turn.
    pub pairs: Vec<(K::Id, K::Id)>,
    pub refusals: Vec<Refused<K>>,
    pub pending: Option<Pending<K>>,
    pub unplaced: Vec<Unplaced<K>>,
}

impl<K> State<K>
where
    K: ClassKind,
{
    pub fn is_decided(&self) -> bool {
        matches!(self, State::Decided(_))
    }

    /// The class of the part `key` names, where the state has one: the
    /// shape's place for it, or the class the bounds of an undecided class
    /// have named it by.
    pub fn part(&self, key: &PartKey) -> Option<K::Id> {
        match self {
            State::Undecided(undecided) => undecided.parts.get(key).map(|part| part.class),
            State::Decided(decided) => decided.shape.place(key).copied(),
        }
    }

    /// The class of every part.
    pub fn parts(&self) -> Vec<K::Id> {
        match self {
            State::Undecided(undecided) => undecided.parts.values().map(|part| part.class).collect(),
            State::Decided(decided) => decided.shape.places().into_iter().map(|(_, class)| *class).collect(),
        }
    }
}

impl<K> Class<K>
where
    K: ClassKind,
{
    /// The same class, with every id it holds given by `f`.
    pub fn renamed<F>(self, mut f: F) -> Self
    where
        F: FnMut(K::Id) -> K::Id,
    {
        let state = match self.state {
            State::Decided(decided) => State::Decided(Decided {
                shape: decided.shape.map(|class| f(*class)),
                by: decided.by,
                origin: decided.origin,
            }),
            State::Undecided(undecided) => {
                let parts = undecided
                    .parts
                    .into_iter()
                    .map(|(key, part)| {
                        let part = Part {
                            class: f(part.class),
                            lower: part.lower,
                            upper: part.upper,
                            carried: part.carried,
                        };
                        (key, part)
                    })
                    .collect();
                State::Undecided(Undecided {
                    parts,
                    domains: undecided.domains,
                })
            }
        };
        Class {
            scope: self.scope,
            state,
            connections: self.connections.into_iter().map(|(a, b, at)| (f(a), f(b), at)).collect(),
        }
    }

    /// Makes this class and `other` one class, held at `root`, by `at`.
    ///
    /// The class belongs to the outer of the two scopes. Two undecided
    /// classes hold the constraints of both. Two decided classes must be of
    /// one layer; where they are not, this class's shape is kept and the
    /// program is refused. An undecided class merged with a decided one
    /// gives each of its parts to the decided one's place for it, and leaves
    /// each part with no place, its calls and domains to its caller.
    pub fn merge(self, other: Self, root: K::Id, scopes: &Scopes, at: ItemId) -> Merged<K> {
        let scope = scopes.outer(self.scope, other.scope);
        let mut connections = self.connections;
        connections.extend(other.connections);
        let mut pairs = Vec::new();
        let mut refusals = Vec::new();
        let mut pending = None;
        let mut unplaced = Vec::new();
        let state = match (self.state, other.state) {
            (State::Undecided(mut kept), State::Undecided(gone)) => {
                for (key, part) in gone.parts {
                    match kept.parts.entry(key) {
                        Entry::Vacant(vacant) => {
                            vacant.insert(part);
                        }
                        Entry::Occupied(mut occupied) => {
                            let mine = occupied.get_mut();
                            pairs.push((mine.class, part.class));
                            mine.lower.extend(part.lower);
                            mine.upper.extend(part.upper);
                            mine.carried.extend(part.carried);
                        }
                    }
                }
                kept.domains.extend(gone.domains);
                State::Undecided(kept)
            }
            (State::Decided(kept), State::Decided(gone)) => {
                if kept.shape.layer() == gone.shape.layer() {
                    for ((_, mine), (_, theirs)) in kept.shape.places().into_iter().zip(gone.shape.places()) {
                        pairs.push((*mine, *theirs));
                    }
                    let mut by = kept.by;
                    by.extend(gone.by);
                    // What a declaration writes never changes, whatever it is
                    // made one with.
                    let origin = kept.origin.max(gone.origin);
                    State::Decided(Decided {
                        shape: kept.shape,
                        by,
                        origin,
                    })
                } else {
                    // The head that is not kept is gone with its class, so what
                    // decided it is said here.
                    refusals.push(Refused {
                        cause: Cause::Mismatch,
                        items: gone.by.iter().copied().chain([at]).collect(),
                        of: vec![root],
                    });
                    State::Decided(kept)
                }
            }
            (State::Undecided(undecided), State::Decided(decided))
            | (State::Decided(decided), State::Undecided(undecided)) => {
                for (key, part) in undecided.parts {
                    match decided.shape.place(&key) {
                        Some(&place) => pairs.push((place, part.class)),
                        None => unplaced.push(Unplaced {
                            class: root,
                            key,
                            part,
                            at,
                        }),
                    }
                }
                if !undecided.domains.is_empty() {
                    pending = Some(Pending {
                        class: root,
                        domains: undecided.domains,
                    });
                }
                State::Decided(decided)
            }
        };
        Merged {
            class: Class {
                scope,
                state,
                connections,
            },
            pairs,
            refusals,
            pending,
            unplaced,
        }
    }
}
