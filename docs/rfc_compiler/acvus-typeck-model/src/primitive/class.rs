//! Equivalence classes.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::hash::Hash;

use super::bound::{Bound, BoundInto, BoundOf};
use super::domain::Domain;
use super::id::Id;
use super::refusal::{Cause, Refusal};
use super::scope::Scopes;
use super::ty::{PartKey, Shape};
use super::{ClassId, ItemId, ScopeId, TemplateId};

/// A kind of class, which gives its classes their ids.
///
/// Classes of two kinds are kept in two stores, and an id of one kind names
/// no class of the other:
///
/// ```compile_fail
/// use acvus_typeck_model::primitive::class::{Live, Template};
/// use acvus_typeck_model::primitive::scope::Scopes;
/// use acvus_typeck_model::store::classes::Classes;
///
/// let scopes = Scopes::new();
/// let mut templates = Classes::<Template>::default();
/// let live = Classes::<Live>::default();
/// let hole = templates.undecided(scopes.root());
/// live.find(hole);
/// ```
///
/// A kind is a marker with no value of its own; it asks for the traits the
/// records of its classes derive, as a derive asks them of the kind.
pub trait ClassKind: Copy + Ord + Hash + Debug + Default {
    type Id: Id + Ord + Hash + Debug;
    /// What relates two classes of this kind. A store holds it on those of
    /// the two that are undecided, without reading what it means.
    type Bound: BoundOf<Self>;
}

/// The classes that are solved.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Live;

impl ClassKind for Live {
    type Id = ClassId;
    type Bound = Bound<Self>;
}

/// The classes of a template: a signature or the text of a body, recorded
/// once, never solved, and copied at each use.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Template;

impl ClassKind for Template {
    type Id = TemplateId;
    type Bound = Bound<Self>;
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Class<K>
where
    K: ClassKind,
{
    /// Where two classes are merged, the outer of their scopes (Brand types).
    pub scope: ScopeId,
    pub state: State<K>,
}

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
    Undecided(Constraints<K>),
    Decided(Decided<K>),
}

/// What has been recorded on an undecided class, each with what recorded it.
/// Deciding the class consumes them.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Constraints<K>
where
    K: ClassKind,
{
    /// The bounds this class is an end of.
    pub bounds: Bounds<K>,
    /// The parts projections have named, each waiting for the class's head.
    /// They give the class no head (O7). A name has one class, however many
    /// projections name it on either side.
    pub parts: BTreeMap<PartKey, Part<K>>,
    /// Calls made on the class. A function value has no parts, so a call is
    /// no part, and it is only ever an upper bound.
    pub calls: BTreeSet<(Call<K>, ItemId)>,
    /// Every domain registered. Their intersection is computed (O8).
    pub domains: BTreeSet<(Domain, ItemId)>,
}

/// Bounds, each with the item that recorded it.
///
/// Each end of a bound that is undecided holds it. A bound keeps the ids it was
/// recorded with: they may name classes since merged into others, and both may
/// name the class that holds it, as a class can be its own lower bound. Once no
/// end is undecided, it is given back.
pub type Bounds<K> = BTreeSet<(<K as ClassKind>::Bound, ItemId)>;

/// The side a part is named on.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Side {
    /// It enters the class: a write.
    Lower,
    /// The class goes to it: a read.
    Upper,
}

/// A part of an undecided class, and the bounds that named it.
///
/// A part is added only together with the item that names it, so `lower` and
/// `upper` are never both empty.
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
}

/// `f(ā)`: the class is called with the arguments `ā`, giving `result`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Call<K>
where
    K: ClassKind,
{
    pub args: Vec<K::Id>,
    pub result: K::Id,
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

/// Where a decided class's head comes from (O13). It outlasts the decision, so
/// a hole once decided stays apart from what a declaration writes. An undecided
/// class is free, so only a decided class holds this.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Origin {
    /// Solving or the text decided it. A part that corresponds to it is
    /// unified with it (O9).
    Free,
    /// A declaration or the host writes it. It never changes, so a part that
    /// corresponds to it takes it as an upper bound (O9).
    External,
}

/// What an undecided class held besides its parts and bounds when it became
/// decided, or what was recorded on a class already decided. Nothing in the
/// store judges these; its caller checks each against the class's shape.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Pending<K>
where
    K: ClassKind,
{
    pub class: K::Id,
    pub calls: BTreeSet<(Call<K>, ItemId)>,
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
    pub refusals: Vec<Refusal>,
    pub pending: Option<Pending<K>>,
    /// The bounds an undecided class held when it became decided. Of these,
    /// a bound an undecided class still holds stays with that class; the
    /// rest are given back.
    pub bounds: Bounds<K>,
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
            State::Undecided(constraints) => constraints.parts.get(key).map(|part| part.class),
            State::Decided(decided) => decided.shape.place(key).copied(),
        }
    }

    /// The class of every part.
    pub fn parts(&self) -> Vec<K::Id> {
        match self {
            State::Undecided(constraints) => constraints.parts.values().map(|part| part.class).collect(),
            State::Decided(decided) => decided.shape.places().into_iter().map(|(_, class)| *class).collect(),
        }
    }

    /// The bounds it holds. A decided class holds none.
    pub fn bounds(&self) -> Option<&Bounds<K>> {
        let State::Undecided(constraints) = self else {
            return None;
        };
        Some(&constraints.bounds)
    }

    /// Every id the state holds: its parts, the ends of its bounds, and the
    /// classes of its calls.
    pub fn held(&self) -> Vec<K::Id> {
        let State::Undecided(constraints) = self else {
            return self.parts();
        };
        let bounds = constraints
            .bounds
            .iter()
            .flat_map(|(bound, _)| [bound.lower(), bound.upper()]);
        let calls = constraints
            .calls
            .iter()
            .flat_map(|(call, _)| call.args.iter().copied().chain([call.result]));
        self.parts().into_iter().chain(bounds).chain(calls).collect()
    }
}

impl<K> Class<K>
where
    K: ClassKind,
{
    /// The same class, with every id it holds given by `f`, as a class of the
    /// kind `J`.
    pub fn translate<J, F>(self, mut f: F) -> Class<J>
    where
        J: ClassKind,
        K::Bound: BoundInto<K, J>,
        F: FnMut(K::Id) -> J::Id,
    {
        let state = match self.state {
            State::Decided(decided) => State::Decided(Decided {
                shape: decided.shape.map(|class| f(*class)),
                by: decided.by,
                origin: decided.origin,
            }),
            State::Undecided(constraints) => {
                let bounds = constraints
                    .bounds
                    .into_iter()
                    .map(|(bound, by)| (bound.translate(&mut f), by))
                    .collect();
                let parts = constraints
                    .parts
                    .into_iter()
                    .map(|(key, part)| {
                        let part = Part {
                            class: f(part.class),
                            lower: part.lower,
                            upper: part.upper,
                        };
                        (key, part)
                    })
                    .collect();
                let calls = constraints
                    .calls
                    .into_iter()
                    .map(|(call, by)| {
                        let args = call.args.into_iter().map(&mut f).collect();
                        let call = Call {
                            args,
                            result: f(call.result),
                        };
                        (call, by)
                    })
                    .collect();
                State::Undecided(Constraints {
                    bounds,
                    parts,
                    calls,
                    domains: constraints.domains,
                })
            }
        };
        Class {
            scope: self.scope,
            state,
        }
    }

    /// Makes this class and `other` one class, held at `root`, by `at`.
    ///
    /// The class belongs to the outer of the two scopes. Two undecided
    /// classes hold the constraints of both, and a bound between them
    /// becomes the class's bound on itself. Two decided classes must be of
    /// one layer; where they are not, this class's shape is kept and the
    /// program is refused. An undecided class merged with a decided one
    /// gives each of its parts to the decided one's place for it, and leaves
    /// each part with no place, its bounds, calls and domains to its caller.
    pub fn merge(self, other: Self, root: K::Id, scopes: &Scopes, at: ItemId) -> Merged<K> {
        let scope = scopes.outer(self.scope, other.scope);
        let mut pairs = Vec::new();
        let mut refusals = Vec::new();
        let mut pending = None;
        let mut bounds = BTreeSet::new();
        let mut unplaced = Vec::new();
        let state = match (self.state, other.state) {
            (State::Undecided(mut kept), State::Undecided(gone)) => {
                kept.bounds.extend(gone.bounds);
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
                        }
                    }
                }
                kept.calls.extend(gone.calls);
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
                    refusals.push(Refusal {
                        cause: Cause::Mismatch,
                        items: kept.by.iter().chain(&gone.by).copied().chain([at]).collect(),
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
                bounds = undecided.bounds;
                if !undecided.calls.is_empty() || !undecided.domains.is_empty() {
                    pending = Some(Pending {
                        class: root,
                        calls: undecided.calls,
                        domains: undecided.domains,
                    });
                }
                State::Decided(decided)
            }
        };
        Merged {
            class: Class { scope, state },
            pairs,
            refusals,
            pending,
            bounds,
            unplaced,
        }
    }
}
