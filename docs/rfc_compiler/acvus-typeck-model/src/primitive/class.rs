//! Equivalence classes.

use std::collections::{BTreeMap, BTreeSet};

use super::domain::Domain;
use super::ty::{PartKey, Shape};
use super::{ClassId, ItemId, ScopeId};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Class {
    /// Where two classes are unified, the outer of their scopes (Brand types).
    pub scope: ScopeId,
    pub state: State,
}

/// A class is undecided, holding what has been recorded on it, or decided,
/// holding its shape alone.
///
/// A decided class never changes: a constraint that reaches it afterwards is
/// checked against its shape at once and is not kept.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum State {
    Undecided(Constraints),
    Decided(Decided),
}

/// What has been recorded on an undecided class, each with what recorded it.
/// Deciding the class consumes them.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Constraints {
    /// Type bounds: the classes that enter this one.
    pub lower: Bounds,
    /// Type bounds: the classes this one goes to.
    pub upper: Bounds,
    /// Structural bounds: the parts named without a constructor. A name has
    /// one class, however many bounds name it on either side.
    pub parts: BTreeMap<PartKey, Part>,
    /// Calls made on the class. A function value has no parts, so a call is
    /// no part, and it is only ever an upper bound.
    pub calls: BTreeSet<(Call, ItemId)>,
    /// Every domain registered. Their intersection is computed (O8).
    pub domains: BTreeSet<(Domain, ItemId)>,
}

/// The classes on one side of a class, each with the items that recorded the
/// bound.
///
/// A key is the id the bound was recorded with: it may name a class since
/// unified into another, and two keys may name one class. It may name the
/// class itself, as a class can be its own lower bound.
pub type Bounds = BTreeMap<ClassId, BTreeSet<ItemId>>;

/// A type bound: the class `lower` enters the class `upper`.
///
/// Each of its two classes that is undecided holds it, the lower one among
/// its upper bounds and the upper one among its lower bounds. Once neither is
/// undecided, it is checked.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Bound {
    pub lower: ClassId,
    pub upper: ClassId,
}

/// The side of a class a bound is on.
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
pub struct Part {
    pub class: ClassId,
    /// Items that named the part as a lower bound: a write into it.
    pub lower: BTreeSet<ItemId>,
    /// Items that named the part as an upper bound: a read of it.
    pub upper: BTreeSet<ItemId>,
}

/// `f(ā)`: the class is called with the arguments `ā`, giving `result`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Call {
    pub args: Vec<ClassId>,
    pub result: ClassId,
}

/// A decided class: its constructor and members, and a class at each part.
///
/// A part is a class of its own, decided or not, so a decided class may hold
/// undecided ones: a signature's hole is an undecided class its decided
/// classes hold as a part.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Decided {
    pub shape: Shape<ClassId>,
    /// The items that decided it, for a refusal to name.
    pub by: BTreeSet<ItemId>,
}
