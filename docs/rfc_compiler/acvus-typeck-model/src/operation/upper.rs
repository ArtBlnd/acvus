//! What upper bounds give a class that nothing enters (O19).
//!
//! A class's upper bounds are the types given it, the classes it enters, and the
//! upper bounds of each class whose class node its head is in: each it enters,
//! and each reference whose target it is. So an upper bound is reached either at
//! the class's own place or at the target's place of a reference.
//!
//! The heads the upper bounds give are candidates. The class takes the one of
//! them, in its domain, that goes to every upper bound. It goes to a boundary by a
//! coercion alone: no conversion chooses a head. Where upper bounds give
//! heads and no one head goes to them all, the class waits: it has information,
//! and no fallback may overrule it (O17).

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::ItemId;
use crate::primitive::bound::{Bound, BoundOf};
use crate::primitive::class::{Bounds, ClassKind, State};
use crate::primitive::domain::Domain;
use crate::primitive::ty::{Mutability, Shape};
use crate::store::classes::Classes;

use super::Outcome;
use super::bound::{given, shape_of};
use super::change::may_coerce;

/// Where a class's head stands in the class node an upper bound is reached
/// through.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Through {
    /// The class's own class node.
    Own,
    /// A reference of this mutability, whose target the class is. What the class
    /// node has of its mutability is kept, and the class's head fills in the
    /// target's constructor.
    Target(Mutability),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum How {
    Enters,
    Changes,
    Equals,
}

/// One upper bound, as a class reaches it.
struct Upper<I> {
    how: How,
    through: Through,
    /// The head the upper end gives, where it gives one: its own, or the join of
    /// what has entered it so far.
    gives: Option<Shape<I>>,
    by: ItemId,
}

/// What the upper bounds of a class give it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Given<K>
where
    K: ClassKind,
{
    /// No upper bound gives a head, so another dimension decides the class.
    Nothing,
    /// Upper bounds give heads, and no one head goes to them all. The class
    /// waits, as a later fixpoint may leave one.
    NoOneHead,
    Head {
        shape: Shape<K::Id>,
        by: BTreeSet<ItemId>,
    },
}

/// The bounds that matter here by the root of their lower end, and each reference
/// by the root of its target. A change waiting for a class node is held by no
/// class once both its ends are decided, so it comes from `waiting`.
struct Index<K>
where
    K: ClassKind,
{
    going_up: BTreeMap<K::Id, Bounds<K>>,
    references: BTreeMap<K::Id, Vec<(K::Id, Mutability)>>,
}

impl<K> Index<K>
where
    K: ClassKind<Bound = Bound<K>>,
{
    fn new(classes: &Classes<K>, waiting: &Bounds<K>) -> Self {
        let mut going_up: BTreeMap<K::Id, Bounds<K>> = BTreeMap::new();
        let mut references: BTreeMap<K::Id, Vec<(K::Id, Mutability)>> = BTreeMap::new();
        let mut hold = |bound: &Bound<K>, at: ItemId| {
            going_up
                .entry(classes.find(bound.lower()).id())
                .or_default()
                .insert((*bound, at));
        };
        for (bound, at) in waiting {
            hold(bound, *at);
        }
        for root in classes.roots() {
            match &classes.get(root).state {
                State::Undecided(constraints) => {
                    for (bound, at) in &constraints.bounds {
                        hold(bound, *at);
                    }
                }
                State::Decided(decided) => {
                    if let Shape::Ref(mutability, target) = decided.shape {
                        references
                            .entry(classes.find(target).id())
                            .or_default()
                            .push((root.id(), mutability));
                    }
                }
            }
        }
        Self { going_up, references }
    }

    /// Every upper bound of `class`.
    fn uppers(&self, classes: &Classes<K>, class: K::Id) -> Vec<Upper<K::Id>> {
        let root = classes.find(class).id();
        let mut stack = vec![(root, Through::Own)];
        for (reference, mutability) in self.references.get(&root).into_iter().flatten() {
            stack.push((*reference, Through::Target(*mutability)));
        }
        let mut seen = BTreeSet::new();
        let mut found = Vec::new();
        while let Some((at_class, through)) = stack.pop() {
            if !seen.insert((at_class, through)) {
                continue;
            }
            for (bound, by) in self.going_up.get(&at_class).into_iter().flatten() {
                let upper = classes.find(bound.upper()).id();
                let reached = |how, gives| Upper {
                    how,
                    through,
                    gives,
                    by: *by,
                };
                match bound {
                    Bound::Enter { .. } if upper == at_class => {}
                    Bound::Enter { .. } => {
                        found.push(reached(How::Enters, join_given(classes, upper)));
                        stack.push((upper, through));
                    }
                    Bound::Change { .. } => found.push(reached(How::Changes, shape_of(classes, upper).cloned())),
                    Bound::Equal { .. } => found.push(reached(How::Equals, shape_of(classes, upper).cloned())),
                    Bound::Lend { mutability, .. } => {
                        let of_reference = match through {
                            Through::Own => *mutability,
                            Through::Target(already) => already.join(*mutability),
                        };
                        stack.push((upper, Through::Target(of_reference)));
                    }
                }
            }
        }
        found
    }
}

/// The head a class gives what enters it: its own, or the join of the heads its
/// lower bounds have given so far, where they join.
fn join_given<K>(classes: &Classes<K>, class: K::Id) -> Option<Shape<K::Id>>
where
    K: ClassKind<Bound = Bound<K>>,
{
    if let Some(shape) = shape_of(classes, class) {
        return Some(shape.clone());
    }
    let mut heads = classes
        .lowers(classes.find(class))
        .into_values()
        .flatten()
        .filter_map(|(bound, _)| given(classes, &bound));
    let first = heads.next()?;
    heads.try_fold(first, |joined, head| joined.join(&head))
}

/// The head `upper` gives the class that reaches it.
fn offered<K>(classes: &Classes<K>, upper: &Upper<K::Id>) -> Option<Shape<K::Id>>
where
    K: ClassKind,
{
    match (upper.through, upper.gives.as_ref()?) {
        (Through::Own, gives) => Some(gives.clone()),
        (Through::Target(_), Shape::Ref(_, target)) => shape_of(classes, *target).cloned(),
        (Through::Target(_), _) => None,
    }
}

/// Whether `head`, filling in what the class node `upper` is reached through
/// leaves open, goes to `upper`.
fn goes_to<K>(classes: &Classes<K>, head: &Shape<K::Id>, upper: &Upper<K::Id>) -> bool
where
    K: ClassKind,
{
    let Some(to) = &upper.gives else {
        return true;
    };
    let within = match upper.through {
        Through::Own => None,
        Through::Target(mutability) => Some(mutability),
    };
    match (upper.how, within) {
        (How::Changes, within) => may_coerce(classes, head, within, to),
        (How::Enters, None) => head.enters(to),
        (How::Equals, None) => head.layer() == to.layer(),
        (How::Enters, Some(mutability)) => matches!(
            to,
            Shape::Ref(into, target) if mutability.join(*into) == *into
                && shape_of(classes, *target).is_none_or(|target| head.enters(target))
        ),
        (How::Equals, Some(mutability)) => matches!(
            to,
            Shape::Ref(into, target) if mutability == *into
                && shape_of(classes, *target).is_none_or(|target| head.layer() == target.layer())
        ),
    }
}

impl<K> Index<K>
where
    K: ClassKind<Bound = Bound<K>>,
{
    /// What the upper bounds of `class` give it.
    fn given(&self, classes: &Classes<K>, class: K::Id) -> Given<K> {
        let uppers = self.uppers(classes, class);
        let mut candidates: BTreeMap<Shape<()>, Shape<K::Id>> = BTreeMap::new();
        for upper in &uppers {
            if let Some(head) = offered(classes, upper) {
                candidates.entry(head.layer()).or_insert(head);
            }
        }
        if candidates.is_empty() {
            return Given::Nothing;
        }
        let domain = match &classes.get(classes.find(class)).state {
            State::Undecided(constraints) => Domain::intersection(constraints.domains.iter().map(|(domain, _)| domain)),
            State::Decided(_) => None,
        };
        let mut heads = candidates.into_values().filter(|head| {
            let in_domain = domain
                .as_ref()
                .is_none_or(|domain| matches!(head, Shape::Scalar(scalar) if domain.0.contains(scalar)));
            in_domain && uppers.iter().all(|upper| goes_to(classes, head, upper))
        });
        match (heads.next(), heads.next()) {
            (Some(shape), None) => Given::Head {
                shape,
                by: uppers.iter().map(|upper| upper.by).collect(),
            },
            _ => Given::NoOneHead,
        }
    }
}

/// Each of `classes` as its upper bounds find it.
pub(super) fn given_by_uppers<K>(classes: &Classes<K>, waiting: &Bounds<K>, visit: Vec<K::Id>) -> Vec<(K::Id, Given<K>)>
where
    K: ClassKind<Bound = Bound<K>>,
{
    let index = Index::new(classes, waiting);
    visit
        .into_iter()
        .map(|class| (class, index.given(classes, class)))
        .collect()
}

impl<K> Outcome<K>
where
    K: ClassKind<Bound = Bound<K>>,
{
    /// Gives `class` the head `shape`, with a class of its own at each part: what
    /// an upper bound gives is a constructor and members, and the parts are what
    /// a boundary or a part will connect once there is something to connect.
    pub(super) fn decide_with_new_parts(
        &mut self,
        classes: &mut Classes<K>,
        scopes: &crate::primitive::scope::Scopes,
        class: K::Id,
        shape: &Shape<K::Id>,
        by: BTreeSet<ItemId>,
        at: ItemId,
    ) {
        let scope = classes.get(classes.find(class)).scope;
        let shape = shape.map(|_| classes.undecided(scope));
        self.decide(classes, scopes, class, shape, by, at);
    }
}
