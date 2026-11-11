//! Entering: a value's class is a lower bound of the class it meets in.
//!
//! The class's head is the join of every head that enters it. A join is not
//! subtyping: it merges bounds, so objects that meet have the fields of both.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::ItemId;
use crate::primitive::adjustment::Adjustment;
use crate::primitive::components::components;
use crate::primitive::class::ClassKind;
use crate::primitive::constraint::{Bound, Constraint, Suspended};
use crate::primitive::adjustment::Subtyping;
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::{Mutability, Shape};
use crate::store::classes::Classes;

use super::super::law::Looked;
use super::{Told, View, parts_by_name};

/// What `lower` entering `upper` comes to.
///
/// The parts of one name on the two are one type from the start. Once both have
/// heads, the lower one must be a head the join took in: its join with the upper
/// head is the upper head. An exclusive reference that enters a shared one is
/// reborrowed.
pub(super) fn look<K>(view: &View<'_, K>, lower: K::Id, upper: K::Id, at: ItemId) -> Told<K>
where
    K: ClassKind,
{
    let classes = view.classes;
    let pairs = parts_by_name(classes, lower, upper);
    let (Some(entering), Some(head)) = (view.head(lower), view.head(upper)) else {
        // What will enter a head already there is held against that head.
        let asks = view.head(upper).map(|head| (lower, head.clone())).into_iter().collect();
        let looked = Looked::Waits {
            on: BTreeSet::from([lower, upper]),
            apart: BTreeSet::new(),
            pairs,
        };
        return Told { looked, asks, then: None };
    };
    if !entering.enters(head) {
        return Told::done(Looked::Refused(Refused {
            cause: Cause::Mismatch,
            items: BTreeSet::from([at]),
            of: vec![lower, upper],
        }));
    }
    let reborrowed = matches!(
        (entering, head),
        (Shape::Ref(Mutability::Exclusive, _), Shape::Ref(Mutability::Shared, _))
    );
    Told::done(Looked::Holds {
        pairs,
        constraints: Vec::new(),
        adjustments: reborrowed.then_some(Adjustment::Subtype(Subtyping::Reborrow)).into_iter().collect(),
    })
}

/// The head `constraint` gives its upper end to be joined: none where nothing enters
/// by it, and none yet where what enters is not decided, at every depth where
/// the constraint asks that.
pub(super) fn given<K>(view: &View<'_, K>, constraint: &Constraint<K>) -> Option<Option<Shape<K::Id>>>
where
    K: ClassKind,
{
    match constraint {
        Constraint::Bound(Bound::Enter { lower, .. }) => Some(view.head(*lower).cloned()),
        Constraint::Suspended(Suspended::Alone { .. } | Suspended::Exit { .. } | Suspended::Named { .. }) => Some(None),
        Constraint::Bound(Bound::Equal { .. } | Bound::Change { .. }) => None,
    }
}

/// The two ends of each entering that lies on a cycle of enterings: classes that
/// enter one another would each wait for the other's head, and are one type.
///
/// An entering lies on a cycle where its two ends reach one another, which is
/// where they are of one strongly connected component of the enterings.
pub fn entering_one_another<K>(classes: &Classes<K>) -> Vec<(K::Id, K::Id, ItemId)>
where
    K: ClassKind<Constraint = Constraint<K>>,
{
    let mut enterings = Vec::new();
    let mut entered: BTreeMap<K::Id, Vec<K::Id>> = BTreeMap::new();
    for (constraint, at) in classes.constraints() {
        if let Constraint::Bound(Bound::Enter { lower, upper }) = constraint {
            let (lower, upper) = (classes.find(*lower).id(), classes.find(*upper).id());
            if lower != upper {
                enterings.push((lower, upper, *at));
                entered.entry(lower).or_default().push(upper);
            }
        }
    }
    let component = components(&entered);
    enterings
        .into_iter()
        .filter(|(lower, upper, _)| component.get(lower).is_some_and(|of_lower| component.get(upper) == Some(of_lower)))
        .collect()
}
