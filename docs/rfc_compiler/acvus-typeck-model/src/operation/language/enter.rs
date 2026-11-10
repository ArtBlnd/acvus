//! Entering: a value's class is a lower bound of the class it meets in.
//!
//! The class's head is the join of every head that enters it. A join is not
//! subtyping: it merges bounds, so objects that meet have the fields of both.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::ItemId;
use crate::primitive::adjustment::Adjustment;
use crate::primitive::bound::Bound;
use crate::primitive::class::ClassKind;
use crate::primitive::adjustment::Subtyping;
use crate::primitive::refusal::{Cause, Refusal};
use crate::primitive::ty::{Mutability, Shape};
use crate::store::classes::Classes;

use super::super::law::Looked;
use super::{Told, View, decided_by, parts_by_name};

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
        return Told { looked, asks, way: None };
    };
    if !entering.enters(head) {
        let mut items = decided_by(classes, lower);
        items.extend(decided_by(classes, upper));
        items.insert(at);
        return Told::done(Looked::Refused(Refusal {
            cause: Cause::Mismatch,
            items,
        }));
    }
    let reborrowed = matches!(
        (entering, head),
        (Shape::Ref(Mutability::Exclusive, _), Shape::Ref(Mutability::Shared, _))
    );
    Told::done(Looked::Holds {
        pairs,
        bounds: Vec::new(),
        adjustments: reborrowed.then_some(Adjustment::Subtype(Subtyping::Reborrow)).into_iter().collect(),
    })
}

/// The head `bound` gives its upper end to be joined: none where nothing enters
/// by it, and none yet where what enters is not decided, at every depth where
/// the bound asks that.
pub(super) fn given<K>(view: &View<'_, K>, bound: &Bound<K>) -> Option<Option<Shape<K::Id>>>
where
    K: ClassKind,
{
    match bound {
        Bound::Enter { lower, .. } => Some(view.head(*lower).cloned()),
        Bound::Alone { .. } => Some(None),
        Bound::Equal { .. } | Bound::Change { .. } => None,
    }
}

/// The two ends of each entering that lies on a cycle of enterings: classes that
/// enter one another would each wait for the other's head, and are one type.
pub fn entering_one_another<K>(classes: &Classes<K>) -> Vec<(K::Id, K::Id, ItemId)>
where
    K: ClassKind<Bound = Bound<K>>,
{
    let mut entered: BTreeMap<K::Id, BTreeSet<(K::Id, ItemId)>> = BTreeMap::new();
    for (bound, at) in classes.bounds() {
        if let Bound::Enter { lower, upper } = bound {
            let (lower, upper) = (classes.find(*lower).id(), classes.find(*upper).id());
            if lower != upper {
                entered.entry(lower).or_default().insert((upper, *at));
            }
        }
    }
    let reaches = |from: K::Id, to: K::Id| {
        let mut seen = BTreeSet::new();
        let mut stack = vec![from];
        while let Some(class) = stack.pop() {
            if class == to {
                return true;
            }
            if seen.insert(class) {
                stack.extend(entered.get(&class).into_iter().flatten().map(|(next, _)| *next));
            }
        }
        false
    };
    entered
        .iter()
        .flat_map(|(lower, uppers)| uppers.iter().map(|(upper, at)| (*lower, *upper, *at)))
        .filter(|(lower, upper, _)| reaches(*upper, *lower))
        .collect()
}
