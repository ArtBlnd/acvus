//! Being equal: a class has the head of a class a declaration writes.

use std::collections::BTreeSet;

use crate::primitive::ItemId;
use crate::primitive::class::ClassKind;
use crate::primitive::refusal::{Cause, Refusal};

use super::super::law::Looked;
use super::{Told, View, decided_by, parts_by_name};

/// What `lower` being equal to `upper` comes to.
///
/// The parts of one name on the two are one type from the start. Once both have
/// heads, the two are of one constructor and the same members: nothing the
/// language gives goes at a part.
pub(super) fn look<K>(view: &View<'_, K>, lower: K::Id, upper: K::Id, at: ItemId) -> Told<K>
where
    K: ClassKind,
{
    let classes = view.classes;
    let pairs = parts_by_name(classes, lower, upper);
    let (Some(head), Some(declared)) = (view.head(lower), view.head(upper)) else {
        let asks = view.head(upper).map(|declared| (lower, declared.clone())).into_iter().collect();
        let looked = Looked::Waits {
            on: BTreeSet::from([lower, upper]),
            apart: BTreeSet::new(),
            pairs,
        };
        return Told { looked, asks, way: None };
    };
    if head.layer() != declared.layer() {
        let mut items = decided_by(classes, lower);
        items.extend(decided_by(classes, upper));
        items.insert(at);
        return Told::done(Looked::Refused(Refusal {
            cause: Cause::Mismatch,
            items,
        }));
    }
    Told::done(Looked::Holds {
        pairs,
        bounds: Vec::new(),
        adjustments: Vec::new(),
    })
}
