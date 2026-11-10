//! Connecting: making two classes one type.

use std::collections::BTreeSet;

use crate::primitive::ItemId;
use crate::primitive::class::{ClassKind, Pending, Side, State, Unplaced};
use crate::primitive::refusal::{Cause, Refusal};
use crate::primitive::scope::Scopes;
use crate::store::classes::{Classes, Unified};

use super::Outcome;
use super::law::{Law, Pair, Place};

/// Makes the two classes of `pair` one type, by `at`, and says whether they were
/// not so already.
///
/// | The two classes | What being one type comes to |
/// |---|---|
/// | neither decided | They are one class: what enters either enters it, and the parts of one name on the two are one type in turn. |
/// | one decided by solving or by the text | The other is that class. What was recorded on it stays to be judged against the head, and each part named on it is one type with the part in its place. |
/// | one a declaration writes | Nothing is merged with it. The law's bound says the other has its head, and is judged as any bound. |
/// | both decided | They have one head, or the program is refused, and their parts are one type in turn. |
///
/// This is the only place two classes are merged. A part named on a class
/// decided as a head with no place of that name is asked of the law
/// ([`Law::place`]), and so is the head a literal's class takes
/// ([`Law::admits`]).
pub fn connect<K, L>(
    classes: &mut Classes<K>,
    scopes: &Scopes,
    law: &L,
    outcome: &mut Outcome<K>,
    pair: Pair<K>,
    at: ItemId,
) -> bool
where
    K: ClassKind,
    L: Law<K>,
{
    let mut pairs = vec![(pair, at)];
    let mut connected = false;
    while let Some((pair, at)) = pairs.pop() {
        let (a, b) = match pair {
            Pair::Classes(a, b) => (a, b),
            Pair::Part { part, of, key } => match classes.part(of, key, Side::Carried, at) {
                Some(named) => (named, part),
                None => continue,
            },
        };
        let (a, b) = (classes.find(a).id(), classes.find(b).id());
        if a == b {
            continue;
        }
        let compared = match (classes.is_external(a), classes.is_external(b)) {
            (_, true) => Some(law.same(a, b)),
            (true, false) => Some(law.same(b, a)),
            (false, false) => None,
        };
        match compared {
            Some(bound) => connected |= classes.bound(bound, at),
            None => {
                let unified = classes.unify(scopes, a, b, at);
                connected = true;
                pairs.extend(take_in(classes, law, outcome, unified));
            }
        }
    }
    connected
}

/// Takes in what a union or a decision left: its refusals, and of the class
/// that came to be decided, what a literal registered on it and each part named
/// on it that its head has no place for. It gives the pairs that are one type by
/// where the law says such a part lies.
pub(super) fn take_in<K, L>(classes: &mut Classes<K>, law: &L, outcome: &mut Outcome<K>, unified: Unified<K>) -> Vec<(Pair<K>, ItemId)>
where
    K: ClassKind,
    L: Law<K>,
{
    for refusal in unified.refusals {
        outcome.refuse(refusal);
    }
    for Pending { class, calls, domains } in unified.pending {
        let (head, by) = decided(classes, class);
        for (domain, registered) in &domains {
            if !law.admits(&head, domain) {
                let mut items = by.clone();
                items.insert(*registered);
                outcome.refuse(Refusal {
                    cause: Cause::EmptyDomain,
                    items,
                });
            }
        }
        if !calls.is_empty() {
            outcome.calls.push(Pending {
                class,
                calls,
                domains: BTreeSet::new(),
            });
        }
    }
    let mut pairs = Vec::new();
    for Unplaced { class, key, part, at } in unified.unplaced {
        let projections: Vec<(Side, ItemId)> = part
            .lower
            .iter()
            .map(|item| (Side::Lower, *item))
            .chain(part.upper.iter().map(|item| (Side::Upper, *item)))
            .collect();
        // A name only carried here leaves nothing to do: the class a projection
        // named it on holds that part too, and places it there.
        if projections.is_empty() {
            continue;
        }
        let (head, by) = decided(classes, class);
        let named = match law.place(&head, &key) {
            Place::Within { through, adjustment } => head.place(&through).copied().and_then(|owner| {
                let mut named = None;
                for (side, item) in &projections {
                    named = Some(classes.part(owner, key.clone(), *side, *item)?);
                }
                named.map(|named| (named, adjustment))
            }),
            Place::Nowhere => None,
        };
        match named {
            Some((named, adjustment)) => {
                for (_, item) in &projections {
                    outcome.adjustments.entry(*item).or_default().push(adjustment.clone());
                }
                pairs.push((Pair::Classes(named, part.class), at));
            }
            None => {
                let mut items = by;
                items.extend(projections.iter().map(|(_, item)| *item));
                items.insert(at);
                outcome.refuse(Refusal {
                    cause: Cause::NotHeld,
                    items,
                });
            }
        }
    }
    pairs
}

fn decided<K>(classes: &Classes<K>, class: K::Id) -> (crate::primitive::ty::Shape<K::Id>, BTreeSet<ItemId>)
where
    K: ClassKind,
{
    match &classes.get(classes.find(class)).state {
        State::Decided(decided) => (decided.shape.clone(), decided.by.clone()),
        State::Undecided(_) => unreachable!("what a union leaves is left of a class that came to be decided"),
    }
}
