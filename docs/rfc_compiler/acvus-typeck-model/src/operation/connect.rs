//! Connecting: making two classes one type.

use std::collections::BTreeSet;

use crate::primitive::ItemId;
use crate::primitive::class::{ClassKind, Pending, Side, State, Unplaced};
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::Shape;
use crate::store::Store;
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
pub fn connect<K, L>(store: &mut Store<K>, law: &L, outcome: &mut Outcome<K>, pair: Pair<K>, at: ItemId) -> bool
where
    K: ClassKind,
    L: Law<K>,
{
    let Store { classes, scopes, .. } = store;
    let mut pairs = vec![(pair, at)];
    let mut connected = false;
    while let Some((pair, at)) = pairs.pop() {
        let (a, b) = match pair {
            Pair::Classes(a, b) => (a, b),
            Pair::Part { part, of, key } => {
                let Some(named) = classes.part(of, key, Side::Carried, at) else {
                    continue;
                };
                (named, part)
            }
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
            Some(constraint) => connected |= classes.constrain(constraint, at),
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
        let head = decided(classes, class);
        for (domain, registered) in &domains {
            if !law.admits(&head, domain) {
                outcome.refuse(Refused {
                    cause: Cause::EmptyDomain,
                    items: BTreeSet::from([*registered]),
                    of: vec![class],
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
        // The law says the part lies within another part. Where that part is
        // decided as a head with no place of the name either, the law is asked
        // of that head in turn, and each answer is one more adjustment.
        let mut head = decided(classes, class);
        let mut adjustments = Vec::new();
        let mut asked = BTreeSet::from([classes.find(class).id()]);
        let named = loop {
            let Place::Within { through, adjustment } = law.place(&head, &key) else {
                break None;
            };
            let Some(owner) = head.place(&through).copied() else {
                break None;
            };
            adjustments.push(adjustment);
            // The part is read there, whatever it was where it was named: a
            // part reached through another is not written into the class that
            // holds it.
            let mut named = None;
            for (_, item) in &projections {
                named = classes.part(owner, key.clone(), Side::Upper, *item);
            }
            if named.is_some() {
                break named;
            }
            // A class that holds itself has no end to ask to.
            if !asked.insert(classes.find(owner).id()) {
                break None;
            }
            head = decided(classes, owner);
        };
        match named {
            Some(named) => {
                for (_, item) in &projections {
                    outcome.adjustments.entry(*item).or_default().extend(adjustments.iter().cloned());
                }
                pairs.push((Pair::Classes(named, part.class), at));
            }
            None => {
                outcome.refuse(Refused {
                    cause: Cause::NotHeld,
                    items: projections.iter().map(|(_, item)| *item).chain([at]).collect(),
                    of: vec![class],
                });
            }
        }
    }
    pairs
}

/// The head of `class`, which a union or a decision left decided.
fn decided<K>(classes: &Classes<K>, class: K::Id) -> Shape<K::Id>
where
    K: ClassKind,
{
    let State::Decided(decided) = &classes.get(classes.find(class)).state else {
        unreachable!("what a union leaves is left of a class that came to be decided");
    };
    decided.shape.clone()
}
