//! Taking in what a union or a decision left.
//!
//! The store merges and decides, and judges nothing: what a class that came to
//! be decided held besides its parts, it gives back. Each thing given back is
//! asked of the law here.
//!
//! | What is left | What is asked | What follows |
//! |---|---|---|
//! | a refusal | nothing | it is recorded |
//! | a domain a literal registered | whether the head is of it ([`Law::admits`]) | it is refused where not |
//! | a part the head has no place for | where it lies ([`Law::place`]) | it is one type with the part found there, or refused |

use std::collections::BTreeSet;

use crate::primitive::ItemId;
use crate::primitive::adjustment::Adjustment;
use crate::primitive::class::{ClassKind, Pending, Side, State, Unplaced};
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::{PartKey, Shape};
use crate::store::classes::{Classes, Unified};

use super::Outcome;
use super::law::{Law, Pair, Place};

/// Takes in what `unified` left, and gives the pairs that are one type by where
/// the law says a part lies.
pub(super) fn take_in<K, L>(classes: &mut Classes<K>, law: &L, outcome: &mut Outcome<K>, unified: Unified<K>) -> Vec<(Pair<K>, ItemId)>
where
    K: ClassKind,
    L: Law<K>,
{
    for refusal in unified.refusals {
        outcome.refuse(refusal);
    }
    for pending in unified.pending {
        refuse_outside_domain(classes, law, outcome, pending);
    }
    let mut pairs = Vec::new();
    for unplaced in unified.unplaced {
        pairs.extend(place(classes, law, outcome, unplaced));
    }
    pairs
}

/// Refuses each domain a literal registered on a class that the head the class
/// came to have is not of.
fn refuse_outside_domain<K, L>(classes: &Classes<K>, law: &L, outcome: &mut Outcome<K>, pending: Pending<K>)
where
    K: ClassKind,
    L: Law<K>,
{
    let Pending { class, domains } = pending;
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
}

/// Places a part named on a class whose head has no place of that name: it is
/// one type with the part found where the law says it lies, and each projection
/// that named it takes the adjustments of the way there. Where it lies nowhere,
/// the program is refused.
fn place<K, L>(classes: &mut Classes<K>, law: &L, outcome: &mut Outcome<K>, unplaced: Unplaced<K>) -> Option<(Pair<K>, ItemId)>
where
    K: ClassKind,
    L: Law<K>,
{
    let Unplaced { class, key, part, at } = unplaced;
    let projections: Vec<ItemId> = part.lower.iter().chain(&part.upper).copied().collect();
    // A name only carried here leaves nothing to do: the class a projection
    // named it on holds that part too, and places it there.
    if projections.is_empty() {
        return None;
    }
    let Some((found, adjustments)) = through(classes, law, class, &key, &projections) else {
        outcome.refuse(Refused {
            cause: Cause::NotHeld,
            items: projections.into_iter().chain([at]).collect(),
            of: vec![class],
        });
        return None;
    };
    for projection in projections {
        outcome.adjustments.entry(projection).or_default().extend(adjustments.iter().cloned());
    }
    Some((Pair::Classes(found, part.class), at))
}

/// Follows the law from `class` through each part it says `key` lies within, to
/// the first class that has a part of that name: that part, named there by
/// `projections`, and the adjustment each step takes. None where the law says the
/// part lies nowhere.
fn through<K, L>(classes: &mut Classes<K>, law: &L, class: K::Id, key: &PartKey, projections: &[ItemId]) -> Option<(K::Id, Vec<Adjustment>)>
where
    K: ClassKind,
    L: Law<K>,
{
    let mut adjustments = Vec::new();
    let mut asked = BTreeSet::from([classes.find(class).id()]);
    let mut at = class;
    loop {
        let head = decided(classes, at);
        let Place::Within { through, adjustment } = law.place(&head, key) else {
            return None;
        };
        let owner = *head.place(&through)?;
        adjustments.push(adjustment);
        if let Some(found) = read_on(classes, owner, key, projections) {
            return Some((found, adjustments));
        }
        // A class that holds itself has no end to ask to.
        if !asked.insert(classes.find(owner).id()) {
            return None;
        }
        at = owner;
    }
}

/// Names the part `key` on `owner` by each of `projections`, and gives it: none
/// where `owner` is decided as a head with no place of that name.
///
/// The part is read there, whatever it was where it was named: a part reached
/// through another is not written into the class that holds it.
fn read_on<K>(classes: &mut Classes<K>, owner: K::Id, key: &PartKey, projections: &[ItemId]) -> Option<K::Id>
where
    K: ClassKind,
{
    let mut found = None;
    for projection in projections {
        found = classes.part(owner, key.clone(), Side::Upper, *projection);
    }
    found
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
