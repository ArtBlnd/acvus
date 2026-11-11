//! Deciding: giving an undecided class its head.

use crate::primitive::ItemId;
use crate::primitive::class::ClassKind;
use crate::store::Store;
use crate::store::classes::Classes;

use super::Outcome;
use super::connect::connect;
use super::law::{Head, Heads, Law};
use super::settle::asked;
use super::take_in::take_in;

/// What gives a class its head.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Giving {
    /// What is recorded of the class ([`Law::settled`]). Nothing still to come to
    /// the class changes this.
    WhatIsRecorded,
    /// That nothing more is to come to the class ([`Law::fallback`]).
    NothingMoreToCome,
}

/// Decides every class that `giving` gives a head now, and says whether one
/// was decided. `at` is the item a decision goes by, and `visit` gives the
/// classes in the order they are asked in, which the outcome does not depend on.
///
/// That nothing more is to come is true of a class of a body that is finished,
/// and of none of a frozen one ([`Store::is_frozen`]), so it gives no head to
/// those.
///
/// What is held is found once ([`Classes::deciding`]), the law says which classes
/// not held take which heads, and each is given through the one way a class is
/// decided. A head decided has the class's own parts, so what the law gives
/// decides the head alone, and the bounds on the class connect its parts.
///
/// This is the only place a class is decided by solving.
pub fn decide<K, L, F>(store: &mut Store<K>, law: &L, outcome: &mut Outcome<K>, at: ItemId, visit: &F, giving: Giving) -> bool
where
    K: ClassKind,
    L: Law<K>,
    F: Fn(&Classes<K>) -> Vec<K::Id>,
{
    let mut undecided = asked(&store.classes, visit);
    undecided.retain(|class| !store.classes.is_decided(*class));
    if giving == Giving::NothingMoreToCome {
        undecided.retain(|class| !store.is_frozen(*class));
    }
    let mut deciding = store.classes.deciding();
    let held = |class| deciding.is_held(class);
    let Heads { heads, refused } = match giving {
        Giving::WhatIsRecorded => law.settled(deciding.classes(), &undecided, &held),
        Giving::NothingMoreToCome => law.fallback(deciding.classes(), &undecided, &held),
    };
    for refusal in refused {
        outcome.refuse(refusal);
    }
    let mut left = Vec::new();
    for Head { class, shape, by } in heads {
        left.extend(deciding.decide(&store.scopes, class, &shape, by, at));
    }
    let decided = !left.is_empty();
    for unified in left {
        for (pair, at) in take_in(&mut store.classes, law, outcome, unified) {
            connect(store, law, outcome, pair, at);
        }
    }
    decided
}
