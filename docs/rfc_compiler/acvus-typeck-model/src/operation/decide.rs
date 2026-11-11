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

/// Decides every class that may be decided now, and says whether one was
/// decided. `at` is the item a decision goes by, and `visit` gives the classes in
/// the order they are asked in, which the outcome does not depend on.
///
/// What is held is found once ([`Classes::deciding`]), the law says which classes
/// not held take which heads ([`Law::heads`]), and each is given through the one
/// way a class is decided. A head decided has the class's own parts, so what the
/// law gives decides the head alone, and the bounds on the class connect its
/// parts.
///
/// This is the only place a class is decided by solving.
pub fn decide<K, L, F>(store: &mut Store<K>, law: &L, outcome: &mut Outcome<K>, at: ItemId, visit: &F) -> bool
where
    K: ClassKind,
    L: Law<K>,
    F: Fn(&Classes<K>) -> Vec<K::Id>,
{
    let mut undecided = asked(&store.classes, visit);
    undecided.retain(|class| !store.classes.is_decided(*class));
    let mut deciding = store.classes.deciding();
    let Heads { heads, refused } = law.heads(deciding.classes(), &undecided, &|class| deciding.is_held(class));
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
