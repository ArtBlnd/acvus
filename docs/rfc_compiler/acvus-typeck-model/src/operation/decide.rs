//! Deciding: giving an undecided class its head.

use crate::primitive::ClassId;
use crate::primitive::ItemId;
use crate::primitive::constraint::ConstraintOf;
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
/// not held take which heads, and each is given through the one way a class is
/// decided. A head decided has the class's own parts, so what the law gives
/// decides the head alone, and the bounds on the class connect its parts.
///
/// `ask` is the question the law is asked: [`Law::heads`] as solving settles,
/// and [`Law::fixpoint`] at the fixpoint.
///
/// This is the only place a class is decided by solving.
pub fn decide<C, L, F, H>(store: &mut Store<C>, law: &L, outcome: &mut Outcome, at: ItemId, visit: &F, ask: H) -> bool
where
    C: ConstraintOf,
    L: Law<C>,
    F: Fn(&Classes<C>) -> Vec<ClassId>,
    H: Fn(&L, &Classes<C>, &[ClassId], &dyn Fn(ClassId) -> bool) -> Heads,
{
    let mut undecided = asked(&store.classes, visit);
    undecided.retain(|class| !store.classes.is_decided(*class));
    let mut deciding = store.classes.deciding();
    let Heads { heads, refused } = ask(law, deciding.classes(), &undecided, &|class| deciding.is_held(class));
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
