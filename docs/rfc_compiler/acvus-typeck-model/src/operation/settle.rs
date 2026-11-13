//! Settling: taking the three operations in turn until nothing changes.

use std::collections::BTreeSet;

use crate::primitive::ClassId;
use crate::primitive::ItemId;
use crate::primitive::constraint::ConstraintOf;
use crate::store::Store;
use crate::store::classes::Classes;

use super::Outcome;
use super::connect::connect;
use super::decide::decide;
use super::judge::look;
use super::law::{Law, Pair};

/// Settles what has been recorded (O18).
///
/// ```text
/// look:    each constraint awake is looked at, and each pair the law finds one type is
///          connected, until no constraint is awake.
/// decide:  every class the law gives a head as solving settles is decided
///          ([`Law::heads`]). Where one was, back to look.
/// ```
///
/// Where none is decided nothing changes any more: the fixpoint.
///
/// A constraint is awake when it is recorded, and when a class it waits on is merged
/// or decided, so nothing is looked at that nothing has happened to. A class is
/// decided only after looking has run to its end, so every guard is as its constraint
/// last stated it.
pub fn settle<C, L, F>(store: &mut Store<C>, law: &L, outcome: &mut Outcome, at: ItemId, visit: &F)
where
    C: ConstraintOf,
    L: Law<C>,
    F: Fn(&Classes<C>) -> Vec<ClassId>,
{
    loop {
        loop {
            let mut changed = false;
            for (a, b, by) in law.one(&store.classes) {
                changed |= connect(store, law, outcome, Pair::Classes(a, b), by);
            }
            let awake = store.classes.awake();
            let none_awake = awake.is_empty();
            for constraint in awake {
                changed |= look(store, law, outcome, constraint);
            }
            if !changed && none_awake {
                break;
            }
        }
        if !decide(store, law, outcome, at, visit, L::heads) {
            break;
        }
    }
}

/// The classes `visit` gives, each once, by its root, in the order it first
/// gives them.
pub(super) fn asked<C, F>(classes: &Classes<C>, visit: &F) -> Vec<ClassId>
where
    C: ConstraintOf,
    F: Fn(&Classes<C>) -> Vec<ClassId>,
{
    let mut seen = BTreeSet::new();
    visit(classes)
        .into_iter()
        .map(|class| classes.find(class).id())
        .filter(|root| seen.insert(*root))
        .collect()
}
