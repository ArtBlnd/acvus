//! Settling: taking the three operations in turn until nothing is left to do.

use std::collections::BTreeSet;

use crate::primitive::ItemId;
use crate::primitive::class::ClassKind;
use crate::store::Store;
use crate::store::classes::Classes;

use super::Outcome;
use super::connect::connect;
use super::decide::decide;
use super::judge::look;
use super::law::{Law, Pair};

/// Solves what has been recorded.
///
/// ```text
/// look:    each constraint woken is looked at, and each pair the law finds one type is
///          connected, until no constraint is woken.
/// decide:  every class that may be decided now is decided, by the first rung that
///          decides any. Where one was, back to look.
/// end:     what the law refuses of each class is refused, and what settled the
///          classes of each refusal is gathered.
/// ```
///
/// A constraint is woken when it is recorded, and when a class it waits on is merged
/// or decided, so nothing is looked at that nothing has happened to. A class is
/// decided only after looking has run to its end, so every guard is as its constraint
/// last stated it.
pub fn settle<K, L, F>(store: &mut Store<K>, law: &L, outcome: &mut Outcome<K>, at: ItemId, visit: &F)
where
    K: ClassKind,
    L: Law<K>,
    F: Fn(&Classes<K>) -> Vec<K::Id>,
{
    loop {
        loop {
            let mut changed = false;
            for (a, b, by) in law.one(&store.classes) {
                changed |= connect(store, law, outcome, Pair::Classes(a, b), by);
            }
            let woken = store.classes.take_woken();
            let none_woken = woken.is_empty();
            for (constraint, by) in woken {
                if store.classes.constraints().contains(&(constraint.clone(), by)) {
                    changed |= look(store, law, outcome, constraint, by);
                }
            }
            if !changed && none_woken {
                break;
            }
        }
        if !decide(store, law, outcome, at, visit) {
            break;
        }
    }
    for class in asked(&store.classes, visit) {
        for refused in law.end(&store.classes, class) {
            outcome.refuse(refused);
        }
    }
    outcome.gather(&store.classes);
}

/// The classes `visit` gives, each once, by its root, in the order it first
/// gives them.
pub(super) fn asked<K, F>(classes: &Classes<K>, visit: &F) -> Vec<K::Id>
where
    K: ClassKind,
    F: Fn(&Classes<K>) -> Vec<K::Id>,
{
    let mut seen = BTreeSet::new();
    visit(classes)
        .into_iter()
        .map(|class| classes.find(class).id())
        .filter(|root| seen.insert(*root))
        .collect()
}
