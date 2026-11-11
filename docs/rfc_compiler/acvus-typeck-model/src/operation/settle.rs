//! Settling: taking the three operations in turn until nothing is left to do.

use crate::primitive::ItemId;
use crate::primitive::class::ClassKind;
use crate::primitive::scope::Scopes;
use crate::store::classes::Classes;
use crate::store::closures::Closures;

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
pub fn settle<K, L, F>(
    classes: &mut Classes<K>,
    scopes: &mut Scopes,
    closures: &mut Closures<K>,
    law: &L,
    outcome: &mut Outcome<K>,
    at: ItemId,
    visit: &F,
)
where
    K: ClassKind,
    L: Law<K>,
    F: Fn(&Classes<K>) -> Vec<K::Id>,
{
    loop {
        loop {
            let mut changed = false;
            for (a, b, by) in law.one(classes) {
                changed |= connect(classes, scopes, law, outcome, Pair::Classes(a, b), by);
            }
            let woken = classes.take_woken();
            let none_woken = woken.is_empty();
            for (constraint, by) in woken {
                if classes.constraints().contains(&(constraint.clone(), by)) {
                    changed |= look(classes, scopes, closures, law, outcome, constraint, by);
                }
            }
            if !changed && none_woken {
                break;
            }
        }
        if !decide(classes, scopes, law, outcome, at, visit) {
            break;
        }
    }
    for class in visit(classes) {
        for refused in law.end(classes, class) {
            outcome.refuse(refused);
        }
    }
    outcome.gather(classes);
}
