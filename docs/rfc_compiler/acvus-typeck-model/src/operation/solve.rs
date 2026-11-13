//! Solving: settling, and what is done at each fixpoint and at the end.

use crate::primitive::ItemId;
use crate::primitive::class::ClassKind;
use crate::store::Store;
use crate::store::classes::Classes;

use super::Outcome;
use super::decide::decide;
use super::law::{Bodies, Law};
use super::settle::{asked, settle};

/// Solves what has been recorded.
///
/// ```text
/// settle:    what has been recorded is settled to its fixpoint ([`settle`]).
/// fixpoint:  every class the law gives a head at the fixpoint is decided
///            ([`Law::fixpoint`]). Where one was, back to settle (O19, O20).
/// end:       what the law refuses of each class is refused (O21), and what settled
///            the classes of each refusal is gathered.
/// ```
pub fn solve<K, L, F>(store: &mut Store<K>, law: &L, outcome: &mut Outcome<K>, at: ItemId, visit: &F)
where
    K: ClassKind,
    L: Law<K>,
    F: Fn(&Classes<K>) -> Vec<K::Id>,
{
    loop {
        settle(store, law, outcome, at, visit);
        if !decide(store, law, outcome, at, visit, L::fixpoint) {
            break;
        }
    }
    let bodies = Bodies::new(&store.scopes, &store.closures, law.awaiting(store));
    for class in asked(&store.classes, visit) {
        for refused in law.end(&store.classes, &bodies, class) {
            outcome.refuse(refused);
        }
    }
    outcome.gather(&store.classes);
}
