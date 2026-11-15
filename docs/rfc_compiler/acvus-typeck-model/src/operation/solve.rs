//! Solving: settling, and what is done at each fixpoint and at the end.

use crate::primitive::ClassId;
use crate::primitive::ItemId;
use crate::primitive::constraint::ConstraintOf;
use crate::store::Store;
use crate::store::classes::Classes;

use super::Outcome;
use super::decide::decide;
use super::law::Law;
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
pub fn solve<C, L, F>(store: &mut Store<C>, law: &L, outcome: &mut Outcome, at: ItemId, visit: &F)
where
    C: ConstraintOf,
    L: Law<C>,
    F: Fn(&Classes<C>) -> Vec<ClassId>,
{
    loop {
        settle(store, law, outcome, at, visit);
        if !decide(store, law, outcome, at, visit, L::fixpoint) {
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
