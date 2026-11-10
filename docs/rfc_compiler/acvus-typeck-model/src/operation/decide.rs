//! Deciding: giving an undecided class its head.

use crate::primitive::ItemId;
use crate::primitive::class::ClassKind;
use crate::primitive::scope::Scopes;
use crate::store::classes::Classes;

use super::Outcome;
use super::connect::{connect, take_in};
use super::law::{Law, Offer};

/// Decides every class that may be decided now, by the first rung of the law
/// that decides any, and says whether one was decided. `at` is the item a
/// decision goes by, and `visit` gives the classes in the order they are asked
/// in, which the outcome does not depend on.
///
/// A rung offers heads ([`Law::offer`]), and this gives them, the same for every
/// rung: the offers of one rung are gathered first, and then given together
/// through the one way a class is decided ([`Classes::deciding`]), which passes
/// over a class that is held. A head decided has the class's own parts, so what
/// a rung offers decides the head alone, and the bounds on the class connect its
/// parts.
///
/// This is the only place a class is decided by solving.
pub fn decide<K, L, F>(classes: &mut Classes<K>, scopes: &Scopes, law: &L, outcome: &mut Outcome<K>, at: ItemId, visit: &F) -> bool
where
    K: ClassKind,
    L: Law<K>,
    F: Fn(&Classes<K>) -> Vec<K::Id>,
{
    for rung in 0..law.rungs() {
        let mut heads = Vec::new();
        for class in visit(classes) {
            if classes.is_decided(class) {
                continue;
            }
            match law.offer(classes, rung, class) {
                Offer::Nothing => {}
                Offer::Head { shape, by } => heads.push((class, shape, by)),
                Offer::Refused(refusal) => outcome.refuse(refusal),
            }
        }
        let mut deciding = classes.deciding();
        let mut left = Vec::new();
        for (class, shape, by) in heads {
            left.extend(deciding.decide(scopes, class, &shape, by, at));
        }
        if left.is_empty() {
            continue;
        }
        for unified in left {
            for (pair, at) in take_in(classes, law, outcome, unified) {
                connect(classes, scopes, law, outcome, pair, at);
            }
        }
        return true;
    }
    false
}
