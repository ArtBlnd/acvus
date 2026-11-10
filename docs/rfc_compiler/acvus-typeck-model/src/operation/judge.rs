//! Looking at a bound: judging it, or learning what it waits for.

use crate::primitive::ItemId;
use crate::primitive::class::ClassKind;
use crate::primitive::scope::Scopes;
use crate::store::classes::Classes;

use super::Outcome;
use super::connect::connect;
use super::law::{Law, Looked, Scribe};

/// Looks at `bound`, recorded by `at` and not finished, and says whether that
/// changed anything.
///
/// The law says what the bound comes to as its ends stand ([`Law::look`]), and
/// this does what follows from the answer, the same for every bound:
///
/// | The law's answer | What is done |
/// |---|---|
/// | it holds | The bound is finished, its adjustments are recorded, and its pairs are connected. |
/// | it does not hold | The bound is finished, and the refusal recorded. |
/// | it cannot be told yet | The store is told what the bound waits on and what it holds apart, and the pairs known already are connected. |
///
/// This is the only place a bound is finished, and the only place a guard or a
/// wait is stated. So a rule cannot finish a bound without judging it, nor leave
/// one waiting on nothing.
pub fn look<K, L>(classes: &mut Classes<K>, scopes: &Scopes, law: &L, outcome: &mut Outcome<K>, bound: K::Bound, at: ItemId) -> bool
where
    K: ClassKind,
    L: Law<K>,
{
    let looked = law.look(&mut Scribe::new(classes, scopes), &bound, at);
    let (pairs, changed) = match looked {
        Looked::Holds { pairs, adjustments } => {
            classes.finish(&bound, at);
            if !adjustments.is_empty() {
                outcome.adjustments.entry(at).or_default().extend(adjustments);
            }
            (pairs, true)
        }
        Looked::Refused(refusal) => {
            classes.finish(&bound, at);
            outcome.refuse(refusal);
            (Vec::new(), true)
        }
        Looked::Waits { on, apart, pairs } => {
            classes.guard(&bound, at, &apart);
            classes.wait(&bound, at, &on);
            (pairs, false)
        }
    };
    let mut connected = false;
    for pair in pairs {
        connected |= connect(classes, scopes, law, outcome, pair, at);
    }
    changed || connected
}
