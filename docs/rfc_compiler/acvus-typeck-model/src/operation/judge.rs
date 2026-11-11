//! Looking at a constraint: judging it, or learning what it waits for.

use crate::primitive::ItemId;
use crate::primitive::class::ClassKind;
use crate::primitive::scope::Scopes;
use crate::store::classes::Classes;
use crate::store::closures::Closures;

use super::Outcome;
use super::connect::connect;
use super::law::{Law, Looked, Scribe};

/// Looks at `constraint`, recorded by `at` and not finished, and says whether that
/// changed anything.
///
/// The law says what the constraint comes to as its ends stand ([`Law::look`]), and
/// this does what follows from the answer, the same for every constraint:
///
/// | The law's answer | What is done |
/// |---|---|
/// | it holds | The constraint is finished, its adjustments and the constraints that take its place are recorded, and its pairs are connected. |
/// | it does not hold | The constraint is finished, and the refusal recorded. |
/// | it cannot be told yet | The store is told what the constraint waits on and what it holds apart, and the pairs known already are connected. |
///
/// This is the only place a constraint is finished, and the only place a guard or a
/// wait is stated. So a rule cannot finish a constraint without judging it, nor leave
/// one waiting on nothing.
pub fn look<K, L>(
    classes: &mut Classes<K>,
    scopes: &mut Scopes,
    closures: &mut Closures<K>,
    law: &L,
    outcome: &mut Outcome<K>,
    constraint: K::Constraint,
    at: ItemId,
) -> bool
where
    K: ClassKind,
    L: Law<K>,
{
    let looked = law.look(&mut Scribe::new(classes, scopes, closures), &constraint, at);
    let (pairs, changed) = match looked {
        Looked::Holds { pairs, adjustments, constraints } => {
            classes.finish(&constraint, at);
            for constraint in constraints {
                classes.constrain(constraint, at);
            }
            if !adjustments.is_empty() {
                outcome.adjustments.entry(at).or_default().extend(adjustments);
            }
            (pairs, true)
        }
        Looked::Refused(refusal) => {
            classes.finish(&constraint, at);
            outcome.refuse(refusal);
            (Vec::new(), true)
        }
        Looked::Waits { on, apart, pairs } => {
            classes.guard(&constraint, at, &apart);
            classes.wait(&constraint, at, &on);
            (pairs, false)
        }
    };
    let mut connected = false;
    for pair in pairs {
        connected |= connect(classes, scopes, law, outcome, pair, at);
    }
    changed || connected
}
