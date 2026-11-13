//! Looking at a constraint: judging it, or learning what it waits for.

use crate::primitive::constraint::ConstraintOf;
use crate::store::Store;
use crate::store::ledger::Awake;

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
pub fn look<C, L>(store: &mut Store<C>, law: &L, outcome: &mut Outcome, awake: Awake<C>) -> bool
where
    C: ConstraintOf,
    L: Law<C>,
{
    let at = awake.at();
    let looked = law.look(&mut Scribe::new(store), awake.constraint(), at);
    let (pairs, changed) = match looked {
        Looked::Holds { pairs, adjustments, constraints } => {
            store.classes.finish(awake);
            for constraint in constraints {
                store.classes.constrain(constraint, at);
            }
            if !adjustments.is_empty() {
                outcome.adjustments.entry(at).or_default().extend(adjustments);
            }
            (pairs, true)
        }
        Looked::Refused(refusal) => {
            store.classes.finish(awake);
            outcome.refuse(refusal);
            (Vec::new(), true)
        }
        Looked::Waits { on, apart, pairs } => {
            store.classes.sleep(awake, &on, &apart);
            (pairs, false)
        }
    };
    let mut connected = false;
    for pair in pairs {
        connected |= connect(store, law, outcome, pair, at);
    }
    changed || connected
}
