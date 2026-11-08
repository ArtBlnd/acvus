//! Operations above the store: the judgments it leaves to its callers.

pub mod bound;
pub mod change;
pub mod projection;
pub mod settle;

#[cfg(test)]
mod testing;

use crate::primitive::adjustment::Adjustments;
use crate::primitive::class::{Bounds, ClassKind, Pending, Unplaced};
use crate::primitive::refusal::Refusal;
use crate::store::classes::Unified;

/// What these operations leave to their caller.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Outcome<K>
where
    K: ClassKind,
{
    pub refusals: Vec<Refusal>,
    /// The calls on classes decided, which no operation here judges yet.
    pub pending: Vec<Pending<K>>,
    /// Changes given back that cannot be made yet.
    pub bounds: Bounds<K>,
    pub adjustments: Adjustments,
}

impl<K> Outcome<K>
where
    K: ClassKind,
{
    /// Takes in what a union left, giving back the parts it left with no place.
    fn take_in(&mut self, unified: Unified<K>) -> Vec<Unplaced<K>> {
        self.refusals.extend(unified.refusals);
        self.pending.extend(unified.pending);
        self.bounds.extend(unified.bounds);
        unified.unplaced
    }

    /// Records `refusal`, unless the same refusal stands already. A class that
    /// stays as it is gives the same one each time it is looked at.
    fn refuse_once(&mut self, refusal: Refusal) {
        if !self.refusals.contains(&refusal) {
            self.refusals.push(refusal);
        }
    }
}
