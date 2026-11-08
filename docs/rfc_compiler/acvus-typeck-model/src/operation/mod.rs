//! Operations above the store: the judgments it leaves to its callers.

pub mod bound;
pub mod projection;

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
    pub pending: Vec<Pending<K>>,
    /// Bounds given back that no operation here judges yet.
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
}
