//! Terms to concrete types, once solving ends (compilation: O(size)).

use rustc_hash::FxHashMap;

use super::{Infer, Term};
use crate::ty::{Concrete, Ty, TyVarId, Types};

/// Each term already frozen, so a shared subterm freezes once.
#[derive(Default)]
pub struct FreezeMemo {
    done: FxHashMap<Term, Ty>,
}

/// A variable still unbound at freeze: defaults and the final refusal ran
/// before freezing, so this is the checker's defect, reported as such.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unsettled {
    pub var: TyVarId,
}

impl Infer {
    pub fn freeze(&self, into: &mut Types<Concrete>, memo: &mut FreezeMemo, term: Term) -> Result<Ty, Unsettled> {
        todo!("infer::freeze")
    }
}
