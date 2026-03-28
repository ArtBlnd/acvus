use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::graph::types::{Constraint, FnConstraint, FnKind, Function, QualifiedRef, Signature};
use crate::ty::{Effect, Param, Ty, TySubst};

/// Shorthand: build a `Param` with a positional dummy name `_0`, `_1`, …
fn p(interner: &Interner, idx: usize, ty: Ty) -> Param {
    Param::new(interner.intern(&format!("_{idx}")), ty)
}

/// All builtins have been migrated to ExternFn registries in acvus-ext.
/// This function returns an empty list.
pub fn standard_builtins(_interner: &Interner) -> Vec<Function> {
    vec![]
}

/// Build a name → Ty::Fn map from all builtins, for use in `TypeEnv.functions`.
pub fn builtin_fn_types(_interner: &Interner) -> FxHashMap<Astr, Ty> {
    FxHashMap::default()
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::ty::Polarity;

    /// Try builtin resolution for a given name with the given arg types.
    /// With all builtins migrated, this always returns Err.
    pub fn try_builtin(
        _interner: &Interner,
        _s: &mut TySubst,
        _name: &str,
        _arg_types: &[Ty],
    ) -> Result<Ty, ()> {
        Err(())
    }
}
