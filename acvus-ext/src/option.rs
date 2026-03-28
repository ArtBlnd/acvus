//! Option operations as ExternFn. All pure, polymorphic.

use std::sync::Arc;

use acvus_interpreter::{Defs, ExternFnBuilder, ExternRegistry, RuntimeError, Uses, Value};
use acvus_mir::graph::{Constraint, FnConstraint, Signature};
use acvus_mir::ty::{Effect, Param, Ty, TySubst};
use acvus_utils::Interner;

// ── Handlers ────────────────────────────────────────────────────────

fn h_unwrap(_: &Interner, (val,): (Value,), Uses(()): Uses<()>) -> Result<(Value, Defs<()>), RuntimeError> {
    match val {
        Value::Variant { payload: Some(p), .. } => {
            let inner = Arc::try_unwrap(p).unwrap_or_else(|arc| arc.as_ref().share());
            Ok((inner, Defs(())))
        }
        Value::Variant { payload: None, .. } => panic!("unwrap: called on None"),
        other => panic!("unwrap: expected Variant, got {other:?}"),
    }
}

fn h_unwrap_or(_: &Interner, (val, default): (Value, Value), Uses(()): Uses<()>) -> Result<(Value, Defs<()>), RuntimeError> {
    match val {
        Value::Variant { payload: Some(p), .. } => {
            let inner = Arc::try_unwrap(p).unwrap_or_else(|arc| arc.as_ref().share());
            Ok((inner, Defs(())))
        }
        Value::Variant { payload: None, .. } => Ok((default, Defs(()))),
        other => panic!("unwrap_or: expected Variant, got {other:?}"),
    }
}

// ── Builders ────────────────────────────────────────────────────────

fn build_unwrap(interner: &Interner) -> acvus_interpreter::ExternFn {
    let mut s = TySubst::new();
    let t = s.fresh_param();
    let named = vec![Param::new(interner.intern("_0"), Ty::Option(Box::new(t.clone())))];
    let constraint = FnConstraint {
        signature: Some(Signature { params: named.clone() }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(t),
            captures: vec![],
            effect: Effect::pure(),
        }),
        effect: None,
    };
    ExternFnBuilder::new("unwrap", constraint).handler(h_unwrap)
}

fn build_unwrap_or(interner: &Interner) -> acvus_interpreter::ExternFn {
    let mut s = TySubst::new();
    let t = s.fresh_param();
    let named = vec![
        Param::new(interner.intern("_0"), Ty::Option(Box::new(t.clone()))),
        Param::new(interner.intern("_1"), t.clone()),
    ];
    let constraint = FnConstraint {
        signature: Some(Signature { params: named.clone() }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(t),
            captures: vec![],
            effect: Effect::pure(),
        }),
        effect: None,
    };
    ExternFnBuilder::new("unwrap_or", constraint).handler(h_unwrap_or)
}

// ── Registry ────────────────────────────────────────────────────────

pub fn option_registry() -> ExternRegistry {
    ExternRegistry::new(|interner| vec![build_unwrap(interner), build_unwrap_or(interner)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_produces_functions() {
        let i = acvus_utils::Interner::new();
        let reg = option_registry().register(&i);
        assert_eq!(reg.functions.len(), 2);
        assert_eq!(reg.executables.len(), 2);
    }
}
