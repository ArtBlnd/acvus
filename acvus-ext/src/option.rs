//! Option operations as ExternFn. All pure, polymorphic.

use std::sync::Arc;

use acvus_interpreter::{ExternFnBuilder, ExternRegistry, RuntimeError, Value};
use acvus_mir::ty::{ParamTerm, Poly, PolyBuilder, TyTerm};
use acvus_utils::Interner;

// ── Handlers ────────────────────────────────────────────────────────

fn h_unwrap(
    _: &Interner,
    (val,): (Value,),
) -> Result<Value, RuntimeError> {
    match val {
        Value::Variant(v) if v.payload.is_some() => {
            let inner =
                Arc::try_unwrap(v.payload.unwrap()).unwrap_or_else(|arc| arc.as_ref().share());
            Ok(inner)
        }
        Value::Variant(_) => panic!("unwrap: called on None"),
        other => panic!("unwrap: expected Variant, got {other:?}"),
    }
}

fn h_unwrap_or(
    _: &Interner,
    (val, default): (Value, Value),
) -> Result<Value, RuntimeError> {
    match val {
        Value::Variant(v) if v.payload.is_some() => {
            let inner =
                Arc::try_unwrap(v.payload.unwrap()).unwrap_or_else(|arc| arc.as_ref().share());
            Ok(inner)
        }
        Value::Variant(_) => Ok(default),
        other => panic!("unwrap_or: expected Variant, got {other:?}"),
    }
}

// ── Builders ────────────────────────────────────────────────────────

fn build_unwrap(interner: &Interner) -> acvus_interpreter::ExternFn {
    let mut b = PolyBuilder::new();
    let t = b.fresh_ty_var();
    let named = vec![ParamTerm::<Poly>::new(
        interner.intern("_0"),
        TyTerm::Option(Box::new(t.clone())),
    )];
    let ty = TyTerm::Fn {
        params: named,
        ret: Box::new(t),
        captures: vec![],
        effect: acvus_mir::ty::Effect::Pure.into(),
    };
    ExternFnBuilder::new("unwrap", ty).handler(h_unwrap)
}

fn build_unwrap_or(interner: &Interner) -> acvus_interpreter::ExternFn {
    let mut b = PolyBuilder::new();
    let t = b.fresh_ty_var();
    let named = vec![
        ParamTerm::<Poly>::new(interner.intern("_0"), TyTerm::Option(Box::new(t.clone()))),
        ParamTerm::<Poly>::new(interner.intern("_1"), t.clone()),
    ];
    let ty = TyTerm::Fn {
        params: named,
        ret: Box::new(t),
        captures: vec![],
        effect: acvus_mir::ty::Effect::Pure.into(),
    };
    ExternFnBuilder::new("unwrap_or", ty).handler(h_unwrap_or)
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
