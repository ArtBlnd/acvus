//! List operations as ExternFn. All pure, polymorphic.

use std::sync::Arc;

use acvus_interpreter::{Defs, ExternFnBuilder, ExternRegistry, RuntimeError, Uses, Value};
use acvus_mir::ty::{Effect, ParamTerm, Poly, PolyBuilder, TyTerm, lift_effect_to_poly};
use acvus_utils::Interner;

// ── Handlers ────────────────────────────────────────────────────────

fn h_len(
    _: &Interner,
    (val,): (Value,),
    Uses(()): Uses<()>,
) -> Result<(i64, Defs<()>), RuntimeError> {
    Ok((val.as_list().len() as i64, Defs(())))
}

fn h_reverse(
    _: &Interner,
    (val,): (Value,),
    Uses(()): Uses<()>,
) -> Result<(Value, Defs<()>), RuntimeError> {
    let list = val.into_list();
    let mut items: Vec<Value> =
        Arc::try_unwrap(list).unwrap_or_else(|arc| arc.iter().map(|v| v.share()).collect());
    items.reverse();
    Ok((Value::list(items), Defs(())))
}

// ── Builders ────────────────────────────────────────────────────────

fn build_len(interner: &Interner) -> acvus_interpreter::ExternFn {
    let mut b = PolyBuilder::new();
    let t = b.fresh_ty_var();
    let named = vec![ParamTerm::<Poly>::new(interner.intern("_0"), TyTerm::List(Box::new(t)))];
    let ty = TyTerm::Fn {
        params: named,
        ret: Box::new(TyTerm::Int),
        captures: vec![],
        effect: lift_effect_to_poly(&Effect::pure()),
        hint: None,
    };
    ExternFnBuilder::new("len", ty).handler(h_len)
}

fn build_reverse(interner: &Interner) -> acvus_interpreter::ExternFn {
    let mut b = PolyBuilder::new();
    let t = b.fresh_ty_var();
    let named = vec![ParamTerm::<Poly>::new(
        interner.intern("_0"),
        TyTerm::List(Box::new(t.clone())),
    )];
    let ty = TyTerm::Fn {
        params: named,
        ret: Box::new(TyTerm::List(Box::new(t))),
        captures: vec![],
        effect: lift_effect_to_poly(&Effect::pure()),
        hint: None,
    };
    ExternFnBuilder::new("reverse", ty).handler(h_reverse)
}

// ── Registry ────────────────────────────────────────────────────────

pub fn list_registry() -> ExternRegistry {
    ExternRegistry::new(|interner| vec![build_len(interner), build_reverse(interner)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_produces_functions() {
        let i = acvus_utils::Interner::new();
        let reg = list_registry().register(&i);
        assert_eq!(reg.functions.len(), 2);
        assert_eq!(reg.executables.len(), 2);
    }
}
