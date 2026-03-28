//! List operations as ExternFn. All pure, polymorphic.

use std::sync::Arc;

use acvus_interpreter::{Defs, ExternFnBuilder, ExternRegistry, RuntimeError, Uses, Value};
use acvus_mir::graph::{Constraint, FnConstraint, Signature};
use acvus_mir::ty::{Effect, Param, Ty, TySubst};
use acvus_utils::Interner;

// ── Handlers ────────────────────────────────────────────────────────

fn h_len(_: &Interner, (val,): (Value,), Uses(()): Uses<()>) -> Result<(i64, Defs<()>), RuntimeError> {
    Ok((val.as_list().len() as i64, Defs(())))
}

fn h_reverse(_: &Interner, (val,): (Value,), Uses(()): Uses<()>) -> Result<(Value, Defs<()>), RuntimeError> {
    let list = val.into_list();
    let mut items: Vec<Value> = Arc::try_unwrap(list)
        .unwrap_or_else(|arc| arc.iter().map(|v| v.share()).collect());
    items.reverse();
    Ok((Value::list(items), Defs(())))
}

// ── Builders ────────────────────────────────────────────────────────

fn build_len(interner: &Interner) -> acvus_interpreter::ExternFn {
    let mut s = TySubst::new();
    let t = s.fresh_param();
    let named = vec![Param::new(interner.intern("_0"), Ty::List(Box::new(t)))];
    let constraint = FnConstraint {
        signature: Some(Signature { params: named.clone() }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(Ty::Int),
            captures: vec![],
            effect: Effect::pure(),
        }),
        effect: None,
    };
    ExternFnBuilder::new("len", constraint).handler(h_len)
}

fn build_reverse(interner: &Interner) -> acvus_interpreter::ExternFn {
    let mut s = TySubst::new();
    let t = s.fresh_param();
    let named = vec![Param::new(interner.intern("_0"), Ty::List(Box::new(t.clone())))];
    let constraint = FnConstraint {
        signature: Some(Signature { params: named.clone() }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(Ty::List(Box::new(t))),
            captures: vec![],
            effect: Effect::pure(),
        }),
        effect: None,
    };
    ExternFnBuilder::new("reverse", constraint).handler(h_reverse)
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
