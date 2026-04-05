//! Deque operations as ExternFn.

use std::sync::Arc;

use acvus_interpreter::{Args, ExternFnBuilder, ExternRegistry, RuntimeError, Value};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::graph::{Constraint, FnConstraint, Signature};
use acvus_mir::ty::{Effect, Ownership, Param, Ty, TySubst};
use acvus_utils::Interner;

// ── Handlers ────────────────────────────────────────────────────────

fn h_append(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let mut deque = match args[0].take() {
        Value::Deque(d) => Arc::try_unwrap(d).unwrap_or_else(|arc| (*arc).clone()),
        other => panic!("append: expected Deque, got {other:?}"),
    };
    let item = args[1].take();
    deque.push(item);
    Ok(Value::Deque(Arc::new(deque)))
}

fn h_extend(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let mut deque = match args[0].take() {
        Value::Deque(d) => Arc::try_unwrap(d).unwrap_or_else(|arc| (*arc).clone()),
        other => panic!("extend: expected Deque, got {other:?}"),
    };
    let items = args[1].take().into_list();
    for item in items.iter() {
        deque.push(item.clone());
    }
    Ok(Value::Deque(Arc::new(deque)))
}

fn h_consume(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let mut deque = match args[0].take() {
        Value::Deque(d) => Arc::try_unwrap(d).unwrap_or_else(|arc| (*arc).clone()),
        other => panic!("consume: expected Deque, got {other:?}"),
    };
    let n = args[1].as_int().max(0) as usize;
    deque.consume(n.min(deque.len()));
    Ok(Value::Deque(Arc::new(deque)))
}

// ── Signature helpers ───────────────────────────────────────────────

fn p(interner: &Interner, idx: usize, ty: Ty) -> Param {
    Param::new(interner.intern(&format!("_{idx}")), ty)
}

fn make(interner: &Interner, name: &str, params: Vec<Ty>, ret: Ty) -> ExternFnBuilder {
    let named: Vec<Param> = params
        .iter()
        .enumerate()
        .map(|(i, ty)| p(interner, i, ty.clone()))
        .collect();
    let constraint = FnConstraint {
        signature: Some(Signature {
            params: named.clone(),
        }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(ret),
            captures: vec![],
            effect: Effect::pure(),
        }),
        effect: None,
        hint: None,
    };
    ExternFnBuilder::new(name, constraint)
}

// ── Registry ────────────────────────────────────────────────────────

pub fn deque_registry(interner: &Interner) -> ExternRegistry {
    let iter_qref = QualifiedRef::root(interner.intern("Iterator"));
    ExternRegistry::new(move |interner| {
        let it = |t: Ty, e: Effect| -> Ty {
            Ty::UserDefined {
                id: iter_qref,
                type_args: vec![t],
                effect_args: vec![e],
                ownership: Ownership::MoveOnly,
            }
        };

        // append: (Deque<T, O>, T) → Deque<T, O>
        let append = {
            let mut s = TySubst::new();
            let t = s.fresh_param();
            let o = s.fresh_param();
            let deque = Ty::Deque(Box::new(t.clone()), Box::new(o.clone()));
            make(interner, "append", vec![deque.clone(), t], deque).sync_handler(h_append)
        };

        // extend: (Deque<T, O>, Iterator<T, E>) → Deque<T, O>
        let extend = {
            let mut s = TySubst::new();
            let t = s.fresh_param();
            let o = s.fresh_param();
            let e = s.fresh_effect_var();
            let deque = Ty::Deque(Box::new(t.clone()), Box::new(o.clone()));
            make(interner, "extend", vec![deque.clone(), it(t, e)], deque).sync_handler(h_extend)
        };

        // consume: (Deque<T, O>, Int) → Deque<T, O>
        let consume = {
            let mut s = TySubst::new();
            let t = s.fresh_param();
            let o = s.fresh_param();
            let deque = Ty::Deque(Box::new(t), Box::new(o));
            make(interner, "consume", vec![deque.clone(), Ty::Int], deque).sync_handler(h_consume)
        };

        vec![append, extend, consume]
    })
}
