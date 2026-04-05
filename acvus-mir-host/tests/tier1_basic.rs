//! End-to-end tests: concrete + generic + callable + monomorphize + scoped + async.

use acvus_mir_host::testing::{DummyRegistrar, DummyScope};
use acvus_mir_host::{
    extern_fn, Callable, EffectParam, ExternFnDef, ExternType, Hosted,
    ITy, Monomorphize, Scope, Ty,
};
use acvus_mir::ty::Effect;
use acvus_utils::{Interner, QualifiedRef};

// ── Concrete: len_str ──────────────────────────────────────────────

#[extern_fn(name = "len_str", LenStrFn)]
fn len_str(s: String) -> (i64,) {
    (s.len() as i64,)
}

#[test]
fn len_str_call() {
    let mut scope = DummyScope::new();
    let arg = scope.alloc();
    let dst = scope.alloc();
    scope.store(arg, "hello".to_string());
    LenStrFn::call(&mut scope, &[arg], &[dst]).unwrap_ready().unwrap();
    let val: i64 = scope.take(dst);
    assert_eq!(val, 5);
}

#[test]
fn len_str_empty() {
    let mut scope = DummyScope::new();
    let arg = scope.alloc();
    let dst = scope.alloc();
    scope.store(arg, "".to_string());
    LenStrFn::call(&mut scope, &[arg], &[dst]).unwrap_ready().unwrap();
    let val: i64 = scope.take(dst);
    assert_eq!(val, 0);
}

#[test]
fn len_str_registration() {
    let mut reg = DummyRegistrar::new();
    LenStrFn::register(&mut reg);
    assert!(reg.has_drop::<String>());
    assert!(reg.has_clone::<String>());
    assert!(!reg.has_copy::<String>());
    assert!(reg.has_drop::<i64>());
    assert!(reg.has_copy::<i64>());
}

#[test]
fn len_str_constraint() {
    let interner = Interner::new();
    let c = LenStrFn::constraint(&interner);
    let sig = c.signature.as_ref().unwrap();
    assert_eq!(sig.params.len(), 1);
    assert_eq!(sig.params[0].ty, Ty::String);
    match &c.output {
        acvus_mir_host::Constraint::Exact(Ty::Fn { params, ret, .. }) => {
            assert_eq!(params.len(), 1);
            assert_eq!(params[0].ty, Ty::String);
            assert_eq!(**ret, Ty::Int);
        }
        other => panic!("expected Exact(Ty::Fn), got {:?}", other),
    }
}

#[test]
fn len_str_function() {
    let interner = Interner::new();
    let f = LenStrFn::function(&interner);
    assert_eq!(f.qref, QualifiedRef::root(interner.intern("len_str")));
    assert!(matches!(f.kind, acvus_mir_host::FnKind::Extern));
}

// ── Concrete: add ──────────────────────────────────────────────────

#[extern_fn(name = "add", AddFn)]
fn add(a: i64, b: i64) -> (i64,) {
    (a + b,)
}

#[test]
fn add_call() {
    let mut scope = DummyScope::new();
    let a = scope.alloc();
    let b = scope.alloc();
    let dst = scope.alloc();
    scope.store(a, 10i64);
    scope.store(b, 20i64);
    AddFn::call(&mut scope, &[a, b], &[dst]).unwrap_ready().unwrap();
    let val: i64 = scope.take(dst);
    assert_eq!(val, 30);
}

#[test]
fn add_constraint() {
    let interner = Interner::new();
    let c = AddFn::constraint(&interner);
    let sig = c.signature.as_ref().unwrap();
    assert_eq!(sig.params.len(), 2);
    assert_eq!(sig.params[0].ty, Ty::Int);
    assert_eq!(sig.params[1].ty, Ty::Int);
    match &c.output {
        acvus_mir_host::Constraint::Exact(Ty::Fn { ret, .. }) => {
            assert_eq!(**ret, Ty::Int);
        }
        other => panic!("expected Exact(Ty::Fn), got {:?}", other),
    }
}

// ── Generic: reverse<T> ───────────────────────────────────────────

#[extern_fn(name = "reverse", ReverseFn)]
fn reverse<T: Hosted>(v: Vec<T>) -> (Vec<T>,) {
    (v.into_iter().rev().collect(),)
}

#[test]
fn reverse_constraint() {
    let interner = Interner::new();
    let c = ReverseFn::constraint(&interner);
    let sig = c.signature.as_ref().unwrap();
    assert_eq!(sig.params.len(), 1);
    match &sig.params[0].ty {
        Ty::List(inner) => assert!(inner.is_param()),
        other => panic!("expected List(Param), got {:?}", other),
    }
    match &c.output {
        acvus_mir_host::Constraint::Exact(Ty::Fn { params, ret, .. }) => {
            match (&params[0].ty, ret.as_ref()) {
                (Ty::List(p), Ty::List(r)) => assert_eq!(p, r, "same type var"),
                other => panic!("expected (List, List), got {:?}", other),
            }
        }
        other => panic!("expected Exact(Ty::Fn), got {:?}", other),
    }
}

// ── Generic 2-param: swap<T, U> ───────────────────────────────────

#[extern_fn(name = "swap", SwapFn)]
fn swap<T: Hosted, U: Hosted>(a: T, b: U) -> (U, T) {
    (b, a)
}

#[test]
fn swap_constraint() {
    let interner = Interner::new();
    let c = SwapFn::constraint(&interner);
    let sig = c.signature.as_ref().unwrap();
    assert_eq!(sig.params.len(), 2);
    let t0 = &sig.params[0].ty;
    let t1 = &sig.params[1].ty;
    assert!(t0.is_param());
    assert!(t1.is_param());
    assert_ne!(t0, t1);
    match &c.output {
        acvus_mir_host::Constraint::Exact(Ty::Fn { params, ret, .. }) => {
            match ret.as_ref() {
                Ty::Tuple(elems) => {
                    assert_eq!(&elems[0], t1);
                    assert_eq!(&elems[1], t0);
                }
                other => panic!("expected Tuple, got {:?}", other),
            }
        }
        other => panic!("expected Exact(Ty::Fn), got {:?}", other),
    }
}

// ── Callable: filter<T, E, F> ─────────────────────────────────────

#[extern_fn(name = "filter", FilterFn)]
fn filter<T: Hosted, E: EffectParam, F: Callable<(T,), bool, E>>(v: Vec<T>, _pred: F) -> (Vec<T>,) {
    (v,) // body doesn't matter for constraint test
}

#[test]
fn filter_constraint() {
    let interner = Interner::new();
    let c = FilterFn::constraint(&interner);
    let sig = c.signature.as_ref().unwrap();
    assert_eq!(sig.params.len(), 2);

    let t_param = match &sig.params[0].ty {
        Ty::List(inner) => { assert!(inner.is_param()); inner.as_ref().clone() }
        other => panic!("expected List(Param), got {:?}", other),
    };

    match &sig.params[1].ty {
        Ty::Fn { params: fn_params, ret, effect, .. } => {
            assert_eq!(fn_params[0].ty, t_param);
            assert_eq!(**ret, Ty::Bool);
            assert!(effect.is_var());
        }
        other => panic!("expected Ty::Fn, got {:?}", other),
    }
}

// ── UserDefined: AcvusIter<T, E> ──────────────────────────────────

#[derive(ExternType)]
#[extern_type(name = "Iterator", effects(E))]
struct AcvusIter<T, E>(std::marker::PhantomData<(T, E)>);

unsafe impl<T: Hosted, E: EffectParam> Hosted for AcvusIter<T, E> {}

#[test]
fn iter_type_decl() {
    let interner = Interner::new();
    let decl = AcvusIter::<(), ()>::type_decl(&interner);
    assert_eq!(decl.qref, QualifiedRef::root(interner.intern("Iterator")));
    assert_eq!(decl.type_params.len(), 1);
    assert_eq!(decl.effect_params.len(), 1);
}

// ── map: Iterator<T, E> × Fn(T)->U → Iterator<U, E> ──────────────

#[extern_fn(name = "map", MapFn)]
fn map_iter<T: Hosted, U: Hosted, E: EffectParam, F: Callable<(T,), U, E>>(
    it: AcvusIter<T, E>, _f: F,
) -> (AcvusIter<U, E>,) {
    let _ = it;
    unreachable!("constraint test only")
}

#[test]
fn map_constraint() {
    let interner = Interner::new();
    let c = MapFn::constraint(&interner);
    let sig = c.signature.as_ref().unwrap();
    assert_eq!(sig.params.len(), 2);

    let (t_param, e_var) = match &sig.params[0].ty {
        Ty::UserDefined { id, type_args, effect_args, .. } => {
            assert_eq!(*id, QualifiedRef::root(interner.intern("Iterator")));
            assert!(type_args[0].is_param());
            assert!(effect_args[0].is_var());
            (type_args[0].clone(), effect_args[0].clone())
        }
        other => panic!("expected UserDefined(Iterator), got {:?}", other),
    };

    let u_param = match &sig.params[1].ty {
        Ty::Fn { params: fn_params, ret, effect, .. } => {
            assert_eq!(fn_params[0].ty, t_param);
            assert!(ret.is_param());
            assert_eq!(*effect, e_var);
            ret.as_ref().clone()
        }
        other => panic!("expected Ty::Fn, got {:?}", other),
    };

    assert_ne!(t_param, u_param);

    match &c.output {
        acvus_mir_host::Constraint::Exact(Ty::Fn { ret, .. }) => {
            match ret.as_ref() {
                Ty::UserDefined { type_args, effect_args, .. } => {
                    assert_eq!(type_args[0], u_param);
                    assert_eq!(effect_args[0], e_var);
                }
                other => panic!("expected UserDefined, got {:?}", other),
            }
        }
        _ => panic!("expected Exact(Ty::Fn)"),
    }
}

// ── collect: Iterator<T, E> → Vec<T> (effect consumed) ────────────

#[extern_fn(name = "collect", CollectFn)]
fn collect_iter<T: Hosted, E: EffectParam>(it: AcvusIter<T, E>) -> (Vec<T>,) {
    let _ = it;
    unreachable!("constraint test only")
}

#[test]
fn collect_constraint() {
    let interner = Interner::new();
    let c = CollectFn::constraint(&interner);
    let sig = c.signature.as_ref().unwrap();
    assert_eq!(sig.params.len(), 1);
    let t_param = match &sig.params[0].ty {
        Ty::UserDefined { type_args, .. } => type_args[0].clone(),
        other => panic!("expected UserDefined, got {:?}", other),
    };
    match &c.output {
        acvus_mir_host::Constraint::Exact(Ty::Fn { ret, .. }) => {
            match ret.as_ref() {
                Ty::List(inner) => assert_eq!(**inner, t_param),
                other => panic!("expected List, got {:?}", other),
            }
        }
        _ => panic!("expected Exact(Ty::Fn)"),
    }
}

// ── Constrained monomorphization: add_num ──────────────────────────

#[extern_fn(name = "add_num", AddNumFn)]
fn add_num<A: Monomorphize<(i64, f64)> + std::ops::Add<Output = A>>(a: A, b: A) -> (A,) {
    (a + b,)
}

#[test]
fn add_num_monomorphized() {
    let interner = Interner::new();
    let c0 = AddNumFn0::constraint(&interner);
    let sig0 = c0.signature.as_ref().unwrap();
    assert_eq!(sig0.params[0].ty, Ty::Int);
    let c1 = AddNumFn1::constraint(&interner);
    let sig1 = c1.signature.as_ref().unwrap();
    assert_eq!(sig1.params[0].ty, Ty::Float);
}

#[test]
fn add_num_call_i64() {
    let mut scope = DummyScope::new();
    let a = scope.alloc();
    let b = scope.alloc();
    let dst = scope.alloc();
    scope.store(a, 10i64);
    scope.store(b, 20i64);
    AddNumFn0::call(&mut scope, &[a, b], &[dst]).unwrap_ready().unwrap();
    let val: i64 = scope.take(dst);
    assert_eq!(val, 30);
}

#[test]
fn add_num_call_f64() {
    let mut scope = DummyScope::new();
    let a = scope.alloc();
    let b = scope.alloc();
    let dst = scope.alloc();
    scope.store(a, 1.5f64);
    scope.store(b, 2.5f64);
    AddNumFn1::call(&mut scope, &[a, b], &[dst]).unwrap_ready().unwrap();
    let val: f64 = scope.take(dst);
    assert_eq!(val, 4.0);
}

#[test]
fn add_num_registration() {
    let mut reg = DummyRegistrar::new();
    AddNumFn::register(&mut reg);
    assert!(reg.has_drop::<i64>());
    assert!(reg.has_copy::<i64>());
    assert!(reg.has_drop::<f64>());
    assert!(reg.has_copy::<f64>());
}

// ── Scoped: count ──────────────────────────────────────────────────

#[extern_fn(name = "count", CountFn)]
fn count<S: Scope, T: Hosted>(scope: &mut S, v: Vec<T>) -> (i64,) {
    let _ = (scope, &v);
    (0i64,)
}

#[test]
fn count_scoped_constraint() {
    let interner = Interner::new();
    let c = CountFn::constraint(&interner);
    let sig = c.signature.as_ref().unwrap();
    assert_eq!(sig.params.len(), 1, "scope should not appear as param");
    match &sig.params[0].ty {
        Ty::List(inner) => assert!(inner.is_param()),
        other => panic!("expected List(Param), got {:?}", other),
    }
}

// ── Scoped + Callable: count_if (async — calls scope.call_1) ──────

#[extern_fn(name = "count_if", CountIfFn)]
async fn count_if<S: Scope, E: EffectParam, F: Callable<(i64,), bool, E>>(
    scope: &mut S, items: Vec<i64>, pred: F,
) -> (i64,) {
    let mut count = 0i64;
    for item in items {
        let pass: bool = scope.call_1(&pred, item).await.unwrap();
        if pass {
            count += 1;
        }
    }
    (count,)
}

#[tokio::test]
async fn count_if_call() {
    let mut scope = DummyScope::new();
    let pred = DummyScope::make_fn_1(|x: i64| -> bool { x > 5 });
    let items_slot = scope.alloc();
    let pred_slot = scope.alloc();
    let dst = scope.alloc();
    scope.store(items_slot, vec![1i64, 3, 7, 10, 2, 8]);
    scope.store(pred_slot, pred);
    CountIfFn::call(&mut scope, &[items_slot, pred_slot], &[dst]).await.unwrap();
    let count: i64 = scope.take(dst);
    assert_eq!(count, 3); // 7, 10, 8 pass
}

// ── Infer: json_parse ──────────────────────────────────────────────

#[extern_fn(name = "json_parse", JsonParseFn, infer)]
fn json_parse(s: String, inferred_ret_ty: Ty) -> acvus_mir_host::Inferrable {
    // Simplified: only handles Int and String.
    use acvus_mir_host::Inferrable;
    match inferred_ret_ty {
        Ty::Int => match s.parse::<i64>() {
            Ok(n) => Inferrable::Int(n),
            Err(_) => Inferrable::Failed,
        },
        Ty::String => Inferrable::String(s),
        _ => Inferrable::Failed,
    }
}

#[test]
fn json_parse_constraint() {
    let interner = Interner::new();
    let c = JsonParseFn::constraint(&interner);
    let sig = c.signature.as_ref().unwrap();
    assert_eq!(sig.params.len(), 1); // only `s: String`, not inferred_ret_ty
    assert_eq!(sig.params[0].ty, Ty::String);
    assert!(matches!(c.output, acvus_mir_host::Constraint::Inferred));
}

#[test]
fn json_parse_call_int() {
    let mut scope = DummyScope::new();
    let arg = scope.alloc();
    let dst = scope.alloc();
    scope.store(arg, "42".to_string());
    JsonParseFn::call(&mut scope, &[arg], &[dst], &Ty::Int).unwrap_ready().unwrap();
    let result: acvus_mir_host::Inferrable = scope.take(dst);
    match result {
        acvus_mir_host::Inferrable::Int(n) => assert_eq!(n, 42),
        other => panic!("expected Int, got {:?}", other),
    }
}

#[test]
fn json_parse_call_string() {
    let mut scope = DummyScope::new();
    let arg = scope.alloc();
    let dst = scope.alloc();
    scope.store(arg, "hello".to_string());
    JsonParseFn::call(&mut scope, &[arg], &[dst], &Ty::String).unwrap_ready().unwrap();
    let result: acvus_mir_host::Inferrable = scope.take(dst);
    match result {
        acvus_mir_host::Inferrable::String(s) => assert_eq!(s, "hello"),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn json_parse_call_failed() {
    let mut scope = DummyScope::new();
    let arg = scope.alloc();
    let dst = scope.alloc();
    scope.store(arg, "not_a_number".to_string());
    JsonParseFn::call(&mut scope, &[arg], &[dst], &Ty::Int).unwrap_ready().unwrap();
    let result: acvus_mir_host::Inferrable = scope.take(dst);
    assert!(matches!(result, acvus_mir_host::Inferrable::Failed));
}
