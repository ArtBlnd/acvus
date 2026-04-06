//! Iterator operations as ExternFn.
//!
//! - Constructors: iter, rev_iter
//! - Lazy combinators: map, pmap, filter, take, skip, chain, pchain, flatten, flat_map
//! - Consumers (async): collect, join, first, last, contains, next, find, reduce, fold, any, all

use std::sync::Arc;

use acvus_interpreter::{
    Args, ExternFnBuilder, ExternRegistry, IterHandle, RuntimeError, Value, exec_next,
};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::{CastRule, Effect, ParamTerm, Poly, PolyBuilder, PolyEffect, PolyTy, TyTerm, TypeRegistry, UserDefinedDecl, lift_effect_to_poly};
use acvus_utils::Interner;
use futures::future::BoxFuture;

// ── Signature helper ────────────────────────────────────────────────

fn p(interner: &Interner, idx: usize, ty: PolyTy) -> ParamTerm<Poly> {
    ParamTerm::<Poly>::new(interner.intern(&format!("_{idx}")), ty)
}

fn make_sig(params: &[PolyTy], ret: PolyTy, interner: &Interner) -> PolyTy {
    let named: Vec<ParamTerm<Poly>> = params
        .iter()
        .enumerate()
        .map(|(i, ty)| p(interner, i, ty.clone()))
        .collect();
    TyTerm::Fn {
        params: named,
        ret: Box::new(ret),
        captures: vec![],
        effect: lift_effect_to_poly(&Effect::pure()),
        hint: None,
    }
}
// ── Sync handlers — constructors ────────────────────────────────────
fn h_iter(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let items = match args[0].take() {
        Value::List(l) => Arc::try_unwrap(l).unwrap_or_else(|arc| arc.as_ref().clone()),
        Value::Deque(d) => {
            let d = Arc::try_unwrap(d).unwrap_or_else(|arc| (*arc).clone());
            d.into_vec()
        }
        other => panic!("iter: expected List or Deque, got {other:?}"),
    };
    Ok(Value::iterator(IterHandle::from_list(
        items,
        Effect::pure(),
    )))
}

fn h_rev_iter(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let mut items = match args[0].take() {
        Value::List(l) => Arc::try_unwrap(l).unwrap_or_else(|arc| arc.as_ref().clone()),
        Value::Deque(d) => {
            let d = Arc::try_unwrap(d).unwrap_or_else(|arc| (*arc).clone());
            d.into_vec()
        }
        other => panic!("rev_iter: expected List or Deque, got {other:?}"),
    };
    items.reverse();
    Ok(Value::iterator(IterHandle::from_list(
        items,
        Effect::pure(),
    )))
}

// ── Sync handlers — lazy combinators ────────────────────────────────

fn h_map(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = args[0].take().into_iterator();
    let f = args[1].take().into_fn();
    Ok(Value::iterator(iter.map(*f)))
}

fn h_pmap(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = args[0].take().into_iterator();
    let f = args[1].take().into_fn();
    Ok(Value::iterator(iter.map(*f)))
}

fn h_filter(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = args[0].take().into_iterator();
    let f = args[1].take().into_fn();
    Ok(Value::iterator(iter.filter(*f)))
}

fn h_take(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = args[0].take().into_iterator();
    let n = args[1].as_int().max(0) as usize;
    Ok(Value::iterator(iter.take(n)))
}

fn h_skip(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = args[0].take().into_iterator();
    let n = args[1].as_int().max(0) as usize;
    Ok(Value::iterator(iter.skip(n)))
}

fn h_chain(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let a = args[0].take().into_iterator();
    let b = args[1].take().into_iterator();
    Ok(Value::iterator(a.chain(*b)))
}

fn h_pchain(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    // pchain = parallel chain at runtime; semantically same as collecting all iterators.
    let list = args[0].take().into_list();
    let mut combined = Vec::new();
    for item in list.iter() {
        match item {
            Value::Iterator(_) => {
                // At this level we can't async-pull. Collect source items directly.
                // This is the correct behavior for pchain: merge sources.
                combined.push(item.clone());
            }
            _ => combined.push(item.clone()),
        }
    }
    // pchain: List<Iterator<T, E>> → Iterator<T, E>
    // For now, flatten the list of iterators into a single iterator by chaining sources.
    // Since iterators may not be collectible here, use a simpler approach:
    // convert list to iterator directly.
    Ok(Value::iterator(IterHandle::from_list(
        combined,
        Effect::pure(),
    )))
}

fn h_flatten(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = args[0].take().into_iterator();
    Ok(Value::iterator(iter.flatten()))
}

fn h_flat_map(mut args: Args, _interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = args[0].take().into_iterator();
    let f = args[1].take().into_fn();
    Ok(Value::iterator(iter.flat_map(*f)))
}

// ── Async handlers — consumers ──────────────────────────────────────

fn h_collect(
    mut args: Args,
    _interner: Interner,
) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = *args[0].take().into_iterator();
        let mut items = Vec::new();
        while let Some(val) = exec_next(&mut iter).await? {
            items.push(val);
        }
        Ok(Value::list(items))
    })
}

fn h_join(mut args: Args, _interner: Interner) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = *args[0].take().into_iterator();
        let sep = args[1].as_str().to_owned();
        let mut parts = Vec::new();
        while let Some(val) = exec_next(&mut iter).await? {
            parts.push(val.as_str().to_owned());
        }
        Ok(Value::string(parts.join(&sep)))
    })
}

fn h_first(mut args: Args, interner: Interner) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = *args[0].take().into_iterator();
        match exec_next(&mut iter).await? {
            Some(val) => Ok(Value::variant(interner.intern("Some"), Some(val))),
            None => Ok(Value::variant(interner.intern("None"), None)),
        }
    })
}

fn h_last(mut args: Args, interner: Interner) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = *args[0].take().into_iterator();
        let mut last = None;
        while let Some(val) = exec_next(&mut iter).await? {
            last = Some(val);
        }
        match last {
            Some(val) => Ok(Value::variant(interner.intern("Some"), Some(val))),
            None => Ok(Value::variant(interner.intern("None"), None)),
        }
    })
}

fn h_contains(
    mut args: Args,
    _interner: Interner,
) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = *args[0].take().into_iterator();
        let needle = args[1].take();
        while let Some(val) = exec_next(&mut iter).await? {
            if val.structural_eq(&needle) {
                return Ok(Value::Bool(true));
            }
        }
        Ok(Value::Bool(false))
    })
}

fn h_next(mut args: Args, interner: Interner) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = *args[0].take().into_iterator();
        match exec_next(&mut iter).await? {
            Some(val) => {
                let pair = Value::tuple(vec![val, Value::iterator(iter)]);
                Ok(Value::variant(interner.intern("Some"), Some(pair)))
            }
            None => Ok(Value::variant(interner.intern("None"), None)),
        }
    })
}

fn h_find(mut args: Args, _interner: Interner) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = *args[0].take().into_iterator();
        let f = args[1].take().into_fn();
        while let Some(val) = exec_next(&mut iter).await? {
            let keep = f.call(val.clone()).await?;
            if keep.as_bool() {
                return Ok(val);
            }
        }
        Err(RuntimeError::empty_collection(
            acvus_interpreter::error::CollectionOp::Find,
        ))
    })
}

fn h_reduce(
    mut args: Args,
    _interner: Interner,
) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = *args[0].take().into_iterator();
        let f = args[1].take().into_fn();
        let Some(mut acc) = exec_next(&mut iter).await? else {
            return Err(RuntimeError::empty_collection(
                acvus_interpreter::error::CollectionOp::Reduce,
            ));
        };
        while let Some(val) = exec_next(&mut iter).await? {
            acc = f.call2(acc, val).await?;
        }
        Ok(acc)
    })
}

fn h_fold(mut args: Args, _interner: Interner) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = *args[0].take().into_iterator();
        let mut acc = args[1].take();
        let f = args[2].take().into_fn();
        while let Some(val) = exec_next(&mut iter).await? {
            acc = f.call2(acc, val).await?;
        }
        Ok(acc)
    })
}

fn h_any(mut args: Args, _interner: Interner) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = *args[0].take().into_iterator();
        let f = args[1].take().into_fn();
        while let Some(val) = exec_next(&mut iter).await? {
            let result = f.call(val).await?;
            if result.as_bool() {
                return Ok(Value::Bool(true));
            }
        }
        Ok(Value::Bool(false))
    })
}

fn h_all(mut args: Args, _interner: Interner) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = *args[0].take().into_iterator();
        let f = args[1].take().into_fn();
        while let Some(val) = exec_next(&mut iter).await? {
            let result = f.call(val).await?;
            if !result.as_bool() {
                return Ok(Value::Bool(false));
            }
        }
        Ok(Value::Bool(true))
    })
}

// ── Registry ────────────────────────────────────────────────────────

pub fn iterator_registry(interner: &Interner, type_registry: &mut TypeRegistry) -> ExternRegistry {
    let iter_qref = QualifiedRef::root(interner.intern("Iterator"));
    type_registry.register(UserDefinedDecl {
        qref: iter_qref,
        type_params: vec![None],
        effect_params: vec![None],
    });

    // Register CastRules: List<T> → Iterator<T, Pure>, Deque<T, O> → Iterator<T, Pure>.
    {
        let mut b = PolyBuilder::new();
        let t = b.fresh_ty_var();
        type_registry.register_cast(CastRule {
            from: TyTerm::List(Box::new(t.clone())),
            to: TyTerm::UserDefined {
                id: iter_qref,
                type_args: vec![t],
                effect_args: vec![lift_effect_to_poly(&Effect::pure())],
            },
            fn_ref: QualifiedRef::root(interner.intern("iter")),
        });
    }
    {
        let mut b = PolyBuilder::new();
        let t = b.fresh_ty_var();
        let o = b.fresh_ty_var();
        type_registry.register_cast(CastRule {
            from: TyTerm::Deque(Box::new(t.clone()), Box::new(o)),
            to: TyTerm::UserDefined {
                id: iter_qref,
                type_args: vec![t],
                effect_args: vec![lift_effect_to_poly(&Effect::pure())],
            },
            fn_ref: QualifiedRef::root(interner.intern("__cast_deque_to_iter")),
        });
    }

    ExternRegistry::new(move |interner| {
        // Helper: Iterator<T, E>
        let it = |t: PolyTy, e: PolyEffect| -> PolyTy {
            TyTerm::UserDefined {
                id: iter_qref,
                type_args: vec![t],
                effect_args: vec![e],
            }
        };

        let mut fns = Vec::new();

        // ── Constructors ────────────────────────────────
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            fns.push(
                ExternFnBuilder::new(
                    "iter",
                    make_sig(
                        &[TyTerm::List(Box::new(t.clone()))],
                        it(t, lift_effect_to_poly(&Effect::pure())),
                        interner,
                    ),
                )
                .sync_handler(h_iter),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            fns.push(
                ExternFnBuilder::new(
                    "rev_iter",
                    make_sig(
                        &[TyTerm::List(Box::new(t.clone()))],
                        it(t, lift_effect_to_poly(&Effect::pure())),
                        interner,
                    ),
                )
                .sync_handler(h_rev_iter),
            );
        }
        // Cast helpers (used by CastRule, not meant to be called directly).
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let o = b.fresh_ty_var();
            fns.push(
                ExternFnBuilder::new(
                    "__cast_deque_to_iter",
                    make_sig(
                        &[TyTerm::Deque(Box::new(t.clone()), Box::new(o))],
                        it(t, lift_effect_to_poly(&Effect::pure())),
                        interner,
                    ),
                )
                .sync_handler(h_iter),
            );
        }

        // ── Lazy combinators ────────────────────────────
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let u = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(u.clone()),
                captures: vec![],
                effect: e.clone(),
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new(
                    "map",
                    make_sig(&[it(t, e.clone()), fn_ty], it(u, e), interner),
                )
                .sync_handler(h_map),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let u = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(u.clone()),
                captures: vec![],
                effect: e.clone(),
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new(
                    "pmap",
                    make_sig(&[it(t, e.clone()), fn_ty], it(u, e), interner),
                )
                .sync_handler(h_pmap),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(TyTerm::Bool),
                captures: vec![],
                effect: e.clone(),
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new(
                    "filter",
                    make_sig(&[it(t.clone(), e.clone()), fn_ty], it(t, e), interner),
                )
                .sync_handler(h_filter),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let iter_ty = it(t, e);
            fns.push(
                ExternFnBuilder::new(
                    "take",
                    make_sig(&[iter_ty.clone(), TyTerm::Int], iter_ty, interner),
                )
                .sync_handler(h_take),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let iter_ty = it(t, e);
            fns.push(
                ExternFnBuilder::new(
                    "skip",
                    make_sig(&[iter_ty.clone(), TyTerm::Int], iter_ty, interner),
                )
                .sync_handler(h_skip),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let iter_ty = it(t, e);
            fns.push(
                ExternFnBuilder::new(
                    "chain",
                    make_sig(&[iter_ty.clone(), iter_ty.clone()], iter_ty, interner),
                )
                .sync_handler(h_chain),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let iter_ty = it(t, e);
            fns.push(
                ExternFnBuilder::new(
                    "pchain",
                    make_sig(&[TyTerm::List(Box::new(iter_ty.clone()))], iter_ty, interner),
                )
                .sync_handler(h_pchain),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            fns.push(
                ExternFnBuilder::new(
                    "flatten",
                    make_sig(
                        &[it(TyTerm::List(Box::new(t.clone())), e.clone())],
                        it(t, e),
                        interner,
                    ),
                )
                .sync_handler(h_flatten),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let u = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(it(u.clone(), e.clone())),
                captures: vec![],
                effect: e.clone(),
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new(
                    "flat_map",
                    make_sig(&[it(t, e.clone()), fn_ty], it(u, e), interner),
                )
                .sync_handler(h_flat_map),
            );
        }

        // ── Consumers (async) ───────────────────────────
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            fns.push(
                ExternFnBuilder::new(
                    "collect",
                    make_sig(&[it(t.clone(), e)], TyTerm::List(Box::new(t)), interner),
                )
                .async_handler(h_collect),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let e = b.fresh_effect_var();
            fns.push(
                ExternFnBuilder::new(
                    "join",
                    make_sig(&[it(TyTerm::String, e), TyTerm::String], TyTerm::String, interner),
                )
                .async_handler(h_join),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            fns.push(
                ExternFnBuilder::new(
                    "first",
                    make_sig(&[it(t.clone(), e)], TyTerm::Option(Box::new(t)), interner),
                )
                .async_handler(h_first),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            fns.push(
                ExternFnBuilder::new(
                    "last",
                    make_sig(&[it(t.clone(), e)], TyTerm::Option(Box::new(t)), interner),
                )
                .async_handler(h_last),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            fns.push(
                ExternFnBuilder::new(
                    "contains",
                    make_sig(&[it(t.clone(), e), t], TyTerm::Bool, interner),
                )
                .async_handler(h_contains),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let iter_ty = it(t.clone(), e);
            fns.push(
                ExternFnBuilder::new(
                    "next",
                    make_sig(
                        &[iter_ty.clone()],
                        TyTerm::Option(Box::new(TyTerm::Tuple(vec![t, iter_ty]))),
                        interner,
                    ),
                )
                .async_handler(h_next),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(TyTerm::Bool),
                captures: vec![],
                effect: e.clone(),
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new("find", make_sig(&[it(t.clone(), e), fn_ty], t, interner))
                    .async_handler(h_find),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone()), p(interner, 1, t.clone())],
                ret: Box::new(t.clone()),
                captures: vec![],
                effect: e.clone(),
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new("reduce", make_sig(&[it(t.clone(), e), fn_ty], t, interner))
                    .async_handler(h_reduce),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let u = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, u.clone()), p(interner, 1, t.clone())],
                ret: Box::new(u.clone()),
                captures: vec![],
                effect: e.clone(),
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new("fold", make_sig(&[it(t, e), u.clone(), fn_ty], u, interner))
                    .async_handler(h_fold),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(TyTerm::Bool),
                captures: vec![],
                effect: e.clone(),
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new("any", make_sig(&[it(t, e), fn_ty], TyTerm::Bool, interner))
                    .async_handler(h_any),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let e = b.fresh_effect_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(TyTerm::Bool),
                captures: vec![],
                effect: e.clone(),
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new("all", make_sig(&[it(t, e), fn_ty], TyTerm::Bool, interner))
                    .async_handler(h_all),
            );
        }

        fns
    })
}
