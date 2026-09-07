//! Iterator operations as ExternFn.
//!
//! - Constructors: iter, rev_iter
//! - Lazy combinators: map, pmap, filter, take, skip, chain, pchain, flatten, flat_map
//! - Consumers (async): collect, join, first, last, contains, next, find, reduce, fold, any, all

use std::sync::Arc;

use acvus_interpreter::{Args, ExternFnBuilder, ExternRegistry, RuntimeError, Value};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::{CastRule, ParamTerm, Poly, PolyBuilder, PolyTy, TyTerm, TypeRegistry, UserDefinedDecl};
use acvus_utils::Interner;
use futures::future::BoxFuture;

use crate::iter_pipeline::{IterHandle, exec_next, into_iter_handle, iter_value, iterator_qref};

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
        hint: None,
    }
}
// ── Sync handlers — constructors ────────────────────────────────────
fn h_iter(mut args: Args, interner: &Interner) -> Result<Value, RuntimeError> {
    let items = match args[0].take() {
        Value::List(l) => Arc::try_unwrap(l).unwrap_or_else(|arc| arc.as_ref().clone()),
        other => panic!("iter: expected List, got {other:?}"),
    };
    Ok(iter_value(interner, IterHandle::from_list(items)))
}

fn h_rev_iter(mut args: Args, interner: &Interner) -> Result<Value, RuntimeError> {
    let mut items = match args[0].take() {
        Value::List(l) => Arc::try_unwrap(l).unwrap_or_else(|arc| arc.as_ref().clone()),
        other => panic!("rev_iter: expected List, got {other:?}"),
    };
    items.reverse();
    Ok(iter_value(interner, IterHandle::from_list(items)))
}

// ── Sync handlers — lazy combinators ────────────────────────────────

fn h_map(mut args: Args, interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = into_iter_handle(args[0].take());
    let f = args[1].take().into_fn();
    Ok(iter_value(interner, iter.map(*f)))
}

fn h_pmap(mut args: Args, interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = into_iter_handle(args[0].take());
    let f = args[1].take().into_fn();
    Ok(iter_value(interner, iter.map(*f)))
}

fn h_filter(mut args: Args, interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = into_iter_handle(args[0].take());
    let f = args[1].take().into_fn();
    Ok(iter_value(interner, iter.filter(*f)))
}

fn h_take(mut args: Args, interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = into_iter_handle(args[0].take());
    let n = args[1].as_int().max(0) as usize;
    Ok(iter_value(interner, iter.take(n)))
}

fn h_skip(mut args: Args, interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = into_iter_handle(args[0].take());
    let n = args[1].as_int().max(0) as usize;
    Ok(iter_value(interner, iter.skip(n)))
}

fn h_chain(mut args: Args, interner: &Interner) -> Result<Value, RuntimeError> {
    let a = into_iter_handle(args[0].take());
    let b = into_iter_handle(args[1].take());
    Ok(iter_value(interner, a.chain(b)))
}

fn h_pchain(mut args: Args, interner: &Interner) -> Result<Value, RuntimeError> {
    // pchain = parallel chain at runtime; semantically same as collecting all iterators.
    let list = args[0].take().into_list();
    let parts = Arc::try_unwrap(list).unwrap_or_else(|arc| arc.as_ref().clone());
    let mut chained = IterHandle::done();
    for part in parts {
        chained = chained.chain(into_iter_handle(part));
    }
    Ok(iter_value(interner, chained))
}

fn h_flatten(mut args: Args, interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = into_iter_handle(args[0].take());
    Ok(iter_value(interner, iter.flatten()))
}

fn h_flat_map(mut args: Args, interner: &Interner) -> Result<Value, RuntimeError> {
    let iter = into_iter_handle(args[0].take());
    let f = args[1].take().into_fn();
    Ok(iter_value(interner, iter.flat_map(*f)))
}

// ── Async handlers — consumers ──────────────────────────────────────

fn h_collect(
    mut args: Args,
    _interner: Interner,
) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = into_iter_handle(args[0].take());
        let mut items = Vec::new();
        while let Some(val) = exec_next(&mut iter).await? {
            items.push(val);
        }
        Ok(Value::list(items))
    })
}

fn h_join(mut args: Args, _interner: Interner) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = into_iter_handle(args[0].take());
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
        let mut iter = into_iter_handle(args[0].take());
        match exec_next(&mut iter).await? {
            Some(val) => Ok(Value::variant(interner.intern("Some"), Some(val))),
            None => Ok(Value::variant(interner.intern("None"), None)),
        }
    })
}

fn h_last(mut args: Args, interner: Interner) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = into_iter_handle(args[0].take());
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
        let mut iter = into_iter_handle(args[0].take());
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
        let mut iter = into_iter_handle(args[0].take());
        match exec_next(&mut iter).await? {
            Some(val) => {
                let pair = Value::tuple(vec![val, iter_value(&interner, iter)]);
                Ok(Value::variant(interner.intern("Some"), Some(pair)))
            }
            None => Ok(Value::variant(interner.intern("None"), None)),
        }
    })
}

fn h_find(mut args: Args, _interner: Interner) -> BoxFuture<'static, Result<Value, RuntimeError>> {
    Box::pin(async move {
        let mut iter = into_iter_handle(args[0].take());
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
        let mut iter = into_iter_handle(args[0].take());
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
        let mut iter = into_iter_handle(args[0].take());
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
        let mut iter = into_iter_handle(args[0].take());
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
        let mut iter = into_iter_handle(args[0].take());
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
    let iter_qref = iterator_qref(interner);
    type_registry.register(UserDefinedDecl {
        qref: iter_qref,
        type_params: vec![None],
    });

    {
        let mut b = PolyBuilder::new();
        let t = b.fresh_ty_var();
        type_registry.register_cast(CastRule {
            from: TyTerm::List(Box::new(t.clone())),
            to: TyTerm::UserDefined {
                id: iter_qref,
                type_args: vec![t],
            },
            fn_ref: QualifiedRef::root(interner.intern("iter")),
        });
    }

    ExternRegistry::new(move |interner| {
        // Helper: Iterator<T, E>
        let it = |t: PolyTy| -> PolyTy {
            TyTerm::UserDefined {
                id: iter_qref,
                type_args: vec![t],
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
                        it(t),
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
                        it(t),
                        interner,
                    ),
                )
                .sync_handler(h_rev_iter),
            );
        }

        // ── Lazy combinators ────────────────────────────
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let u = b.fresh_ty_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(u.clone()),
                captures: vec![],
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new(
                    "map",
                    make_sig(&[it(t), fn_ty], it(u), interner),
                )
                .sync_handler(h_map),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let u = b.fresh_ty_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(u.clone()),
                captures: vec![],
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new(
                    "pmap",
                    make_sig(&[it(t), fn_ty], it(u), interner),
                )
                .sync_handler(h_pmap),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(TyTerm::Bool),
                captures: vec![],
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new(
                    "filter",
                    make_sig(&[it(t.clone()), fn_ty], it(t), interner),
                )
                .sync_handler(h_filter),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let iter_ty = it(t);
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
            let iter_ty = it(t);
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
            let iter_ty = it(t);
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
            let iter_ty = it(t);
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
            fns.push(
                ExternFnBuilder::new(
                    "flatten",
                    make_sig(
                        &[it(TyTerm::List(Box::new(t.clone())))],
                        it(t),
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
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(it(u.clone())),
                captures: vec![],
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new(
                    "flat_map",
                    make_sig(&[it(t), fn_ty], it(u), interner),
                )
                .sync_handler(h_flat_map),
            );
        }

        // ── Consumers (async) ───────────────────────────
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            fns.push(
                ExternFnBuilder::new(
                    "collect",
                    make_sig(&[it(t.clone())], TyTerm::List(Box::new(t)), interner),
                )
                .async_handler(h_collect),
            );
        }
        {
            fns.push(
                ExternFnBuilder::new(
                    "join",
                    make_sig(&[it(TyTerm::String), TyTerm::String], TyTerm::String, interner),
                )
                .async_handler(h_join),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            fns.push(
                ExternFnBuilder::new(
                    "first",
                    make_sig(&[it(t.clone())], TyTerm::Option(Box::new(t)), interner),
                )
                .async_handler(h_first),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            fns.push(
                ExternFnBuilder::new(
                    "last",
                    make_sig(&[it(t.clone())], TyTerm::Option(Box::new(t)), interner),
                )
                .async_handler(h_last),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            fns.push(
                ExternFnBuilder::new(
                    "contains",
                    make_sig(&[it(t.clone()), t], TyTerm::Bool, interner),
                )
                .async_handler(h_contains),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let iter_ty = it(t.clone());
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
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(TyTerm::Bool),
                captures: vec![],
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new("find", make_sig(&[it(t.clone()), fn_ty], t, interner))
                    .async_handler(h_find),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone()), p(interner, 1, t.clone())],
                ret: Box::new(t.clone()),
                captures: vec![],
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new("reduce", make_sig(&[it(t.clone()), fn_ty], t, interner))
                    .async_handler(h_reduce),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let u = b.fresh_ty_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, u.clone()), p(interner, 1, t.clone())],
                ret: Box::new(u.clone()),
                captures: vec![],
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new("fold", make_sig(&[it(t), u.clone(), fn_ty], u, interner))
                    .async_handler(h_fold),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(TyTerm::Bool),
                captures: vec![],
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new("any", make_sig(&[it(t), fn_ty], TyTerm::Bool, interner))
                    .async_handler(h_any),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let fn_ty = TyTerm::Fn {
                params: vec![p(interner, 0, t.clone())],
                ret: Box::new(TyTerm::Bool),
                captures: vec![],
                hint: None,
            };
            fns.push(
                ExternFnBuilder::new("all", make_sig(&[it(t), fn_ty], TyTerm::Bool, interner))
                    .async_handler(h_all),
            );
        }

        fns
    })
}
