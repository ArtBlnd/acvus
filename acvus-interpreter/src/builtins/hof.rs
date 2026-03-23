//! Async iterator-consuming builtins (HOFs and terminal operations).
//!
//! Each function uses phantom-typed parameters for self-describing signatures,
//! matching the same pattern as sync builtins in `iter.rs`, `seq.rs`, etc.
//! All execution is delegated to [`ExecCtx`] — no dependency on `Interpreter`.

use std::marker::PhantomData;
use std::sync::Arc;

use acvus_mir::builtins::BuiltinId;
use acvus_mir::graph::ContextId;
use acvus_utils::YieldHandle;

use super::exec_ctx::{collect_vec, ExecCtx};
use super::types::{Deq, FromTyped, Fun1, Fun2, IntoTyped, Iter, Opt, TVal, E, O, T};
use crate::error::{CollectionOp, RuntimeError, ValueKind};
use crate::value::{PureValue, TypedValue, Value};

// ── Iterator consumers ──────────────────────────────────────────────

/// `first : Iterator<T, E> -> Option<T>`
pub async fn first_iter_impl<'a, Ctx: ExecCtx>(
    ctx: Ctx,
    iter: Iter<T<0>, E<0>>,
    handle: &'a YieldHandle<TypedValue, ContextId>,
) -> Result<(Ctx, Opt<T<0>>), RuntimeError> {
    let (ctx, result) = ctx.exec_next(iter.0, handle).await?;
    match result {
        Some((item, _rest)) => Ok((ctx, Opt(Some(item), PhantomData))),
        None => Ok((ctx, Opt(None, PhantomData))),
    }
}

/// `last : Iterator<T, E> -> Option<T>`
pub async fn last_iter_impl<'a, Ctx: ExecCtx>(
    mut ctx: Ctx,
    iter: Iter<T<0>, E<0>>,
    handle: &'a YieldHandle<TypedValue, ContextId>,
) -> Result<(Ctx, Opt<T<0>>), RuntimeError> {
    let mut current = iter.0;
    let mut last_item: Option<Value> = None;
    loop {
        let result;
        (ctx, result) = ctx.exec_next(current, handle).await?;
        match result {
            Some((item, rest)) => {
                last_item = Some(item);
                current = rest;
            }
            None => break,
        }
    }
    Ok((ctx, Opt(last_item, PhantomData)))
}

/// `contains : Iterator<T, E> -> T -> Bool`
pub async fn contains_iter_impl<'a, Ctx: ExecCtx>(
    mut ctx: Ctx,
    iter: Iter<T<0>, E<0>>,
    needle: TVal<T<0>>,
    handle: &'a YieldHandle<TypedValue, ContextId>,
) -> Result<(Ctx, bool), RuntimeError> {
    let mut current = iter.0;
    loop {
        let result;
        (ctx, result) = ctx.exec_next(current, handle).await?;
        match result {
            Some((item, rest)) => {
                if item.structural_eq(&needle.0) {
                    return Ok((ctx, true));
                }
                current = rest;
            }
            None => return Ok((ctx, false)),
        }
    }
}

/// `next : Iterator<T, E> -> Option<(T, Iterator<T, E>)>`
///
/// Return type: `Opt<Tup2<TVal<T<0>>, Iter<T<0>, E<0>>>>` (self-describing).
/// The `Tup2` is phantom-only — the actual value is `Value::tuple(...)`.
pub async fn next_impl<'a, Ctx: ExecCtx>(
    ctx: Ctx,
    iter: Iter<T<0>, E<0>>,
    handle: &'a YieldHandle<TypedValue, ContextId>,
) -> Result<(Ctx, Opt<T<0>>), RuntimeError> {
    let (ctx, result) = ctx.exec_next(iter.0, handle).await?;
    match result {
        None => Ok((ctx, Opt(None, PhantomData))),
        Some((item, rest)) => Ok((
            ctx,
            Opt(
                Some(Value::tuple(vec![item, Value::iterator(rest)])),
                PhantomData,
            ),
        )),
    }
}

/// `collect : Iterator<T, E> -> [T]`
pub async fn collect_impl<'a, Ctx: ExecCtx>(
    ctx: Ctx,
    iter: Iter<T<0>, E<0>>,
    handle: &'a YieldHandle<TypedValue, ContextId>,
) -> Result<(Ctx, Vec<Value>), RuntimeError> {
    collect_vec(ctx, iter.0, handle).await
}

/// `join : Iterator<String, E> -> String -> String`
pub async fn join_impl<'a, Ctx: ExecCtx>(
    ctx: Ctx,
    iter: Iter<String, E<0>>,
    sep: String,
    handle: &'a YieldHandle<TypedValue, ContextId>,
) -> Result<(Ctx, String), RuntimeError> {
    let (ctx, items) = collect_vec(ctx, iter.0, handle).await?;
    let mut parts = Vec::with_capacity(items.len());
    for v in items {
        match v {
            Value::Pure(PureValue::String(s)) => parts.push(s),
            other => {
                return Err(RuntimeError::unexpected_type(
                    "join",
                    &[ValueKind::String],
                    other.kind(),
                ))
            }
        }
    }
    Ok((ctx, parts.join(&sep)))
}

// ── Higher-order functions ──────────────────────────────────────────

/// `find : Iterator<T, E> -> (T -> Bool) -> T`
///
/// Errors with `EmptyCollection(Find)` if no element matches.
pub async fn find_impl<'a, Ctx: ExecCtx>(
    mut ctx: Ctx,
    iter: Iter<T<0>, E<0>>,
    f: Fun1<T<0>, bool, E<0>>,
    handle: &'a YieldHandle<TypedValue, ContextId>,
) -> Result<(Ctx, TVal<T<0>>), RuntimeError> {
    let (ctx2, items) = collect_vec(ctx, iter.0, handle).await?;
    ctx = ctx2;
    for item in items {
        let arc_item = Arc::new(item);
        let matched;
        (ctx, matched) = ctx
            .call_closure(f.0.clone(), vec![Arc::clone(&arc_item)], handle)
            .await?;
        if matches!(matched, Value::Pure(PureValue::Bool(true))) {
            return Ok((ctx, TVal(Arc::unwrap_or_clone(arc_item), PhantomData)));
        }
    }
    Err(RuntimeError::empty_collection(CollectionOp::Find))
}

/// `reduce : Iterator<T, E> -> (T -> T -> T) -> T`
///
/// Errors with `EmptyCollection(Reduce)` if the iterator is empty.
pub async fn reduce_impl<'a, Ctx: ExecCtx>(
    mut ctx: Ctx,
    iter: Iter<T<0>, E<0>>,
    f: Fun2<T<0>, T<0>, T<0>, E<0>>,
    handle: &'a YieldHandle<TypedValue, ContextId>,
) -> Result<(Ctx, TVal<T<0>>), RuntimeError> {
    let (ctx2, items) = collect_vec(ctx, iter.0, handle).await?;
    ctx = ctx2;
    let mut it = items.into_iter();
    let Some(mut acc) = it.next() else {
        return Err(RuntimeError::empty_collection(CollectionOp::Reduce));
    };
    for item in it {
        (ctx, acc) = ctx
            .call_closure(f.0.clone(), vec![Arc::new(acc), Arc::new(item)], handle)
            .await?;
    }
    Ok((ctx, TVal(acc, PhantomData)))
}

/// `fold : Iterator<T, E> -> A -> (A -> T -> A) -> A`
pub async fn fold_impl<'a, Ctx: ExecCtx>(
    mut ctx: Ctx,
    iter: Iter<T<0>, E<0>>,
    init: TVal<T<1>>,
    f: Fun2<T<1>, T<0>, T<1>, E<0>>,
    handle: &'a YieldHandle<TypedValue, ContextId>,
) -> Result<(Ctx, TVal<T<1>>), RuntimeError> {
    let (ctx2, items) = collect_vec(ctx, iter.0, handle).await?;
    ctx = ctx2;
    let mut acc = init.0;
    for item in items {
        (ctx, acc) = ctx
            .call_closure(f.0.clone(), vec![Arc::new(acc), Arc::new(item)], handle)
            .await?;
    }
    Ok((ctx, TVal(acc, PhantomData)))
}

/// `any : Iterator<T, E> -> (T -> Bool) -> Bool`
pub async fn any_impl<'a, Ctx: ExecCtx>(
    mut ctx: Ctx,
    iter: Iter<T<0>, E<0>>,
    f: Fun1<T<0>, bool, E<0>>,
    handle: &'a YieldHandle<TypedValue, ContextId>,
) -> Result<(Ctx, bool), RuntimeError> {
    let (ctx2, items) = collect_vec(ctx, iter.0, handle).await?;
    ctx = ctx2;
    for item in items {
        let result;
        (ctx, result) = ctx
            .call_closure(f.0.clone(), vec![Arc::new(item)], handle)
            .await?;
        if matches!(result, Value::Pure(PureValue::Bool(true))) {
            return Ok((ctx, true));
        }
    }
    Ok((ctx, false))
}

/// `all : Iterator<T, E> -> (T -> Bool) -> Bool`
pub async fn all_impl<'a, Ctx: ExecCtx>(
    mut ctx: Ctx,
    iter: Iter<T<0>, E<0>>,
    f: Fun1<T<0>, bool, E<0>>,
    handle: &'a YieldHandle<TypedValue, ContextId>,
) -> Result<(Ctx, bool), RuntimeError> {
    let (ctx2, items) = collect_vec(ctx, iter.0, handle).await?;
    ctx = ctx2;
    for item in items {
        let result;
        (ctx, result) = ctx
            .call_closure(f.0.clone(), vec![Arc::new(item)], handle)
            .await?;
        if matches!(result, Value::Pure(PureValue::Bool(false))) {
            return Ok((ctx, false));
        }
    }
    Ok((ctx, true))
}

/// `extend : Deque<T, O> -> Iterator<T, E> -> Deque<T, O>`
pub async fn extend_impl<'a, Ctx: ExecCtx>(
    ctx: Ctx,
    deque: Deq<T<0>, O<0>>,
    iter: Iter<T<0>, E<0>>,
    handle: &'a YieldHandle<TypedValue, ContextId>,
) -> Result<(Ctx, Deq<T<0>, O<0>>), RuntimeError> {
    let mut d = deque.0;
    let (ctx, items) = collect_vec(ctx, iter.0, handle).await?;
    d.extend(items);
    Ok((ctx, Deq(d, PhantomData)))
}

// ── Dispatch ────────────────────────────────────────────────────────

/// Dispatch an async builtin by ID.
///
/// Extracts arguments via `FromTyped` (phantom-typed wrappers), calls the
/// appropriate implementation, and converts the result back to `Value`.
///
/// Returns `None` if the given `BuiltinId` is not an async builtin.
pub async fn dispatch<'a, Ctx: ExecCtx>(
    ctx: Ctx,
    id: BuiltinId,
    args: Vec<TypedValue>,
    handle: &'a YieldHandle<TypedValue, ContextId>,
) -> Result<Option<(Ctx, Value)>, RuntimeError> {
    let mut it = args.into_iter();
    match id {
        // -- Iterator consumers --
        BuiltinId::First => {
            let iter = Iter::from_typed(it.next().expect("missing arg"))?;
            let (ctx, opt) = first_iter_impl(ctx, iter, handle).await?;
            Ok(Some((ctx, opt.into_typed().into_inner())))
        }
        BuiltinId::Last => {
            let iter = Iter::from_typed(it.next().expect("missing arg"))?;
            let (ctx, opt) = last_iter_impl(ctx, iter, handle).await?;
            Ok(Some((ctx, opt.into_typed().into_inner())))
        }
        BuiltinId::Contains => {
            let iter = Iter::from_typed(it.next().expect("missing arg"))?;
            let needle = TVal::from_typed(it.next().expect("missing arg"))?;
            let (ctx, found) = contains_iter_impl(ctx, iter, needle, handle).await?;
            Ok(Some((ctx, Value::bool_(found))))
        }
        BuiltinId::Next | BuiltinId::NextSeq => {
            let iter = Iter::from_typed(it.next().expect("missing arg"))?;
            let (ctx, opt) = next_impl(ctx, iter, handle).await?;
            Ok(Some((ctx, opt.into_typed().into_inner())))
        }
        BuiltinId::Collect => {
            let iter = Iter::from_typed(it.next().expect("missing arg"))?;
            let (ctx, items) = collect_impl(ctx, iter, handle).await?;
            Ok(Some((ctx, Value::list(items))))
        }
        BuiltinId::Join => {
            let iter = Iter::from_typed(it.next().expect("missing arg"))?;
            let sep = String::from_typed(it.next().expect("missing arg"))?;
            let (ctx, joined) = join_impl(ctx, iter, sep, handle).await?;
            Ok(Some((ctx, Value::string(joined))))
        }

        // -- Higher-order functions --
        BuiltinId::Find => {
            let iter = Iter::from_typed(it.next().expect("missing arg"))?;
            let f = Fun1::from_typed(it.next().expect("missing arg"))?;
            let (ctx, result) = find_impl(ctx, iter, f, handle).await?;
            Ok(Some((ctx, result.0)))
        }
        BuiltinId::Reduce => {
            let iter = Iter::from_typed(it.next().expect("missing arg"))?;
            let f = Fun2::from_typed(it.next().expect("missing arg"))?;
            let (ctx, result) = reduce_impl(ctx, iter, f, handle).await?;
            Ok(Some((ctx, result.0)))
        }
        BuiltinId::Fold => {
            let iter = Iter::from_typed(it.next().expect("missing arg"))?;
            let init = TVal::from_typed(it.next().expect("missing arg"))?;
            let f = Fun2::from_typed(it.next().expect("missing arg"))?;
            let (ctx, result) = fold_impl(ctx, iter, init, f, handle).await?;
            Ok(Some((ctx, result.0)))
        }
        BuiltinId::Any => {
            let iter = Iter::from_typed(it.next().expect("missing arg"))?;
            let f = Fun1::from_typed(it.next().expect("missing arg"))?;
            let (ctx, found) = any_impl(ctx, iter, f, handle).await?;
            Ok(Some((ctx, Value::bool_(found))))
        }
        BuiltinId::All => {
            let iter = Iter::from_typed(it.next().expect("missing arg"))?;
            let f = Fun1::from_typed(it.next().expect("missing arg"))?;
            let (ctx, found) = all_impl(ctx, iter, f, handle).await?;
            Ok(Some((ctx, Value::bool_(found))))
        }

        // -- Deque extend --
        BuiltinId::Extend => {
            let deque = Deq::from_typed(it.next().expect("missing arg"))?;
            let iter = Iter::from_typed(it.next().expect("missing arg"))?;
            let (ctx, result) = extend_impl(ctx, deque, iter, handle).await?;
            Ok(Some((ctx, result.into_typed().into_inner())))
        }

        // Not an async builtin
        _ => Ok(None),
    }
}
