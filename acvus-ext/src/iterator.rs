//! Iterator functions.
//!
//! - Constructors: iter, iter_array, rev_iter
//! - Lazy combinators: map, pmap, filter, take, skip, chain, pchain, flatten,
//!   flatten_arrays, flat_map
//! - Consumers (async, effect E): collect, join, first, last, contains, next,
//!   find, reduce, fold, any, all

use acvus_extern::{
    Arr, EffectVar, ExternRegistry, Fn1, Fn2, FromValue, Interner, LenVar, RuntimeError, TyVar,
    extern_fn, extern_registry,
};

use crate::iter_pipeline::{Iter, IterHandle, exec_next};
use crate::list::List;

fn count(name: &'static str, n: i64) -> Result<usize, RuntimeError> {
    usize::try_from(n).map_err(|_| RuntimeError::extern_call(name, format!("negative count {n}")))
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn iter<T, E>(i: &Interner, items: List<T>) -> Iter<T, E>
where
    T: TyVar,
    E: EffectVar,
{
    let values = items.0.into_iter().map(|v| v.into_value(i)).collect();
    Iter::new(IterHandle::from_list(values))
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn iter_array<T, N, E>(i: &Interner, items: Arr<T, N>) -> Iter<T, E>
where
    T: TyVar,
    N: LenVar,
    E: EffectVar,
{
    let values = items.0.into_iter().map(|v| v.into_value(i)).collect();
    Iter::new(IterHandle::from_list(values))
}

#[extern_fn(effect = pure)]
fn rev_iter<T, E>(i: &Interner, items: List<T>) -> Iter<T, E>
where
    T: TyVar,
    E: EffectVar,
{
    let values = items.0.into_iter().rev().map(|v| v.into_value(i)).collect();
    Iter::new(IterHandle::from_list(values))
}

#[extern_fn(effect = pure)]
fn map<T, U, E>(_: &Interner, it: Iter<T, E>, f: Fn1<T, U, E>) -> Iter<U, E>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
{
    Iter::new(it.0.map(f.0))
}

#[extern_fn(effect = pure)]
fn pmap<T, U, E>(_: &Interner, it: Iter<T, E>, f: Fn1<T, U, E>) -> Iter<U, E>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
{
    Iter::new(it.0.map(f.0))
}

#[extern_fn(effect = pure)]
fn filter<T, E>(_: &Interner, it: Iter<T, E>, f: Fn1<T, bool, E>) -> Iter<T, E>
where
    T: TyVar,
    E: EffectVar,
{
    Iter::new(it.0.filter(f.0))
}

#[extern_fn(effect = pure)]
fn take<T, E>(_: &Interner, it: Iter<T, E>, n: i64) -> Result<Iter<T, E>, RuntimeError>
where
    T: TyVar,
    E: EffectVar,
{
    Ok(Iter::new(it.0.take(count("take", n)?)))
}

#[extern_fn(effect = pure)]
fn skip<T, E>(_: &Interner, it: Iter<T, E>, n: i64) -> Result<Iter<T, E>, RuntimeError>
where
    T: TyVar,
    E: EffectVar,
{
    Ok(Iter::new(it.0.skip(count("skip", n)?)))
}

#[extern_fn(effect = pure)]
fn chain<T, E>(_: &Interner, a: Iter<T, E>, b: Iter<T, E>) -> Iter<T, E>
where
    T: TyVar,
    E: EffectVar,
{
    Iter::new(a.0.chain(b.0))
}

#[extern_fn(effect = pure)]
fn pchain<T, E>(_: &Interner, parts: List<Iter<T, E>>) -> Iter<T, E>
where
    T: TyVar,
    E: EffectVar,
{
    let mut chained = IterHandle::done();
    for part in parts.0 {
        chained = chained.chain(part.0);
    }
    Iter::new(chained)
}

#[extern_fn(effect = pure)]
fn flatten<T, E>(_: &Interner, it: Iter<List<T>, E>) -> Iter<T, E>
where
    T: TyVar,
    E: EffectVar,
{
    Iter::new(it.0.flatten())
}

#[extern_fn(effect = pure)]
fn flatten_arrays<T, N, E>(_: &Interner, it: Iter<Arr<T, N>, E>) -> Iter<T, E>
where
    T: TyVar,
    N: LenVar,
    E: EffectVar,
{
    Iter::new(it.0.flatten())
}

#[extern_fn(effect = pure)]
fn flat_map<T, U, E>(_: &Interner, it: Iter<T, E>, f: Fn1<T, Iter<U, E>, E>) -> Iter<U, E>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
{
    Iter::new(it.0.flat_map(f.0))
}

#[extern_fn(effect = E)]
async fn collect<T, E>(i: Interner, mut it: Iter<T, E>) -> Result<List<T>, RuntimeError>
where
    T: TyVar,
    E: EffectVar,
{
    let mut items = Vec::new();
    while let Some(val) = exec_next(&mut it.0).await? {
        items.push(T::from_value(val, &i)?);
    }
    Ok(List(items))
}

#[extern_fn(effect = E)]
async fn join<E>(i: Interner, mut it: Iter<String, E>, sep: String) -> Result<String, RuntimeError>
where
    E: EffectVar,
{
    let mut parts = Vec::new();
    while let Some(val) = exec_next(&mut it.0).await? {
        parts.push(String::from_value(val, &i)?);
    }
    Ok(parts.join(&sep))
}

#[extern_fn(effect = E)]
async fn first<T, E>(i: Interner, mut it: Iter<T, E>) -> Result<Option<T>, RuntimeError>
where
    T: TyVar,
    E: EffectVar,
{
    match exec_next(&mut it.0).await? {
        Some(val) => Ok(Some(T::from_value(val, &i)?)),
        None => Ok(None),
    }
}

#[extern_fn(effect = E)]
async fn last<T, E>(i: Interner, mut it: Iter<T, E>) -> Result<Option<T>, RuntimeError>
where
    T: TyVar,
    E: EffectVar,
{
    let mut last = None;
    while let Some(val) = exec_next(&mut it.0).await? {
        last = Some(val);
    }
    match last {
        Some(val) => Ok(Some(T::from_value(val, &i)?)),
        None => Ok(None),
    }
}

#[extern_fn(effect = E)]
async fn contains<T, E>(i: Interner, mut it: Iter<T, E>, needle: T) -> Result<bool, RuntimeError>
where
    T: TyVar,
    E: EffectVar,
{
    let needle = needle.into_value(&i);
    while let Some(val) = exec_next(&mut it.0).await? {
        if val.structural_eq(&needle) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[extern_fn(effect = E)]
async fn next<T, E>(
    i: Interner,
    mut it: Iter<T, E>,
) -> Result<Option<(T, Iter<T, E>)>, RuntimeError>
where
    T: TyVar,
    E: EffectVar,
{
    match exec_next(&mut it.0).await? {
        Some(val) => Ok(Some((T::from_value(val, &i)?, it))),
        None => Ok(None),
    }
}

#[extern_fn(effect = E)]
async fn find<T, E>(i: Interner, mut it: Iter<T, E>, f: Fn1<T, bool, E>) -> Result<T, RuntimeError>
where
    T: TyVar,
    E: EffectVar,
{
    while let Some(val) = exec_next(&mut it.0).await? {
        if f.call(&i, T::from_value(val.clone(), &i)?).await? {
            return Ok(T::from_value(val, &i)?);
        }
    }
    Err(RuntimeError::empty_collection(
        acvus_interpreter::error::CollectionOp::Find,
    ))
}

#[extern_fn(effect = E)]
async fn reduce<T, E>(
    i: Interner,
    mut it: Iter<T, E>,
    f: Fn2<T, T, T, E>,
) -> Result<T, RuntimeError>
where
    T: TyVar,
    E: EffectVar,
{
    let Some(first) = exec_next(&mut it.0).await? else {
        return Err(RuntimeError::empty_collection(
            acvus_interpreter::error::CollectionOp::Reduce,
        ));
    };
    let mut acc = T::from_value(first, &i)?;
    while let Some(val) = exec_next(&mut it.0).await? {
        acc = f.call(&i, acc, T::from_value(val, &i)?).await?;
    }
    Ok(acc)
}

#[extern_fn(effect = E)]
async fn fold<T, U, E>(
    i: Interner,
    mut it: Iter<T, E>,
    init: U,
    f: Fn2<U, T, U, E>,
) -> Result<U, RuntimeError>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
{
    let mut acc = init;
    while let Some(val) = exec_next(&mut it.0).await? {
        acc = f.call(&i, acc, T::from_value(val, &i)?).await?;
    }
    Ok(acc)
}

#[extern_fn(effect = E)]
async fn any<T, E>(
    i: Interner,
    mut it: Iter<T, E>,
    f: Fn1<T, bool, E>,
) -> Result<bool, RuntimeError>
where
    T: TyVar,
    E: EffectVar,
{
    while let Some(val) = exec_next(&mut it.0).await? {
        if f.call(&i, T::from_value(val, &i)?).await? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[extern_fn(effect = E)]
async fn all<T, E>(
    i: Interner,
    mut it: Iter<T, E>,
    f: Fn1<T, bool, E>,
) -> Result<bool, RuntimeError>
where
    T: TyVar,
    E: EffectVar,
{
    while let Some(val) = exec_next(&mut it.0).await? {
        if !f.call(&i, T::from_value(val, &i)?).await? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn iterator_registry() -> ExternRegistry {
    extern_registry! {
        types: [Iter<_, _>],
        fns: [
            iter, iter_array, rev_iter,
            map, pmap, filter, take, skip, chain, pchain, flatten, flatten_arrays, flat_map,
            collect, join, first, last, contains, next, find, reduce, fold, any, all,
        ],
    }
}
