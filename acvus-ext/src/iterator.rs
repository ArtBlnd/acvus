//! Iterator functions.
//!
//! - Constructors: iter, iter_array, rev_iter
//! - Lazy combinators: map, pmap, filter, take, skip, chain, pchain, flatten,
//!   flatten_arrays, flat_map
//! - Consumers (async, effect E): collect, join, first, last, contains, next,
//!   find, reduce, fold, any, all

use acvus_extern::{
    Arr, EffectVar, ExternError, ExternRegistry, Fn1, Fn2, FromValue, Interner, IntoValue, LenVar,
    Runtime, TyVar, extern_fn, extern_registry,
};

use crate::iter_pipeline::Iter;
use crate::list::List;

fn count(name: &'static str, n: i64) -> Result<usize, ExternError> {
    usize::try_from(n).map_err(|_| ExternError::call(name, format!("negative count {n}")))
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn iter<T, E, Rt>(_: &Interner, items: List<T>) -> Iter<T, E, Rt>
where
    T: TyVar + IntoValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    Iter::from_items(items.0)
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn iter_array<T, N, E, Rt>(_: &Interner, items: Arr<T, N>) -> Iter<T, E, Rt>
where
    T: TyVar + IntoValue<Rt>,
    N: LenVar,
    E: EffectVar,
    Rt: Runtime,
{
    Iter::from_items(items.0)
}

#[extern_fn(effect = pure)]
fn rev_iter<T, E, Rt>(_: &Interner, items: List<T>) -> Iter<T, E, Rt>
where
    T: TyVar + IntoValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    let mut items = items.0;
    items.reverse();
    Iter::from_items(items)
}

#[extern_fn(effect = pure)]
fn map<T, U, E, Rt>(_: &Interner, it: Iter<T, E, Rt>, f: Fn1<T, U, E, Rt>) -> Iter<U, E, Rt>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    it.map(f.0)
}

#[extern_fn(effect = pure)]
fn pmap<T, U, E, Rt>(_: &Interner, it: Iter<T, E, Rt>, f: Fn1<T, U, E, Rt>) -> Iter<U, E, Rt>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    it.map(f.0)
}

#[extern_fn(effect = pure)]
fn filter<T, E, Rt>(_: &Interner, it: Iter<T, E, Rt>, f: Fn1<T, bool, E, Rt>) -> Iter<T, E, Rt>
where
    T: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    it.filter(f.0)
}

#[extern_fn(effect = pure)]
fn take<T, E, Rt>(_: &Interner, it: Iter<T, E, Rt>, n: i64) -> Result<Iter<T, E, Rt>, Rt::Error>
where
    T: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    Ok(it.take(count("take", n)?))
}

#[extern_fn(effect = pure)]
fn skip<T, E, Rt>(_: &Interner, it: Iter<T, E, Rt>, n: i64) -> Result<Iter<T, E, Rt>, Rt::Error>
where
    T: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    Ok(it.skip(count("skip", n)?))
}

#[extern_fn(effect = pure)]
fn chain<T, E, Rt>(_: &Interner, a: Iter<T, E, Rt>, b: Iter<T, E, Rt>) -> Iter<T, E, Rt>
where
    T: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    a.chain(b)
}

#[extern_fn(effect = pure)]
fn pchain<T, E, Rt>(_: &Interner, parts: List<Iter<T, E, Rt>>) -> Iter<T, E, Rt>
where
    T: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    parts.0.into_iter().fold(Iter::empty(), Iter::chain)
}

#[extern_fn(effect = pure)]
fn flatten<T, E, Rt>(_: &Interner, it: Iter<List<T>, E, Rt>) -> Iter<T, E, Rt>
where
    T: TyVar + FromValue<Rt> + IntoValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    it.flatten()
}

#[extern_fn(effect = pure)]
fn flatten_arrays<T, N, E, Rt>(_: &Interner, it: Iter<Arr<T, N>, E, Rt>) -> Iter<T, E, Rt>
where
    T: TyVar + FromValue<Rt> + IntoValue<Rt>,
    N: LenVar,
    E: EffectVar,
    Rt: Runtime,
{
    it.flatten()
}

#[extern_fn(effect = pure)]
fn flat_map<T, U, E, Rt>(
    _: &Interner,
    it: Iter<T, E, Rt>,
    f: Fn1<T, List<U>, E, Rt>,
) -> Iter<U, E, Rt>
where
    T: TyVar,
    U: TyVar + FromValue<Rt> + IntoValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    it.flat_map::<List<U>, U>(f.0)
}

#[extern_fn(effect = E)]
async fn collect<T, E, Rt>(i: Interner, mut it: Iter<T, E, Rt>) -> Result<List<T>, Rt::Error>
where
    T: TyVar + FromValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    let mut items = Vec::new();
    while let Some(item) = it.next(&i).await? {
        items.push(item);
    }
    Ok(List(items))
}

#[extern_fn(effect = E)]
async fn join<E, Rt>(
    i: Interner,
    mut it: Iter<String, E, Rt>,
    sep: String,
) -> Result<String, Rt::Error>
where
    E: EffectVar,
    Rt: Runtime,
{
    let mut parts = Vec::new();
    while let Some(part) = it.next(&i).await? {
        parts.push(part);
    }
    Ok(parts.join(&sep))
}

#[extern_fn(effect = E)]
async fn first<T, E, Rt>(i: Interner, mut it: Iter<T, E, Rt>) -> Result<Option<T>, Rt::Error>
where
    T: TyVar + FromValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    it.next(&i).await
}

#[extern_fn(effect = E)]
async fn last<T, E, Rt>(i: Interner, mut it: Iter<T, E, Rt>) -> Result<Option<T>, Rt::Error>
where
    T: TyVar + FromValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    let mut last = None;
    while let Some(item) = it.next(&i).await? {
        last = Some(item);
    }
    Ok(last)
}

#[extern_fn(effect = E)]
async fn contains<T, E, Rt>(
    i: Interner,
    mut it: Iter<T, E, Rt>,
    needle: T,
) -> Result<bool, Rt::Error>
where
    T: TyVar + FromValue<Rt> + IntoValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    let needle = needle.into_value(&i);
    while let Some(item) = it.next(&i).await? {
        if Rt::equals(&item.into_value(&i), &needle) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[extern_fn(effect = E)]
async fn next<T, E, Rt>(
    i: Interner,
    mut it: Iter<T, E, Rt>,
) -> Result<Option<(T, Iter<T, E, Rt>)>, Rt::Error>
where
    T: TyVar + FromValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    match it.next(&i).await? {
        Some(item) => Ok(Some((item, it))),
        None => Ok(None),
    }
}

#[extern_fn(effect = E)]
async fn find<T, E, Rt>(
    i: Interner,
    mut it: Iter<T, E, Rt>,
    f: Fn1<T, bool, E, Rt>,
) -> Result<T, Rt::Error>
where
    T: TyVar + FromValue<Rt> + IntoValue<Rt> + Clone,
    E: EffectVar,
    Rt: Runtime,
{
    while let Some(item) = it.next(&i).await? {
        if f.call(&i, item.clone()).await? {
            return Ok(item);
        }
    }
    Err(ExternError::call("find", "no element matched").into())
}

#[extern_fn(effect = E)]
async fn reduce<T, E, Rt>(
    i: Interner,
    mut it: Iter<T, E, Rt>,
    f: Fn2<T, T, T, E, Rt>,
) -> Result<T, Rt::Error>
where
    T: TyVar + FromValue<Rt> + IntoValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    let Some(mut acc) = it.next(&i).await? else {
        return Err(ExternError::call("reduce", "empty iterator").into());
    };
    while let Some(item) = it.next(&i).await? {
        acc = f.call(&i, acc, item).await?;
    }
    Ok(acc)
}

#[extern_fn(effect = E)]
async fn fold<T, U, E, Rt>(
    i: Interner,
    mut it: Iter<T, E, Rt>,
    init: U,
    f: Fn2<U, T, U, E, Rt>,
) -> Result<U, Rt::Error>
where
    T: TyVar + FromValue<Rt> + IntoValue<Rt>,
    U: TyVar + FromValue<Rt> + IntoValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    let mut acc = init;
    while let Some(item) = it.next(&i).await? {
        acc = f.call(&i, acc, item).await?;
    }
    Ok(acc)
}

#[extern_fn(effect = E)]
async fn any<T, E, Rt>(
    i: Interner,
    mut it: Iter<T, E, Rt>,
    f: Fn1<T, bool, E, Rt>,
) -> Result<bool, Rt::Error>
where
    T: TyVar + FromValue<Rt> + IntoValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    while let Some(item) = it.next(&i).await? {
        if f.call(&i, item).await? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[extern_fn(effect = E)]
async fn all<T, E, Rt>(
    i: Interner,
    mut it: Iter<T, E, Rt>,
    f: Fn1<T, bool, E, Rt>,
) -> Result<bool, Rt::Error>
where
    T: TyVar + FromValue<Rt> + IntoValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    while let Some(item) = it.next(&i).await? {
        if !f.call(&i, item).await? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn iterator_registry<Rt: Runtime>() -> ExternRegistry<Rt> {
    extern_registry! {
        types: [Iter<_, _, Rt>],
        fns: [
            iter, iter_array, rev_iter,
            map, pmap, filter, take, skip, chain, pchain, flatten, flatten_arrays, flat_map,
            collect, join, first, last, contains, next, find, reduce, fold, any, all,
        ],
    }
}
