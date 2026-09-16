//! Iterator functions.
//!
//! - Constructors: the shared signatures `iter::into_iter` (consuming) and
//!   `iter::as_iter` (over a borrowed container, yielding references), with
//!   instances for `Vec` and `Array` here and for `Deque` in `deque`;
//!   rev_iter (consuming)
//! - Lazy combinators: map, pmap, filter, take, skip, chain, pchain, flatten,
//!   flatten_arrays, flat_map
//! - Consumers (async, effect E): collect, join, contains, next, find,
//!   reduce, fold, any, all

use acvus_extern::{
    Arr, ClosureFn, EffectVar, Fn1, Fn2, IdentityVar, LenVar, Ref, Registry, Runtime, Trap, TyVar,
    extern_fn, extern_registry,
};

use crate::iter_pipeline::Iter;

/// The shared signatures every container declares instances of (RFC-0027).
pub mod sig {
    use acvus_extern::{Ref, extern_signature};

    use crate::iter_pipeline::Iter;

    extern_signature! {
        ns: "iter",
        fn into_iter<C, T, E, I, Rt>(items: C) -> Iter<T, E, I, Rt>
        where
            C: TyVar,
            T: TyVar,
            E: EffectVar,
            I: IdentityVar,
            Rt: Runtime;
    }

    extern_signature! {
        ns: "iter",
        fn as_iter<C, T, E, I, Rt>(items: &C) -> Iter<Ref<T, Rt>, E, I, Rt>
        where
            C: TyVar,
            T: TyVar,
            E: EffectVar,
            I: IdentityVar,
            Rt: Runtime;
    }
}

/// An iterator over references into a borrowed container, read by `at`.
pub(crate) fn lent_iter<C, T, E, I, Rt>(
    items: Ref<C, Rt>,
    at: impl Fn(&C, usize) -> Option<&T> + Send + Sync + 'static,
) -> Iter<Ref<T, Rt>, E, I, Rt>
where
    C: TyVar,
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut index = 0;
    Iter::from_fn(move |rt| {
        let item = items.try_map(rt, |container| at(container, index));
        index += 1;
        item.map(Ref::into_value)
    })
}

#[extern_fn(instance_of = sig::into_iter, effect = pure)]
#[extern_cast]
fn into_iter_vec<T, E, I, Rt>(items: Vec<T>) -> Iter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    Iter::from_items(items)
}

#[extern_fn(instance_of = sig::into_iter, effect = pure)]
#[extern_cast]
fn into_iter_array<T, N, E, I, Rt>(items: Arr<T, N>) -> Iter<T, E, I, Rt>
where
    T: TyVar,
    N: LenVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    Iter::from_items(items.0)
}

#[extern_fn(instance_of = sig::as_iter, effect = pure)]
fn as_iter_vec<T, E, I, Rt>(items: Ref<Vec<T>, Rt>) -> Iter<Ref<T, Rt>, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    lent_iter(items, |items, i| items.get(i))
}

#[extern_fn(instance_of = sig::as_iter, effect = pure)]
fn as_iter_array<T, N, E, I, Rt>(items: Ref<Arr<T, N>, Rt>) -> Iter<Ref<T, Rt>, E, I, Rt>
where
    T: TyVar,
    N: LenVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    lent_iter(items, |array, i| array.0.get(i))
}

#[extern_fn(effect = pure)]
fn rev_iter<T, E, I, Rt>(items: Vec<T>) -> Iter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut items = items;
    items.reverse();
    Iter::from_items(items)
}

#[extern_fn(effect = pure)]
fn map<T, U, E, I, Rt>(it: Iter<T, E, I, Rt>, f: Fn1<T, U, E, Rt>) -> Iter<U, E, I, Rt>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.map(f)
}

#[extern_fn(effect = pure)]
fn pmap<T, U, E, I, Rt>(it: Iter<T, E, I, Rt>, f: Fn1<T, U, E, Rt>) -> Iter<U, E, I, Rt>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.map(f)
}

#[extern_fn(effect = pure)]
fn filter<T, E, I, Rt>(it: Iter<T, E, I, Rt>, f: Fn1<Ref<T, Rt>, bool, E, Rt>) -> Iter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.filter(f)
}

#[extern_fn(effect = pure)]
fn take<T, E, I, Rt>(it: Iter<T, E, I, Rt>, n: u64) -> Iter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.take(n)
}

#[extern_fn(effect = pure)]
fn skip<T, E, I, Rt>(it: Iter<T, E, I, Rt>, n: u64) -> Iter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.skip(n)
}

#[extern_fn(effect = pure)]
fn chain<T, E, I, J, K, Rt>(a: Iter<T, E, I, Rt>, b: Iter<T, E, J, Rt>) -> Iter<T, E, K, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    J: IdentityVar,
    K: IdentityVar,
    Rt: Runtime,
{
    a.chain(b)
}

#[extern_fn(effect = pure)]
fn pchain<T, E, I, K, Rt>(parts: Vec<Iter<T, E, I, Rt>>) -> Iter<T, E, K, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    K: IdentityVar,
    Rt: Runtime,
{
    parts
        .into_iter()
        .fold(Iter::<T, E, K, Rt>::empty(), |acc, part| acc.chain(part))
}

#[extern_fn(effect = pure)]
fn flatten<T, E, I, Rt>(it: Iter<Vec<T>, E, I, Rt>) -> Iter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.flatten()
}

#[extern_fn(effect = pure)]
fn flatten_arrays<T, N, E, I, Rt>(it: Iter<Arr<T, N>, E, I, Rt>) -> Iter<T, E, I, Rt>
where
    T: TyVar,
    N: LenVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.flatten()
}

#[extern_fn(effect = pure)]
fn flat_map<T, U, E, I, Rt>(it: Iter<T, E, I, Rt>, f: Fn1<T, Vec<U>, E, Rt>) -> Iter<U, E, I, Rt>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.flat_map::<Vec<U>, U>(f)
}

#[extern_fn(effect = E)]
async fn collect<T, E, I, Rt>(rt: &Rt, mut it: Iter<T, E, I, Rt>) -> Result<Vec<T>, Rt::Error>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut items = Vec::new();
    while let Some(item) = it.next(rt).await? {
        items.push(item);
    }
    Ok(items)
}

#[extern_fn(effect = E)]
async fn join<E, I, Rt>(
    rt: &Rt,
    mut it: Iter<String, E, I, Rt>,
    sep: String,
) -> Result<String, Rt::Error>
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut parts = Vec::new();
    while let Some(part) = it.next(rt).await? {
        parts.push(part);
    }
    Ok(parts.join(&sep))
}

#[extern_fn(effect = E)]
async fn contains<T, E, I, Rt>(
    rt: &Rt,
    mut it: Iter<T, E, I, Rt>,
    needle: T,
) -> Result<bool, Rt::Error>
where
    T: acvus_extern::Monomorphize<(i64, f64, bool, u8, String)> + PartialEq,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    while let Some(item) = it.next(rt).await? {
        if item == needle {
            return Ok(true);
        }
    }
    Ok(false)
}

#[extern_fn(effect = E)]
async fn next<T, E, I, Rt>(rt: &Rt, it: &mut Iter<T, E, I, Rt>) -> Result<Option<T>, Rt::Error>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.next(rt).await
}

#[extern_fn(effect = E)]
async fn find<T, E, I, Rt>(
    rt: &Rt,
    mut it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
) -> Result<T, Rt::Error>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    while let Some(item) = it.next(rt).await? {
        let lent = unsafe { rt.erase::<T>(item) };
        let keep = f.call(rt, (unsafe { rt.reference(&lent) },)).await;
        let item = unsafe { rt.materialize::<T>(lent) };
        if unsafe { rt.materialize::<bool>(keep?) } {
            return Ok(item);
        }
    }
    Err(Trap::call("find", "no element matched").into())
}

#[extern_fn(effect = E)]
async fn reduce<T, E, I, Rt>(
    rt: &Rt,
    mut it: Iter<T, E, I, Rt>,
    f: Fn2<T, T, T, E, Rt>,
) -> Result<T, Rt::Error>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let Some(mut acc) = it.next(rt).await? else {
        return Err(Trap::call("reduce", "empty iterator").into());
    };
    while let Some(item) = it.next(rt).await? {
        let out = f
            .call(
                rt,
                (unsafe { rt.erase::<T>(acc) }, unsafe {
                    rt.erase::<T>(item)
                }),
            )
            .await?;
        acc = unsafe { rt.materialize::<T>(out) };
    }
    Ok(acc)
}

#[extern_fn(effect = E)]
async fn fold<T, U, E, I, Rt>(
    rt: &Rt,
    mut it: Iter<T, E, I, Rt>,
    init: U,
    f: Fn2<U, T, U, E, Rt>,
) -> Result<U, Rt::Error>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut acc = init;
    while let Some(item) = it.next(rt).await? {
        let out = f
            .call(
                rt,
                (unsafe { rt.erase::<U>(acc) }, unsafe {
                    rt.erase::<T>(item)
                }),
            )
            .await?;
        acc = unsafe { rt.materialize::<U>(out) };
    }
    Ok(acc)
}

#[extern_fn(effect = E)]
async fn any<T, E, I, Rt>(
    rt: &Rt,
    mut it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
) -> Result<bool, Rt::Error>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    while let Some(item) = it.next(rt).await? {
        let lent = unsafe { rt.erase::<T>(item) };
        let keep = f.call(rt, (unsafe { rt.reference(&lent) },)).await;
        drop(unsafe { rt.materialize::<T>(lent) });
        if unsafe { rt.materialize::<bool>(keep?) } {
            return Ok(true);
        }
    }
    Ok(false)
}

#[extern_fn(effect = E)]
async fn all<T, E, I, Rt>(
    rt: &Rt,
    mut it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
) -> Result<bool, Rt::Error>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    while let Some(item) = it.next(rt).await? {
        let lent = unsafe { rt.erase::<T>(item) };
        let keep = f.call(rt, (unsafe { rt.reference(&lent) },)).await;
        drop(unsafe { rt.materialize::<T>(lent) });
        if !unsafe { rt.materialize::<bool>(keep?) } {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn iterator_registry<Rt>() -> Registry<Rt>
where
    Rt: Runtime,
{
    extern_registry! {
        ns: "std",
        types: [Iter<_, _, _, Rt>],
        signatures: [sig::into_iter, sig::as_iter],
        fns: [
            into_iter_vec, into_iter_array, as_iter_vec, as_iter_array, rev_iter,
            map, pmap, filter, take, skip, chain, pchain, flatten, flatten_arrays, flat_map,
            collect, join, contains, next, find, reduce, fold, any, all,
        ],
    }
}
