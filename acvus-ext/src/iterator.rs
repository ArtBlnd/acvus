//! Iterator functions.
//!
//! - Constructors: the shared signatures `iter::into_iter` (consuming) and
//!   `iter::as_iter` (over a borrowed container, yielding references), with
//!   instances for `Vec` and `Array` here and for `Deque` in `deque`;
//!   rev_iter (consuming); range, range_step
//! - Lazy combinators: map, pmap, filter, take, skip, step_by, take_while,
//!   skip_while, chunks, dedup, chain, pchain, flatten, flatten_arrays,
//!   flat_map
//! - Consumers (async, effect E): collect, join, contains, next, find,
//!   reduce, fold, any, all, count, last, nth, position, min_by_key,
//!   max_by_key (the key is `i64`)
//! - Aggregates over a numeric element (async, effect E): sum, product,
//!   min, max
//!
//! A typed per-element operation takes `Iter<Erased<Rt, T>>` with `T`
//! bounded by `Monomorphize`: the element is read in place through
//! `Erased::as_ref`, and the `Iter` slot stays uniform (RFC-0041).
//!
//! Not here, each an `acvus-extern` contract: `enumerate`, `zip`,
//! `partition` yield a tuple, and no tuple implements `Cross` (a tuple has
//! `TyArg` only); `repeat` needs a clone of a runtime value, which
//! `Runtime` does not offer; `min_by_key`/`max_by_key` take an `i64` key
//! and not a `Monomorphize<(i64, f64)>` member, because a member fn's glue
//! crosses every parameter naming the member through `CrossSpecialized`,
//! which `Fn1` does not implement.

use acvus_extern::{
    Arr, ClosureFn, EffectVar, Erased, Fn1, Fn2, FromValue, IdentityVar, LenVar, Monomorphize,
    OneValue, Ref, Registry, Runtime, Stored, TransparentOver, TyVar, extern_fn, extern_registry,
};

use crate::iter::{Iter, drain, drain_now};

/// The arithmetic the aggregates need of a `Monomorphize<(i64, f64)>`
/// member; `add` and `mul` are what `Iterator::sum` and `Iterator::product`
/// do in a release build.
trait Num: Copy + Send + Sync + 'static {
    const ZERO: Self;
    const ONE: Self;
    fn add(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
}

impl Num for i64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;

    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }

    fn mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }

    fn min(self, other: Self) -> Self {
        Ord::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        Ord::max(self, other)
    }
}

/// `min` and `max` are `f64::min`/`f64::max`: when one side is NaN the
/// other side is the result, so a NaN never wins unless every element is
/// NaN.
impl Num for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn mul(self, other: Self) -> Self {
        self * other
    }

    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
}

/// The shared signatures every container declares instances of (RFC-0027).
pub mod sig {
    use acvus_extern::{Ref, extern_signature};

    use crate::iter::Iter;

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
    T: TyVar + TransparentOver<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut index = 0;
    Iter::generate(move |rt| {
        let item = items.try_map(rt, |container| at(container, index));
        index += 1;
        item
    })
}

#[extern_fn(instance_of = sig::into_iter, effect = pure)]
#[extern_cast]
fn into_iter_vec<T, E, I, Rt>(items: Vec<T>) -> Iter<T, E, I, Rt>
where
    T: TyVar + OneValue<Rt>,
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
    T: TyVar + OneValue<Rt>,
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
    T: TyVar + TransparentOver<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    lent_iter(items, |items, i| items.get(i))
}

#[extern_fn(instance_of = sig::as_iter, effect = pure)]
fn as_iter_array<T, N, E, I, Rt>(items: Ref<Arr<T, N>, Rt>) -> Iter<Ref<T, Rt>, E, I, Rt>
where
    T: TyVar + TransparentOver<Rt>,
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
    T: TyVar + OneValue<Rt>,
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
    Iter::chain_all(parts)
}

#[extern_fn(effect = pure)]
fn flatten<T, E, I, Rt>(it: Iter<Vec<T>, E, I, Rt>) -> Iter<T, E, I, Rt>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.flatten()
}

#[extern_fn(effect = pure)]
fn flatten_arrays<T, N, E, I, Rt>(it: Iter<Arr<T, N>, E, I, Rt>) -> Iter<T, E, I, Rt>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
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
    U: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.flat_map::<Vec<U>, U>(f)
}

fn collect_now<T, E, I, Rt>(rt: &Rt, frame: &mut Rt::Frame<'_>, mut it: Iter<T, E, I, Rt>) -> Vec<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut items = Vec::new();
    drain_now!(it, rt, frame, |value| {
        items.push(T::from_value(rt, value));
    });
    items
}

#[extern_fn(effect = E, sync = collect_now)]
async fn collect<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
) -> Vec<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut items = Vec::new();
    drain!(it, rt, frame, |value| {
        items.push(T::from_value(rt, value));
    });
    items
}

fn join_now<E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<Erased<Rt, String>, E, I, Rt>,
    sep: String,
) -> String
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut parts: Vec<String> = Vec::new();
    drain_now!(it, rt, frame, |value| {
        parts.push(Erased::<Rt, String>::from_value(rt, value).into_inner(rt));
    });
    parts.join(&sep)
}

#[extern_fn(effect = E, sync = join_now)]
async fn join<E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<Erased<Rt, String>, E, I, Rt>,
    sep: String,
) -> String
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut parts: Vec<String> = Vec::new();
    drain!(it, rt, frame, |value| {
        parts.push(Erased::<Rt, String>::from_value(rt, value).into_inner(rt));
    });
    parts.join(&sep)
}

fn contains_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<Erased<Rt, T>, E, I, Rt>,
    needle: T,
) -> bool
where
    T: acvus_extern::Monomorphize<(i64, f64, bool, u8, String)> + Stored<Rt> + PartialEq,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut found = false;
    drain_now!(it, rt, frame, |value| {
        if *Erased::<Rt, T>::from_value(rt, value).as_ref(rt) == needle {
            found = true;
            break;
        }
    });
    found
}

#[extern_fn(effect = E, sync = contains_now)]
async fn contains<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<Erased<Rt, T>, E, I, Rt>,
    needle: T,
) -> bool
where
    T: acvus_extern::Monomorphize<(i64, f64, bool, u8, String)> + Stored<Rt> + PartialEq,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut found = false;
    drain!(it, rt, frame, |value| {
        if *Erased::<Rt, T>::from_value(rt, value).as_ref(rt) == needle {
            found = true;
            break;
        }
    });
    found
}

fn next_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    it: &mut Iter<T, E, I, Rt>,
) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.next_now(rt, frame)
}

#[extern_fn(effect = E, sync = next_now)]
async fn next<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    it: &mut Iter<T, E, I, Rt>,
) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.next(rt, frame).await
}

fn find_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.filter(f).next_now(rt, frame)
}

#[extern_fn(effect = E, sync = find_now)]
async fn find<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.filter(f).next(rt, frame).await
}

fn reduce_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
    f: Fn2<T, T, T, E, Rt>,
) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut acc = it.next_now(rt, frame)?;
    drain_now!(it, rt, frame, |value| {
        let item = T::from_value(rt, value);
        acc = f.call_now(rt, frame, (acc, item));
    });
    Some(acc)
}

#[extern_fn(effect = E, sync = reduce_now)]
async fn reduce<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
    f: Fn2<T, T, T, E, Rt>,
) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut acc = it.next(rt, frame).await?;
    drain!(it, rt, frame, |value| {
        let item = T::from_value(rt, value);
        acc = f.call(rt, frame, (acc, item)).await;
    });
    Some(acc)
}

fn fold_now<T, U, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
    init: U,
    f: Fn2<U, T, U, E, Rt>,
) -> U
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    U: TyVar + OneValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut acc = init;
    drain_now!(it, rt, frame, |value| {
        let item = T::from_value(rt, value);
        acc = f.call_now(rt, frame, (acc, item));
    });
    acc
}

#[extern_fn(effect = E, sync = fold_now)]
async fn fold<T, U, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
    init: U,
    f: Fn2<U, T, U, E, Rt>,
) -> U
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    U: TyVar + OneValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut acc = init;
    drain!(it, rt, frame, |value| {
        let item = T::from_value(rt, value);
        acc = f.call(rt, frame, (acc, item)).await;
    });
    acc
}

fn any_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
) -> bool
where
    T: TyVar + OneValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut found = false;
    drain_now!(it, rt, frame, |value| {
        if f.call_now(rt, frame, (Ref::lend(rt, &value),)) {
            found = true;
            break;
        }
    });
    found
}

#[extern_fn(effect = E, sync = any_now)]
async fn any<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
) -> bool
where
    T: TyVar + OneValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut found = false;
    drain!(it, rt, frame, |value| {
        if f.call(rt, frame, (Ref::lend(rt, &value),)).await {
            found = true;
            break;
        }
    });
    found
}

fn all_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
) -> bool
where
    T: TyVar + OneValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut holds_throughout = true;
    drain_now!(it, rt, frame, |value| {
        if !f.call_now(rt, frame, (Ref::lend(rt, &value),)) {
            holds_throughout = false;
            break;
        }
    });
    holds_throughout
}

#[extern_fn(effect = E, sync = all_now)]
async fn all<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
) -> bool
where
    T: TyVar + OneValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut holds_throughout = true;
    drain!(it, rt, frame, |value| {
        if !f.call(rt, frame, (Ref::lend(rt, &value),)).await {
            holds_throughout = false;
            break;
        }
    });
    holds_throughout
}

/// `start..end`: empty when `end <= start`.
#[extern_fn(effect = pure)]
fn range<E, I, Rt>(start: i64, end: i64) -> Iter<i64, E, I, Rt>
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut current = start;
    Iter::generate(move |_| {
        if current >= end {
            return None;
        }
        let item = current;
        current += 1;
        Some(item)
    })
}

/// `start`, `start + step`, … while short of `end`: upward for a positive
/// `step`, downward for a negative one. A zero `step` traps.
#[extern_fn(effect = pure)]
fn range_step<E, I, Rt>(start: i64, end: i64, step: i64) -> Iter<i64, E, I, Rt>
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    assert!(step != 0, "range_step: step is zero");
    let mut current = start;
    Iter::generate(move |_| {
        let short_of_end = if step > 0 {
            current < end
        } else {
            current > end
        };
        if !short_of_end {
            return None;
        }
        let item = current;
        current = current.checked_add(step)?;
        Some(item)
    })
}

#[extern_fn(effect = pure)]
fn step_by<T, E, I, Rt>(it: Iter<T, E, I, Rt>, n: u64) -> Iter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    assert!(n != 0, "step_by: step is zero");
    it.step_by(n)
}

#[extern_fn(effect = pure)]
fn take_while<T, E, I, Rt>(
    it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
) -> Iter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.take_while(f)
}

#[extern_fn(effect = pure)]
fn skip_while<T, E, I, Rt>(
    it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
) -> Iter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.skip_while(f)
}

#[extern_fn(effect = pure)]
fn chunks<T, E, I, Rt>(it: Iter<T, E, I, Rt>, n: u64) -> Iter<Vec<T>, E, I, Rt>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    assert!(n != 0, "chunks: chunk size is zero");
    it.chunks(n)
}

#[extern_fn(effect = pure)]
fn dedup<T, E, I, Rt>(it: Iter<Erased<Rt, T>, E, I, Rt>) -> Iter<Erased<Rt, T>, E, I, Rt>
where
    T: Monomorphize<(i64, f64, bool, String)> + Stored<Rt> + PartialEq + Clone,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.dedup()
}

fn count_now<T, E, I, Rt>(rt: &Rt, frame: &mut Rt::Frame<'_>, mut it: Iter<T, E, I, Rt>) -> i64
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut n = 0;
    drain_now!(it, rt, frame, |_value| {
        n += 1;
    });
    n
}

#[extern_fn(effect = E, sync = count_now)]
async fn count<T, E, I, Rt>(rt: &Rt, frame: &mut Rt::Frame<'_>, mut it: Iter<T, E, I, Rt>) -> i64
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut n = 0;
    drain!(it, rt, frame, |_value| {
        n += 1;
    });
    n
}

fn last_now<T, E, I, Rt>(rt: &Rt, frame: &mut Rt::Frame<'_>, mut it: Iter<T, E, I, Rt>) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut last = None;
    drain_now!(it, rt, frame, |value| {
        last = Some(T::from_value(rt, value));
    });
    last
}

#[extern_fn(effect = E, sync = last_now)]
async fn last<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut last = None;
    drain!(it, rt, frame, |value| {
        last = Some(T::from_value(rt, value));
    });
    last
}

fn nth_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    it: Iter<T, E, I, Rt>,
    n: u64,
) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.skip(n).next_now(rt, frame)
}

#[extern_fn(effect = E, sync = nth_now)]
async fn nth<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    it: Iter<T, E, I, Rt>,
    n: u64,
) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    it.skip(n).next(rt, frame).await
}

fn position_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
) -> Option<i64>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut index = 0;
    let mut at = None;
    drain_now!(it, rt, frame, |value| {
        if f.call_now(rt, frame, (Ref::lend(rt, &value),)) {
            at = Some(index);
            break;
        }
        index += 1;
    });
    at
}

#[extern_fn(effect = E, sync = position_now)]
async fn position<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
) -> Option<i64>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut index = 0;
    let mut at = None;
    drain!(it, rt, frame, |value| {
        if f.call(rt, frame, (Ref::lend(rt, &value),)).await {
            at = Some(index);
            break;
        }
        index += 1;
    });
    at
}

fn sum_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<Erased<Rt, T>, E, I, Rt>,
) -> T
where
    T: Monomorphize<(i64, f64)> + Stored<Rt> + Num,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut acc = T::ZERO;
    drain_now!(it, rt, frame, |value| {
        let item = Erased::<Rt, T>::from_value(rt, value);
        acc = acc.add(*item.as_ref(rt));
    });
    acc
}

#[extern_fn(effect = E, sync = sum_now)]
async fn sum<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<Erased<Rt, T>, E, I, Rt>,
) -> T
where
    T: Monomorphize<(i64, f64)> + Stored<Rt> + Num,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut acc = T::ZERO;
    drain!(it, rt, frame, |value| {
        let item = Erased::<Rt, T>::from_value(rt, value);
        acc = acc.add(*item.as_ref(rt));
    });
    acc
}

fn product_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<Erased<Rt, T>, E, I, Rt>,
) -> T
where
    T: Monomorphize<(i64, f64)> + Stored<Rt> + Num,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut acc = T::ONE;
    drain_now!(it, rt, frame, |value| {
        let item = Erased::<Rt, T>::from_value(rt, value);
        acc = acc.mul(*item.as_ref(rt));
    });
    acc
}

#[extern_fn(effect = E, sync = product_now)]
async fn product<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<Erased<Rt, T>, E, I, Rt>,
) -> T
where
    T: Monomorphize<(i64, f64)> + Stored<Rt> + Num,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut acc = T::ONE;
    drain!(it, rt, frame, |value| {
        let item = Erased::<Rt, T>::from_value(rt, value);
        acc = acc.mul(*item.as_ref(rt));
    });
    acc
}

fn min_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<Erased<Rt, T>, E, I, Rt>,
) -> Option<T>
where
    T: Monomorphize<(i64, f64)> + Stored<Rt> + Num,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut best: Option<T> = None;
    drain_now!(it, rt, frame, |value| {
        let current = *Erased::<Rt, T>::from_value(rt, value).as_ref(rt);
        best = Some(match best {
            Some(best) => best.min(current),
            None => current,
        });
    });
    best
}

#[extern_fn(effect = E, sync = min_now)]
async fn min<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<Erased<Rt, T>, E, I, Rt>,
) -> Option<T>
where
    T: Monomorphize<(i64, f64)> + Stored<Rt> + Num,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut best: Option<T> = None;
    drain!(it, rt, frame, |value| {
        let current = *Erased::<Rt, T>::from_value(rt, value).as_ref(rt);
        best = Some(match best {
            Some(best) => best.min(current),
            None => current,
        });
    });
    best
}

fn max_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<Erased<Rt, T>, E, I, Rt>,
) -> Option<T>
where
    T: Monomorphize<(i64, f64)> + Stored<Rt> + Num,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut best: Option<T> = None;
    drain_now!(it, rt, frame, |value| {
        let current = *Erased::<Rt, T>::from_value(rt, value).as_ref(rt);
        best = Some(match best {
            Some(best) => best.max(current),
            None => current,
        });
    });
    best
}

#[extern_fn(effect = E, sync = max_now)]
async fn max<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<Erased<Rt, T>, E, I, Rt>,
) -> Option<T>
where
    T: Monomorphize<(i64, f64)> + Stored<Rt> + Num,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut best: Option<T> = None;
    drain!(it, rt, frame, |value| {
        let current = *Erased::<Rt, T>::from_value(rt, value).as_ref(rt);
        best = Some(match best {
            Some(best) => best.max(current),
            None => current,
        });
    });
    best
}

struct Keyed<V> {
    value: V,
    key: i64,
}

#[derive(Clone, Copy)]
enum Extreme {
    Min,
    Max,
}

impl Extreme {
    fn prefers(self, candidate: i64, best: i64) -> bool {
        match self {
            Extreme::Min => candidate < best,
            Extreme::Max => candidate > best,
        }
    }
}

async fn extreme_by_key<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, i64, E, Rt>,
    extreme: Extreme,
) -> Option<T>
where
    T: TyVar + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut best: Option<Keyed<Rt::Value>> = None;
    drain!(it, rt, frame, |value| {
        let key = f.call(rt, frame, (Ref::lend(rt, &value),)).await;
        let replace = match &best {
            Some(Keyed { key: best_key, .. }) => extreme.prefers(key, *best_key),
            None => true,
        };
        if replace {
            best = Some(Keyed { value, key });
        }
    });
    Some(T::from_value(rt, best?.value))
}

fn min_by_key_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, i64, E, Rt>,
) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    extreme_by_key_now(rt, frame, it, f, Extreme::Min)
}

fn extreme_by_key_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    mut it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, i64, E, Rt>,
    extreme: Extreme,
) -> Option<T>
where
    T: TyVar + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    let mut best: Option<Keyed<Rt::Value>> = None;
    drain_now!(it, rt, frame, |value| {
        let key = f.call_now(rt, frame, (Ref::lend(rt, &value),));
        let replace = match &best {
            Some(Keyed { key: best_key, .. }) => extreme.prefers(key, *best_key),
            None => true,
        };
        if replace {
            best = Some(Keyed { value, key });
        }
    });
    Some(T::from_value(rt, best?.value))
}

#[extern_fn(effect = E, sync = min_by_key_now)]
async fn min_by_key<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, i64, E, Rt>,
) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    extreme_by_key(rt, frame, it, f, Extreme::Min).await
}

fn max_by_key_now<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, i64, E, Rt>,
) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    extreme_by_key_now(rt, frame, it, f, Extreme::Max)
}

#[extern_fn(effect = E, sync = max_by_key_now)]
async fn max_by_key<T, E, I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    it: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, i64, E, Rt>,
) -> Option<T>
where
    T: TyVar + OneValue<Rt> + FromValue<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    extreme_by_key(rt, frame, it, f, Extreme::Max).await
}

pub fn iterator_registry<Rt>() -> Registry<Rt>
where
    Rt: Runtime,
{
    extern_registry! {
        ns: "iter",
        types: [Iter<_, _, _, Rt>],
        signatures: [sig::into_iter, sig::as_iter],
        fns: [
            into_iter_vec, into_iter_array, as_iter_vec, as_iter_array, rev_iter,
            range, range_step,
            map, pmap, filter, take, skip, step_by, take_while, skip_while, chunks, dedup,
            chain, pchain, flatten, flatten_arrays, flat_map,
            collect, join, contains, next, find, reduce, fold, any, all,
            count, last, nth, position, sum, product, min, max, min_by_key, max_by_key,
        ],
    }
}
