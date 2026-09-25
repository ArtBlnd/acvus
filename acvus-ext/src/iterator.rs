//! Iterator functions: the constructors of the stages in `iter`, and the
//! consumers, each of which requires the `iter::next` of the pipeline it is
//! handed and drives it (RFC-0067, RFC-0068).
//!
//! A consumer is an `async fn` at the pipeline's effect with a `sync =` twin:
//! a site whose effect is `Sync` runs the twin, which calls the instance and
//! returns, and any other site awaits it.
//!
//! - Sources: the shared signatures `iter::into_iter` (consuming) and
//!   `iter::as_iter` (over a borrowed container, yielding references), with
//!   instances for `Vec` and `Array` here and for `Deque`, `Map` and `Set`
//!   in their own modules; `rev_iter`, `range`, `range_step`.
//! - Adaptors: `map`, `unordered`, `filter`, `take`, `skip`, `step_by`,
//!   `take_while`, `skip_while`, `chunks`, `dedup`, `chain`, `flatten`,
//!   `flatten_arrays`, `flat_map`.
//!
//! There is no `pchain`. Its parameter is a `Vec` of pipelines, and a
//! declaration requiring `next` at `I` takes one pipeline as the `Instance`
//! that owns it, which a `Vec<I>` is not (RFC-0067 rule 1).
//! - Consumers: `collect`, `join`, `contains`, `find`, `reduce`,
//!   `fold`, `any`, `all`, `count`, `last`, `nth`, `position`, `min_by_key`,
//!   `max_by_key` (the key is `i64`), and the numeric aggregates `sum`,
//!   `product`, `min`, `max`. One step is the signature itself: `iter::next`
//!   called on a pipeline runs that pipeline's instance.
//!
//! A typed per-element operation stands at `sig::next<I, Erased<Rt, T>, E,
//! Rt>` with `T` bounded by `Monomorphize`: the element is read in place
//! through `Erased::as_ref` (RFC-0041).
//!
//! Not here, each an `acvus-extern` contract: `enumerate`, `zip`,
//! `partition` yield a tuple, and no tuple implements `Cross` (a tuple has
//! `TyArg` only); `repeat` needs a clone of a runtime value, which `Runtime`
//! does not offer; `min_by_key`/`max_by_key` take an `i64` key and not a
//! `Monomorphize<(i64, f64)>` member, because a member fn's glue crosses
//! every parameter naming the member at its specialized representation,
//! which `Closure` does not have.

use std::ops::Deref;

use acvus_extern::PassedByValue;
use acvus_extern::{
    Arr, Closure, ClosureFn, Cross, Ctx, Erased, Monomorphize, Ref, Registry, Runtime, Shared,
    Stored, Suspends, TransparentOver, Var, core, extern_fn, extern_registry, kind,
};
use acvus_extern::{Instance, InstanceOf, Later};

use crate::iter::*;

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

#[extern_fn(instance_of = sig::into_iter, effect = pure)]
fn into_iter_vec<T, I, Rt>(items: Vec<T>) -> Items<T, I, Rt>
where
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(items)
}

#[extern_fn(instance_of = sig::into_iter, effect = pure)]
fn into_iter_array<T, N, I, Rt>(items: Arr<T, N>) -> Items<T, I, Rt>
where
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    N: Var<kind::Length>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(items.0)
}

#[extern_fn(instance_of = sig::as_iter, effect = pure)]
fn as_iter_vec<T, I, Rt>(items: Ref<'_, Vec<T>, Shared, Rt>) -> Refs<'_, Vec<T>, I, Rt>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Refs::of(items)
}

#[extern_fn(instance_of = sig::as_iter, effect = pure)]
fn as_iter_array<T, N, I, Rt>(items: Ref<'_, Arr<T, N>, Shared, Rt>) -> Refs<'_, Arr<T, N>, I, Rt>
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    N: Var<kind::Length>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Refs::of(items)
}

#[extern_fn(effect = pure)]
fn rev_iter<T, I, Rt>(items: Vec<T>) -> Items<T, I, Rt>
where
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    let mut items = items;
    items.reverse();
    Items::of(items)
}

/// `start..end`: empty when `end <= start`.
#[extern_fn(effect = pure)]
fn range<I, Rt>(start: i64, end: i64) -> Range<I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Range::of(start, end, 1)
}

/// `start`, `start + step`, … while short of `end`: upward for a positive
/// `step`, downward for a negative one. A zero `step` traps.
#[extern_fn(effect = pure)]
fn range_step<I, Rt>(start: i64, end: i64, step: i64) -> Range<I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    assert!(step != 0, "range_step: step is zero");
    Range::of(start, end, step)
}

#[extern_fn(effect = pure)]
fn map<'a, I, T, U, E, Rt>(
    it: Instance<'a, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'a, (T,), U, E, Rt>,
) -> Map<'a, I, T, U, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    Map(MapBody { inner: it, f })
}

/// The `map` it is given, with the calls joined: the first pull draws the
/// whole input and runs every call at once, and the results come out in
/// input order (RFC-0075 rule 2). `E: Suspends` refuses a pipeline that
/// cannot suspend (RFC-0011 rule 5).
#[extern_fn(effect = pure)]
fn unordered<I, T, U, E, Rt>(it: Map<'_, I, T, U, E, Rt>) -> Unordered<'_, I, T, U, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect> + Suspends,
    Rt: Runtime,
{
    let Map(MapBody { inner, f }) = it;
    Unordered(UnorderedBody {
        inner,
        f,
        draw: UnorderedDraw::Undrawn,
    })
}

#[extern_fn(effect = pure)]
fn filter<'a, I, T, E, Rt>(
    it: Instance<'a, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'a, (Ref<'a, T, Shared, Rt>,), bool, E, Rt>,
) -> Filter<'a, I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    Filter(FilterBody { inner: it, f })
}

#[extern_fn(effect = pure)]
fn take<'a, I, T, E, Rt>(
    it: Instance<'a, sig::next<I, T, E, Rt>, I, Rt, Later>,
    n: u64,
) -> Take<'a, I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    Take(TakeBody {
        inner: it,
        remaining: n,
    })
}

#[extern_fn(effect = pure)]
fn skip<'a, I, T, E, Rt>(
    it: Instance<'a, sig::next<I, T, E, Rt>, I, Rt, Later>,
    n: u64,
) -> Skip<'a, I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    Skip(SkipBody {
        inner: it,
        remaining: n,
    })
}

#[extern_fn(effect = pure)]
fn step_by<'a, I, T, E, Rt>(
    it: Instance<'a, sig::next<I, T, E, Rt>, I, Rt, Later>,
    n: u64,
) -> StepBy<'a, I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    assert!(n != 0, "step_by: step is zero");
    StepBy(StepByBody {
        inner: it,
        step: n,
        started: false,
    })
}

#[extern_fn(effect = pure)]
fn take_while<'a, I, T, E, Rt>(
    it: Instance<'a, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'a, (Ref<'a, T, Shared, Rt>,), bool, E, Rt>,
) -> TakeWhile<'a, I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    TakeWhile(TakeWhileBody {
        inner: it,
        f,
        done: false,
    })
}

#[extern_fn(effect = pure)]
fn skip_while<'a, I, T, E, Rt>(
    it: Instance<'a, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'a, (Ref<'a, T, Shared, Rt>,), bool, E, Rt>,
) -> SkipWhile<'a, I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    SkipWhile(SkipWhileBody {
        inner: it,
        f,
        skipping: true,
    })
}

#[extern_fn(effect = pure)]
fn chunks<'a, I, T, E, Rt>(
    it: Instance<'a, sig::next<I, T, E, Rt>, I, Rt, Later>,
    n: u64,
) -> Chunks<'a, I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    assert!(n != 0, "chunks: chunk size is zero");
    Chunks(ChunksBody {
        inner: it,
        size: n,
    })
}

#[extern_fn(effect = pure)]
fn dedup<'a, I, T, E, Rt>(
    it: Instance<'a, sig::next<I, T, E, Rt>, I, Rt, Later>,
    eq: InstanceOf<'a, core::eq<T, Rt>, T, Rt>,
) -> Dedup<'a, I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    Dedup::drawing(it, eq)
}

#[extern_fn(effect = pure)]
fn chain<'a, A, B, T, E, Rt>(
    a: Instance<'a, sig::next<A, T, E, Rt>, A, Rt, Later>,
    b: Instance<'a, sig::next<B, T, E, Rt>, B, Rt, Later>,
) -> Chain<'a, A, B, T, E, Rt>
where
    A: Var<kind::Type>,
    B: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    Chain(ChainBody {
        first: a,
        second: b,
        on_first: true,
    })
}

#[extern_fn(effect = pure)]
fn flatten<'a, I, T, E, Rt>(
    it: Instance<'a, sig::next<I, Vec<T>, E, Rt>, I, Rt, Later>,
) -> Flatten<'a, I, Vec<T>, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    Flatten(FlattenBody {
        inner: it,
        pending: Vec::new().into_iter(),
    })
}

#[extern_fn(effect = pure)]
fn flatten_arrays<'a, I, T, N, E, Rt>(
    it: Instance<'a, sig::next<I, Arr<T, N>, E, Rt>, I, Rt, Later>,
) -> Flatten<'a, I, Arr<T, N>, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    N: Var<kind::Length>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    Flatten(FlattenBody {
        inner: it,
        pending: Vec::new().into_iter(),
    })
}

#[extern_fn(effect = pure)]
fn flat_map<'a, I, T, U, E, Rt>(
    it: Instance<'a, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'a, (T,), Vec<U>, E, Rt>,
) -> FlatMap<'a, I, T, U, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    FlatMap(FlatMapBody {
        inner: it,
        f,
        pending: Vec::new().into_iter(),
    })
}

fn collect_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
) -> Vec<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut items = Vec::new();
    while let Some(x) = it.call(ctx, ()) {
        items.push(x);
    }
    items
}

#[extern_fn(effect = E, sync = collect_now)]
async fn collect<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
) -> Vec<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut items = Vec::new();
    while let Some(x) = it.call_await(ctx, ()).await {
        items.push(x);
    }
    items
}

fn join_now<I, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, Erased<Rt, String>, E, Rt>, I, Rt, Later>,
    sep: String,
) -> String
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut it = it;
    let mut parts: Vec<String> = Vec::new();
    while let Some(part) = it.call(ctx, ()) {
        parts.push(part.as_ref(rt).clone());
    }
    parts.join(&sep)
}

#[extern_fn(effect = E, sync = join_now)]
async fn join<I, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, Erased<Rt, String>, E, Rt>, I, Rt, Later>,
    sep: String,
) -> String
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut it = it;
    let mut parts: Vec<String> = Vec::new();
    while let Some(part) = it.call_await(ctx, ()).await {
        parts.push(part.as_ref(rt).clone());
    }
    parts.join(&sep)
}

fn contains_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, Erased<Rt, T>, E, Rt>, I, Rt, Later>,
    needle: T,
) -> bool
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Monomorphize<(i64, f64, bool, u8, String)> + Var<kind::Type> + Stored<Rt> + PartialEq,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut it = it;
    while let Some(item) = it.call(ctx, ()) {
        if *item.as_ref(rt) == needle {
            return true;
        }
    }
    false
}

#[extern_fn(effect = E, sync = contains_now)]
async fn contains<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, Erased<Rt, T>, E, Rt>, I, Rt, Later>,
    needle: T,
) -> bool
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Monomorphize<(i64, f64, bool, u8, String)> + Var<kind::Type> + Stored<Rt> + PartialEq,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut it = it;
    while let Some(item) = it.call_await(ctx, ()).await {
        if *item.as_ref(rt) == needle {
            return true;
        }
    }
    false
}

fn find_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), bool, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    loop {
        let x = it.call(ctx, ())?;
        if f.call_now(ctx, (&x,)) {
            return Some(x);
        }
    }
}

#[extern_fn(effect = E, sync = find_now)]
async fn find<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), bool, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    loop {
        let x = it.call_await(ctx, ()).await?;
        if f.call(ctx, (&x,)).await {
            return Some(x);
        }
    }
}

fn reduce_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (T, T), T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut acc = it.call(ctx, ())?;
    while let Some(x) = it.call(ctx, ()) {
        acc = f.call_now(ctx, (acc, x));
    }
    Some(acc)
}

#[extern_fn(effect = E, sync = reduce_now)]
async fn reduce<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (T, T), T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut acc = it.call_await(ctx, ()).await?;
    while let Some(x) = it.call_await(ctx, ()).await {
        acc = f.call(ctx, (acc, x)).await;
    }
    Some(acc)
}

fn fold_now<I, T, U, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    init: U,
    f: Closure<'_, (U, T), U, E, Rt>,
) -> U
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut acc = init;
    while let Some(x) = it.call(ctx, ()) {
        acc = f.call_now(ctx, (acc, x));
    }
    acc
}

#[extern_fn(effect = E, sync = fold_now)]
async fn fold<I, T, U, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    init: U,
    f: Closure<'_, (U, T), U, E, Rt>,
) -> U
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut acc = init;
    while let Some(x) = it.call_await(ctx, ()).await {
        acc = f.call(ctx, (acc, x)).await;
    }
    acc
}

fn any_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), bool, E, Rt>,
) -> bool
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    while let Some(x) = it.call(ctx, ()) {
        if f.call_now(ctx, (&x,)) {
            return true;
        }
    }
    false
}

#[extern_fn(effect = E, sync = any_now)]
async fn any<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), bool, E, Rt>,
) -> bool
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    while let Some(x) = it.call_await(ctx, ()).await {
        if f.call(ctx, (&x,)).await {
            return true;
        }
    }
    false
}

fn all_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), bool, E, Rt>,
) -> bool
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    while let Some(x) = it.call(ctx, ()) {
        if !f.call_now(ctx, (&x,)) {
            return false;
        }
    }
    true
}

#[extern_fn(effect = E, sync = all_now)]
async fn all<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), bool, E, Rt>,
) -> bool
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    while let Some(x) = it.call_await(ctx, ()).await {
        if !f.call(ctx, (&x,)).await {
            return false;
        }
    }
    true
}

fn count_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
) -> i64
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut n = 0;
    while it.call(ctx, ()).is_some() {
        n += 1;
    }
    n
}

#[extern_fn(effect = E, sync = count_now)]
async fn count<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
) -> i64
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut n = 0;
    while it.call_await(ctx, ()).await.is_some() {
        n += 1;
    }
    n
}

fn last_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut last = None;
    while let Some(x) = it.call(ctx, ()) {
        last = Some(x);
    }
    last
}

#[extern_fn(effect = E, sync = last_now)]
async fn last<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut last = None;
    while let Some(x) = it.call_await(ctx, ()).await {
        last = Some(x);
    }
    last
}

fn nth_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    n: u64,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    for _ in 0..n {
        it.call(ctx, ())?;
    }
    it.call(ctx, ())
}

#[extern_fn(effect = E, sync = nth_now)]
async fn nth<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    n: u64,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    for _ in 0..n {
        it.call_await(ctx, ()).await?;
    }
    it.call_await(ctx, ()).await
}

fn position_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), bool, E, Rt>,
) -> Option<i64>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut index = 0;
    while let Some(x) = it.call(ctx, ()) {
        if f.call_now(ctx, (&x,)) {
            return Some(index);
        }
        index += 1;
    }
    None
}

#[extern_fn(effect = E, sync = position_now)]
async fn position<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), bool, E, Rt>,
) -> Option<i64>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut index = 0;
    while let Some(x) = it.call_await(ctx, ()).await {
        if f.call(ctx, (&x,)).await {
            return Some(index);
        }
        index += 1;
    }
    None
}

fn sum_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, Erased<Rt, T>, E, Rt>, I, Rt, Later>,
) -> T
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Monomorphize<(i64, f64)> + Var<kind::Type> + Stored<Rt> + Num,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut it = it;
    let mut acc = T::ZERO;
    while let Some(item) = it.call(ctx, ()) {
        acc = acc.add(*item.as_ref(rt));
    }
    acc
}

#[extern_fn(effect = E, sync = sum_now)]
async fn sum<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, Erased<Rt, T>, E, Rt>, I, Rt, Later>,
) -> T
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Monomorphize<(i64, f64)> + Var<kind::Type> + Stored<Rt> + Num,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut it = it;
    let mut acc = T::ZERO;
    while let Some(item) = it.call_await(ctx, ()).await {
        acc = acc.add(*item.as_ref(rt));
    }
    acc
}

fn product_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, Erased<Rt, T>, E, Rt>, I, Rt, Later>,
) -> T
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Monomorphize<(i64, f64)> + Var<kind::Type> + Stored<Rt> + Num,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut it = it;
    let mut acc = T::ONE;
    while let Some(item) = it.call(ctx, ()) {
        acc = acc.mul(*item.as_ref(rt));
    }
    acc
}

#[extern_fn(effect = E, sync = product_now)]
async fn product<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, Erased<Rt, T>, E, Rt>, I, Rt, Later>,
) -> T
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Monomorphize<(i64, f64)> + Var<kind::Type> + Stored<Rt> + Num,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut it = it;
    let mut acc = T::ONE;
    while let Some(item) = it.call_await(ctx, ()).await {
        acc = acc.mul(*item.as_ref(rt));
    }
    acc
}

fn min_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, Erased<Rt, T>, E, Rt>, I, Rt, Later>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Monomorphize<(i64, f64)> + Var<kind::Type> + Stored<Rt> + Num,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut it = it;
    let mut best: Option<T> = None;
    while let Some(item) = it.call(ctx, ()) {
        let current = *item.as_ref(rt);
        best = Some(match best {
            Some(best) => best.min(current),
            None => current,
        });
    }
    best
}

#[extern_fn(effect = E, sync = min_now)]
async fn min<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, Erased<Rt, T>, E, Rt>, I, Rt, Later>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Monomorphize<(i64, f64)> + Var<kind::Type> + Stored<Rt> + Num,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut it = it;
    let mut best: Option<T> = None;
    while let Some(item) = it.call_await(ctx, ()).await {
        let current = *item.as_ref(rt);
        best = Some(match best {
            Some(best) => best.min(current),
            None => current,
        });
    }
    best
}

fn max_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, Erased<Rt, T>, E, Rt>, I, Rt, Later>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Monomorphize<(i64, f64)> + Var<kind::Type> + Stored<Rt> + Num,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut it = it;
    let mut best: Option<T> = None;
    while let Some(item) = it.call(ctx, ()) {
        let current = *item.as_ref(rt);
        best = Some(match best {
            Some(best) => best.max(current),
            None => current,
        });
    }
    best
}

#[extern_fn(effect = E, sync = max_now)]
async fn max<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, Erased<Rt, T>, E, Rt>, I, Rt, Later>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Monomorphize<(i64, f64)> + Var<kind::Type> + Stored<Rt> + Num,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut it = it;
    let mut best: Option<T> = None;
    while let Some(item) = it.call_await(ctx, ()).await {
        let current = *item.as_ref(rt);
        best = Some(match best {
            Some(best) => best.max(current),
            None => current,
        });
    }
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

async fn extreme_by_key<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), i64, E, Rt>,
    extreme: Extreme,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut best: Option<Keyed<T>> = None;
    while let Some(value) = it.call_await(ctx, ()).await {
        let key = f.call(ctx, (&value,)).await;
        let replace = match &best {
            Some(Keyed { key: best_key, .. }) => extreme.prefers(key, *best_key),
            None => true,
        };
        if replace {
            best = Some(Keyed { value, key });
        }
    }
    Some(best?.value)
}

fn extreme_by_key_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), i64, E, Rt>,
    extreme: Extreme,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut best: Option<Keyed<T>> = None;
    while let Some(value) = it.call(ctx, ()) {
        let key = f.call_now(ctx, (&value,));
        let replace = match &best {
            Some(Keyed { key: best_key, .. }) => extreme.prefers(key, *best_key),
            None => true,
        };
        if replace {
            best = Some(Keyed { value, key });
        }
    }
    Some(best?.value)
}

fn min_by_key_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), i64, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    extreme_by_key_now(ctx, it, f, Extreme::Min)
}

#[extern_fn(effect = E, sync = min_by_key_now)]
async fn min_by_key<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), i64, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    extreme_by_key(ctx, it, f, Extreme::Min).await
}

fn max_by_key_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), i64, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    extreme_by_key_now(ctx, it, f, Extreme::Max)
}

#[extern_fn(effect = E, sync = max_by_key_now)]
async fn max_by_key<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: Instance<'_, sig::next<I, T, E, Rt>, I, Rt, Later>,
    f: Closure<'_, (Ref<'_, T, Shared, Rt>,), i64, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    extreme_by_key(ctx, it, f, Extreme::Max).await
}

pub fn iterator_registry<Rt>() -> Registry<Rt>
where
    Rt: Runtime,
{
    extern_registry! {
        ns: "iter",
        types: [
            Vec<_>,
            Items<_, _, Rt>, Refs<_, _, Rt>, Range<_, Rt>,
            Map<_, _, _, _, Rt>, Unordered<_, _, _, _, Rt>, Filter<_, _, _, Rt>,
            Take<_, _, _, Rt>, Skip<_, _, _, Rt>, StepBy<_, _, _, Rt>,
            TakeWhile<_, _, _, Rt>, SkipWhile<_, _, _, Rt>,
            Chunks<_, _, _, Rt>, Dedup<_, _, _, Rt>,
            Chain<_, _, _, _, Rt>,
            Flatten<_, _, _, _, Rt>, FlatMap<_, _, _, _, Rt>,
        ],
        signatures: [sig::next, sig::into_iter, sig::as_iter],
        fns: [
            into_iter_vec, next_items, into_iter_array,
            as_iter_vec, next_refs_vec, as_iter_array, next_refs_array,
            rev_iter, range, range_step, next_range,
            map, next_map, unordered, next_unordered, filter, next_filter,
            take, next_take, skip, next_skip, step_by, next_step_by,
            take_while, next_take_while, skip_while, next_skip_while,
            chunks, next_chunks, dedup, next_dedup,
            chain, next_chain,
            flatten, next_flatten_vecs, flatten_arrays, next_flatten_arrays,
            flat_map, next_flat_map,
            collect, join, contains, find, reduce, fold, any, all,
            count, last, nth, position, sum, product, min, max,
            min_by_key, max_by_key,
        ],
    }
}
