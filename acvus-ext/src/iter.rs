//! The iterator stages: one extension type per source and per adaptor, each
//! with its own instance of `iter::next` (RFC-0067, RFC-0068).
//!
//! There is no `Iterator` type. A pipeline's type is the nesting of its
//! stages, `Filter<Map<Items<i64, _>, ..>, ..>`, and a stage reaches the one
//! below it through the `next` instance its constructor required, which it
//! holds beside the stage it names. Per element a stage is one function
//! pointer call.
//!
//! **Task.** A stage holds its inner instance at `Later`, and its own `next`
//! is an `async fn` with a `sync =` twin at the effect variable it shares
//! with the inner signature and its closure. A pipeline with one suspending
//! stage therefore has a suspending effect all the way up, and a site whose
//! effect is `Sync` runs twins all the way down. A source's `next` returns
//! and is a plain `fn` at `pure`.
//!
//! **Identity.** A source names the identity variable of what it came from.
//! An adaptor names none: its source's is inside its first type argument.
//!
//! **Instances.** An `Instance` stands beside the receiver it is a method
//! of, so a stage holds one per pipeline it reads: `Chain` holds two
//! pipelines of two types and the `next` of each. A stage that flattens
//! (`Flatten`, `FlatMap`) buffers the container it last drew and needs no
//! instance for it. An element never leaves the type it was declared at:
//! what an instance returns is a `T`, not a runtime value read back.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::ops::Deref;

use acvus_extern::{Arr, InPlaceElement, PassedByValue};
use acvus_extern::{
    Borrowable, Closure, ClosureFn, Cross, Ctx, ExternType, Instance, Later, Ref, Runtime, Shared,
    Stored, Suspends, TransparentOver, Var, core, extern_fn, kind,
};

/// The shared signatures of the iterator surface.
pub mod sig {
    use acvus_extern::extern_signature;

    extern_signature! {
        ns: "iter",
        effect = E,
        fn next<I, T, E, Rt>(it: &mut I) -> Option<T>
        where
            I: Var<kind::Type>,
            T: Var<kind::Type>,
            E: Var<kind::Effect>,
            Rt: Runtime;
    }

    extern_signature! {
        ns: "iter",
        fn into_iter<C, It, Rt>(items: C) -> It
        where
            C: Var<kind::Type>,
            It: Var<kind::Type>,
            Rt: Runtime;
    }

    extern_signature! {
        ns: "iter",
        fn as_iter<C, It, Rt>(items: &C) -> It
        where
            C: Var<kind::Type>,
            It: Var<kind::Type>,
            Rt: Runtime;
    }
}

pub struct ItemsBody<T> {
    rest: std::vec::IntoIter<T>,
}

/// The owned source: the elements of a container that was consumed.
#[derive(ExternType)]
#[extern_type(name = "Items")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct Items<T, I, Rt>(ItemsBody<T>, PhantomData<(I, Rt)>)
where
    T: Var<kind::Type>,
    I: Var<kind::Identity>,
    Rt: Runtime;

impl<T, I, Rt> Items<T, I, Rt>
where
    T: Var<kind::Type>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    pub fn of(items: Vec<T>) -> Self {
        Items(
            ItemsBody {
                rest: items.into_iter(),
            },
            PhantomData,
        )
    }

    pub(crate) fn step(&mut self) -> Option<T> {
        self.0.rest.next()
    }
}

#[extern_fn(instance_of = sig::next, effect = pure)]
pub(crate) fn next_items<T, I, Rt>(it: &mut Items<T, I, Rt>) -> Option<T>
where
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    it.step()
}

/// The `next` of `Items<#T>`, for a declaration that builds an `Items` at a
/// Rust type of its own (RFC-0068 rule 8).
macro_rules! next_items_of {
    (element: $t:ty, next: $next:ident) => {
        #[::acvus_extern::extern_fn(instance_of = $crate::iter::sig::next, effect = pure)]
        pub(crate) fn $next<I, Rt>(it: &mut $crate::iter::Items<$t, I, Rt>) -> Option<$t>
        where
            I: ::acvus_extern::Var<::acvus_extern::kind::Identity>,
            Rt: ::acvus_extern::Runtime,
        {
            it.step()
        }
    };
}

pub(crate) use next_items_of;

pub struct RefsBody<C, Rt>
where
    C: Var<kind::Type>,
    Rt: Runtime,
{
    pub(crate) items: Ref<C, Shared, Rt>,
    pub(crate) at: usize,
}

/// The borrowed source: references into a container that stays where it is.
/// Each container that can be read by position declares the `next` of its
/// own `Refs`.
#[derive(ExternType)]
#[extern_type(name = "Refs")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct Refs<C, I, Rt>(pub(crate) RefsBody<C, Rt>, PhantomData<I>)
where
    C: Var<kind::Type>,
    I: Var<kind::Identity>,
    Rt: Runtime;

impl<C, I, Rt> Refs<C, I, Rt>
where
    C: Var<kind::Type>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    pub fn of(items: Ref<C, Shared, Rt>) -> Self {
        Refs(RefsBody { items, at: 0 }, PhantomData)
    }

    /// The element `at` reads at this step's position, and the step.
    pub fn step<'a, T>(
        &'a mut self,
        ctx: &Ctx<'_, Rt>,
        at: impl FnOnce(&'a C, usize) -> Option<&'a T>,
    ) -> Option<&'a T>
    where
        C: Borrowable<Rt>,
        T: 'a,
    {
        let index = self.0.at;
        self.0.at += 1;
        self.0.items.with(ctx.rt, |items| at(items, index))
    }
}

#[extern_fn(instance_of = sig::next, effect = pure)]
pub(crate) fn next_refs_vec<'a, T, I, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &'a mut Refs<Vec<T>, I, Rt>,
) -> Option<&'a T>
where
    T: Var<kind::Type> + TransparentOver<Rt> + InPlaceElement<Rt>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    it.step(ctx, |items, index| items.get(index))
}

#[extern_fn(instance_of = sig::next, effect = pure)]
pub(crate) fn next_refs_array<'a, T, N, I, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &'a mut Refs<Arr<T, N>, I, Rt>,
) -> Option<&'a T>
where
    T: Var<kind::Type> + TransparentOver<Rt> + InPlaceElement<Rt>,
    N: Var<kind::Length>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    it.step(ctx, |items, index| items.0.get(index))
}

pub struct RangeBody {
    current: i64,
    end: i64,
    step: i64,
}

/// The counted source: `start`, `start + step`, … while short of `end` —
/// upward for a positive `step`, downward for a negative one, and empty
/// where `start` is already at or past `end`. The walk stops rather than
/// wrapping when the next position would leave `i64`.
#[derive(ExternType)]
#[extern_type(name = "Range")]
#[repr(transparent)]
pub struct Range<I, Rt>(RangeBody, PhantomData<(I, Rt)>)
where
    I: Var<kind::Identity>,
    Rt: Runtime;

impl<I, Rt> Range<I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    /// `step` is not zero: the constructors trap on zero before this is
    /// reached.
    pub fn of(start: i64, end: i64, step: i64) -> Self {
        debug_assert!(step != 0, "a Range steps by a non-zero amount");
        Range(
            RangeBody {
                current: start,
                end,
                step,
            },
            PhantomData,
        )
    }
}

#[extern_fn(instance_of = sig::next, effect = pure)]
pub(crate) fn next_range<I, Rt>(it: &mut Range<I, Rt>) -> Option<i64>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    let current = it.0.current;
    let short_of_end = match it.0.step > 0 {
        true => current < it.0.end,
        false => current > it.0.end,
    };
    if !short_of_end {
        return None;
    }
    it.0.current = current.checked_add(it.0.step)?;
    Some(current)
}

pub struct MapBody<I, T, U, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) inner: I,
    pub(crate) next: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>,
    pub(crate) f: Closure<(T,), U, E, Rt>,
}

#[derive(ExternType)]
#[extern_type(name = "Map")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct Map<I, T, U, E, Rt>(pub(crate) MapBody<I, T, U, E, Rt>)
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

fn next_map_now<I, T, U, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut Map<I, T, U, E, Rt>) -> Option<U>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let x = it.0.next.call(ctx, &mut it.0.inner, ())?;
    Some(it.0.f.call_now(ctx, (x,)))
}

#[extern_fn(instance_of = sig::next, effect = E, sync = next_map_now)]
pub(crate) async fn next_map<I, T, U, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Map<I, T, U, E, Rt>,
) -> Option<U>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let x = it.0.next.call_await(ctx, &mut it.0.inner, ()).await?;
    Some(it.0.f.call(ctx, (x,)).await)
}

pub enum UnorderedDraw<U> {
    Undrawn,
    Drawn(VecDeque<U>),
}

pub struct UnorderedBody<I, T, U, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) inner: I,
    pub(crate) next: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>,
    pub(crate) f: Closure<(T,), U, E, Rt>,
    pub(crate) draw: UnorderedDraw<U>,
}

/// A `map` whose calls are joined (RFC-0075 rule 2). The script author
/// states that the order of the map's calls, and of the draws below it, is
/// irrelevant; the closure is not asked to commute. The first pull draws the
/// whole input in order, calls the closure on every element with the calls
/// joined, and keeps the results; each pull hands out the next one in input
/// order.
#[derive(ExternType)]
#[extern_type(name = "Unordered")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct Unordered<I, T, U, E, Rt>(pub(crate) UnorderedBody<I, T, U, E, Rt>)
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

#[extern_fn(instance_of = sig::next, effect = E)]
pub(crate) async fn next_unordered<I, T, U, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Unordered<I, T, U, E, Rt>,
) -> Option<U>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect> + Suspends,
    Rt: Runtime,
{
    match &mut it.0.draw {
        UnorderedDraw::Drawn(results) => results.pop_front(),
        UnorderedDraw::Undrawn => {
            let mut results = draw_unordered(ctx, &mut it.0).await;
            let first = results.pop_front();
            it.0.draw = UnorderedDraw::Drawn(results);
            first
        }
    }
}

/// Draws the whole input in order, then calls the closure on every element
/// with the calls joined, one rooted frame per call (RFC-0075 rule 2).
async fn draw_unordered<I, T, U, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    body: &mut UnorderedBody<I, T, U, E, Rt>,
) -> VecDeque<U>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect> + Suspends,
    Rt: Runtime,
{
    let mut xs = Vec::new();
    while let Some(x) = body.next.call_await(ctx, &mut body.inner, ()).await {
        xs.push(x);
    }
    let rt = ctx.rt;
    let f = &body.f;
    let mut rooted: Vec<Rt::Rooted<'_>> = xs.iter().map(|_| rt.rooted()).collect();
    let calls = rooted
        .iter_mut()
        .zip(xs)
        .map(|(frame, x)| f.call(Rt::ctx_of(frame), (x,)));
    futures::future::join_all(calls).await.into()
}

pub struct FilterBody<I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) inner: I,
    pub(crate) next: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>,
    pub(crate) f: Closure<(Ref<T, Shared, Rt>,), bool, E, Rt>,
}

#[derive(ExternType)]
#[extern_type(name = "Filter")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct Filter<I, T, E, Rt>(pub(crate) FilterBody<I, T, E, Rt>)
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

fn next_filter_now<I, T, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut Filter<I, T, E, Rt>) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    loop {
        let x = it.0.next.call(ctx, &mut it.0.inner, ())?;
        if it.0.f.call_now(ctx, (&x,)) {
            return Some(x);
        }
    }
}

#[extern_fn(instance_of = sig::next, effect = E, sync = next_filter_now)]
pub(crate) async fn next_filter<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Filter<I, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    loop {
        let x = it.0.next.call_await(ctx, &mut it.0.inner, ()).await?;
        if it.0.f.call(ctx, (&x,)).await {
            return Some(x);
        }
    }
}

pub struct TakeBody<I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) inner: I,
    pub(crate) next: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>,
    pub(crate) remaining: u64,
}

/// The first `n` elements, and fewer when the source ends first.
#[derive(ExternType)]
#[extern_type(name = "Take")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct Take<I, T, E, Rt>(pub(crate) TakeBody<I, T, E, Rt>)
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

fn next_take_now<I, T, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut Take<I, T, E, Rt>) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    it.0.remaining = it.0.remaining.checked_sub(1)?;
    it.0.next.call(ctx, &mut it.0.inner, ())
}

#[extern_fn(instance_of = sig::next, effect = E, sync = next_take_now)]
pub(crate) async fn next_take<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Take<I, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    it.0.remaining = it.0.remaining.checked_sub(1)?;
    it.0.next.call_await(ctx, &mut it.0.inner, ()).await
}

pub struct SkipBody<I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) inner: I,
    pub(crate) next: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>,
    pub(crate) remaining: u64,
}

/// Everything after the first `n` elements; the elements skipped are drawn
/// at the first step, not at construction.
#[derive(ExternType)]
#[extern_type(name = "Skip")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct Skip<I, T, E, Rt>(pub(crate) SkipBody<I, T, E, Rt>)
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

fn next_skip_now<I, T, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut Skip<I, T, E, Rt>) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    while it.0.remaining > 0 {
        it.0.remaining -= 1;
        it.0.next.call(ctx, &mut it.0.inner, ())?;
    }
    it.0.next.call(ctx, &mut it.0.inner, ())
}

#[extern_fn(instance_of = sig::next, effect = E, sync = next_skip_now)]
pub(crate) async fn next_skip<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Skip<I, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    while it.0.remaining > 0 {
        it.0.remaining -= 1;
        it.0.next.call_await(ctx, &mut it.0.inner, ()).await?;
    }
    it.0.next.call_await(ctx, &mut it.0.inner, ()).await
}

pub struct StepByBody<I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) inner: I,
    pub(crate) next: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>,
    pub(crate) step: u64,
    pub(crate) started: bool,
}

/// Every `step`th element, the first included. `step` is at least one: the
/// constructor traps on zero before this is reached.
#[derive(ExternType)]
#[extern_type(name = "StepBy")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct StepBy<I, T, E, Rt>(pub(crate) StepByBody<I, T, E, Rt>)
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

fn next_step_by_now<I, T, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut StepBy<I, T, E, Rt>) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    if it.0.started {
        for _ in 1..it.0.step {
            it.0.next.call(ctx, &mut it.0.inner, ())?;
        }
    }
    it.0.started = true;
    it.0.next.call(ctx, &mut it.0.inner, ())
}

#[extern_fn(instance_of = sig::next, effect = E, sync = next_step_by_now)]
pub(crate) async fn next_step_by<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut StepBy<I, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    if it.0.started {
        for _ in 1..it.0.step {
            it.0.next.call_await(ctx, &mut it.0.inner, ()).await?;
        }
    }
    it.0.started = true;
    it.0.next.call_await(ctx, &mut it.0.inner, ()).await
}

pub struct TakeWhileBody<I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) inner: I,
    pub(crate) next: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>,
    pub(crate) f: Closure<(Ref<T, Shared, Rt>,), bool, E, Rt>,
    pub(crate) done: bool,
}

/// The elements up to the first the predicate refuses; that element is drawn
/// and dropped, and the stage answers `None` from then on without drawing
/// again.
#[derive(ExternType)]
#[extern_type(name = "TakeWhile")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct TakeWhile<I, T, E, Rt>(pub(crate) TakeWhileBody<I, T, E, Rt>)
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

fn next_take_while_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut TakeWhile<I, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    if it.0.done {
        return None;
    }
    let x = it.0.next.call(ctx, &mut it.0.inner, ())?;
    if it.0.f.call_now(ctx, (&x,)) {
        return Some(x);
    }
    it.0.done = true;
    None
}

#[extern_fn(instance_of = sig::next, effect = E, sync = next_take_while_now)]
pub(crate) async fn next_take_while<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut TakeWhile<I, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    if it.0.done {
        return None;
    }
    let x = it.0.next.call_await(ctx, &mut it.0.inner, ()).await?;
    if it.0.f.call(ctx, (&x,)).await {
        return Some(x);
    }
    it.0.done = true;
    None
}

pub struct SkipWhileBody<I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) inner: I,
    pub(crate) next: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>,
    pub(crate) f: Closure<(Ref<T, Shared, Rt>,), bool, E, Rt>,
    pub(crate) skipping: bool,
}

/// Everything from the first element the predicate refuses, that element
/// included; the predicate is not called again after it.
#[derive(ExternType)]
#[extern_type(name = "SkipWhile")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct SkipWhile<I, T, E, Rt>(pub(crate) SkipWhileBody<I, T, E, Rt>)
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

fn next_skip_while_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut SkipWhile<I, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    while it.0.skipping {
        let x = it.0.next.call(ctx, &mut it.0.inner, ())?;
        if !it.0.f.call_now(ctx, (&x,)) {
            it.0.skipping = false;
            return Some(x);
        }
    }
    it.0.next.call(ctx, &mut it.0.inner, ())
}

#[extern_fn(instance_of = sig::next, effect = E, sync = next_skip_while_now)]
pub(crate) async fn next_skip_while<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut SkipWhile<I, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    while it.0.skipping {
        let x = it.0.next.call_await(ctx, &mut it.0.inner, ()).await?;
        if !it.0.f.call(ctx, (&x,)).await {
            it.0.skipping = false;
            return Some(x);
        }
    }
    it.0.next.call_await(ctx, &mut it.0.inner, ()).await
}

pub struct ChunksBody<I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) inner: I,
    pub(crate) next: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>,
    pub(crate) size: u64,
}

/// Consecutive elements in `Vec`s of `size`, the last one shorter when the
/// source runs out. `size` is at least one: the constructor traps on zero
/// before this is reached. One chunk is the only buffer.
#[derive(ExternType)]
#[extern_type(name = "Chunks")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct Chunks<I, T, E, Rt>(pub(crate) ChunksBody<I, T, E, Rt>)
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

fn next_chunks_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Chunks<I, T, E, Rt>,
) -> Option<Vec<T>>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut chunk: Vec<T> = Vec::new();
    while (chunk.len() as u64) < it.0.size {
        let Some(x) = it.0.next.call(ctx, &mut it.0.inner, ()) else {
            break;
        };
        chunk.push(x);
    }
    (!chunk.is_empty()).then_some(chunk)
}

#[extern_fn(instance_of = sig::next, effect = E, sync = next_chunks_now)]
pub(crate) async fn next_chunks<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Chunks<I, T, E, Rt>,
) -> Option<Vec<T>>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut chunk: Vec<T> = Vec::new();
    while (chunk.len() as u64) < it.0.size {
        let Some(x) = it.0.next.call_await(ctx, &mut it.0.inner, ()).await else {
            break;
        };
        chunk.push(x);
    }
    (!chunk.is_empty()).then_some(chunk)
}

/// The element `Dedup` has drawn and not yet yielded. The stage yields the
/// element it holds when it meets the next one that differs, one draw
/// behind its source, because it keeps the element itself and requires no
/// `core::clone` to keep a copy of it.
pub(crate) enum Held<T> {
    NothingDrawn,
    Drawn(T),
    SourceSpent,
}

enum Step<T> {
    DrawAgain,
    Yield(T),
    End,
}

pub struct DedupBody<I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) inner: I,
    pub(crate) next: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>,
    pub(crate) eq: Instance<core::eq<T, Rt>, T, Rt>,
    pub(crate) held: Held<T>,
}

impl<I, T, E, Rt> DedupBody<I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + PassedByValue<Rt> + Borrowable<Rt> + Deref<Target = Rt::Value>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    fn has_source(&self) -> bool {
        !matches!(self.held, Held::SourceSpent)
    }

    fn absorb(&mut self, ctx: &mut Ctx<'_, Rt>, drawn: Option<T>) -> Step<T> {
        match (std::mem::replace(&mut self.held, Held::SourceSpent), drawn) {
            (Held::SourceSpent, _) | (Held::NothingDrawn, None) => Step::End,
            (Held::NothingDrawn, Some(item)) => {
                self.held = Held::Drawn(item);
                Step::DrawAgain
            }
            (Held::Drawn(last), None) => Step::Yield(last),
            (Held::Drawn(last), Some(item)) => {
                if self.eq.call(ctx, &last, (&*item,)) {
                    self.held = Held::Drawn(last);
                    Step::DrawAgain
                } else {
                    self.held = Held::Drawn(item);
                    Step::Yield(last)
                }
            }
        }
    }
}

/// Consecutive equal elements collapsed to the first, the element's own
/// `core::eq` deciding which are equal.
#[derive(ExternType)]
#[extern_type(name = "Dedup")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct Dedup<I, T, E, Rt>(pub(crate) DedupBody<I, T, E, Rt>)
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

impl<I, T, E, Rt> Dedup<I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) fn drawing(
        inner: I,
        next: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>,
        eq: Instance<core::eq<T, Rt>, T, Rt>,
    ) -> Self {
        Dedup(DedupBody {
            inner,
            next,
            eq,
            held: Held::NothingDrawn,
        })
    }
}

fn next_dedup_now<I, T, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut Dedup<I, T, E, Rt>) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + PassedByValue<Rt> + Borrowable<Rt> + Deref<Target = Rt::Value>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    while it.0.has_source() {
        let drawn = it.0.next.call(ctx, &mut it.0.inner, ());
        match it.0.absorb(ctx, drawn) {
            Step::DrawAgain => {}
            Step::Yield(item) => return Some(item),
            Step::End => return None,
        }
    }
    None
}

#[extern_fn(instance_of = sig::next, effect = E, sync = next_dedup_now)]
pub(crate) async fn next_dedup<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Dedup<I, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + PassedByValue<Rt> + Borrowable<Rt> + Deref<Target = Rt::Value>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    while it.0.has_source() {
        let drawn = it.0.next.call_await(ctx, &mut it.0.inner, ()).await;
        match it.0.absorb(ctx, drawn) {
            Step::DrawAgain => {}
            Step::Yield(item) => return Some(item),
            Step::End => return None,
        }
    }
    None
}

pub struct ChainBody<A, B, T, E, Rt>
where
    A: Var<kind::Type>,
    B: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) first: A,
    pub(crate) next_first: Instance<sig::next<A, T, E, Rt>, A, Rt, Later>,
    pub(crate) second: B,
    pub(crate) next_second: Instance<sig::next<B, T, E, Rt>, B, Rt, Later>,
    pub(crate) on_first: bool,
}

/// The first pipeline, then the second. The two may be of different types;
/// each stands beside its own `next`.
#[derive(ExternType)]
#[extern_type(name = "Chain")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct Chain<A, B, T, E, Rt>(pub(crate) ChainBody<A, B, T, E, Rt>)
where
    A: Var<kind::Type>,
    B: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

fn next_chain_now<A, B, T, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut Chain<A, B, T, E, Rt>) -> Option<T>
where
    A: Var<kind::Type> + Deref<Target = Rt::Value>,
    B: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    if it.0.on_first {
        if let Some(x) = it.0.next_first.call(ctx, &mut it.0.first, ()) {
            return Some(x);
        }
        it.0.on_first = false;
    }
    it.0.next_second.call(ctx, &mut it.0.second, ())
}

#[extern_fn(instance_of = sig::next, effect = E, sync = next_chain_now)]
pub(crate) async fn next_chain<A, B, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Chain<A, B, T, E, Rt>,
) -> Option<T>
where
    A: Var<kind::Type> + Deref<Target = Rt::Value>,
    B: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    if it.0.on_first {
        if let Some(x) = it.0.next_first.call_await(ctx, &mut it.0.first, ()).await {
            return Some(x);
        }
        it.0.on_first = false;
    }
    it.0.next_second.call_await(ctx, &mut it.0.second, ()).await
}

pub struct FlattenBody<I, C, T, E, Rt>
where
    I: Var<kind::Type>,
    C: Var<kind::Type> + Cross<Rt> + PassedByValue<Rt>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) inner: I,
    pub(crate) next: Instance<sig::next<I, C, E, Rt>, I, Rt, Later>,
    pub(crate) pending: std::vec::IntoIter<T>,
}

/// The elements of each container the source yields, in order. `C` is the
/// container the source's elements stand at — a `Vec` or an `Arr` — and the
/// one it last drew is the only buffer.
#[derive(ExternType)]
#[extern_type(name = "Flatten")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct Flatten<I, C, T, E, Rt>(pub(crate) FlattenBody<I, C, T, E, Rt>)
where
    I: Var<kind::Type>,
    C: Var<kind::Type> + Cross<Rt> + PassedByValue<Rt>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

fn next_flatten_now<I, C, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Flatten<I, C, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    C: Var<kind::Type>
        + Cross<Rt>
        + PassedByValue<Rt>
        + IntoIterator<Item = T, IntoIter = std::vec::IntoIter<T>>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    loop {
        if let Some(x) = it.0.pending.next() {
            return Some(x);
        }
        let batch = it.0.next.call(ctx, &mut it.0.inner, ())?;
        it.0.pending = batch.into_iter();
    }
}

async fn next_flatten_at<I, C, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Flatten<I, C, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    C: Var<kind::Type>
        + Cross<Rt>
        + PassedByValue<Rt>
        + IntoIterator<Item = T, IntoIter = std::vec::IntoIter<T>>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    loop {
        if let Some(x) = it.0.pending.next() {
            return Some(x);
        }
        let batch = it.0.next.call_await(ctx, &mut it.0.inner, ()).await?;
        it.0.pending = batch.into_iter();
    }
}

fn next_flatten_vecs_now<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Flatten<I, Vec<T>, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    next_flatten_now(ctx, it)
}

#[extern_fn(instance_of = sig::next, effect = E, sync = next_flatten_vecs_now)]
pub(crate) async fn next_flatten_vecs<I, T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Flatten<I, Vec<T>, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    next_flatten_at(ctx, it).await
}

fn next_flatten_arrays_now<I, T, N, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Flatten<I, Arr<T, N>, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    N: Var<kind::Length>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    next_flatten_now(ctx, it)
}

#[extern_fn(instance_of = sig::next, effect = E, sync = next_flatten_arrays_now)]
pub(crate) async fn next_flatten_arrays<I, T, N, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut Flatten<I, Arr<T, N>, T, E, Rt>,
) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    N: Var<kind::Length>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    next_flatten_at(ctx, it).await
}

pub struct FlatMapBody<I, T, U, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub(crate) inner: I,
    pub(crate) next: Instance<sig::next<I, T, E, Rt>, I, Rt, Later>,
    pub(crate) f: Closure<(T,), Vec<U>, E, Rt>,
    pub(crate) pending: std::vec::IntoIter<U>,
}

/// The elements of each `Vec` the closure returns, in order. This is `map`
/// and `flatten` in one stage: two stages would need two instances, and the
/// intermediate one has no name to require an instance at.
#[derive(ExternType)]
#[extern_type(name = "FlatMap")]
#[extern_type(unsafe(uniform_payload))]
#[repr(transparent)]
pub struct FlatMap<I, T, U, E, Rt>(pub(crate) FlatMapBody<I, T, U, E, Rt>)
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

fn next_flat_map_now<I, T, U, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut FlatMap<I, T, U, E, Rt>,
) -> Option<U>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    loop {
        if let Some(y) = it.0.pending.next() {
            return Some(y);
        }
        let x = it.0.next.call(ctx, &mut it.0.inner, ())?;
        it.0.pending = it.0.f.call_now(ctx, (x,)).into_iter();
    }
}

#[extern_fn(instance_of = sig::next, effect = E, sync = next_flat_map_now)]
pub(crate) async fn next_flat_map<I, T, U, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &mut FlatMap<I, T, U, E, Rt>,
) -> Option<U>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    loop {
        if let Some(y) = it.0.pending.next() {
            return Some(y);
        }
        let x = it.0.next.call_await(ctx, &mut it.0.inner, ()).await?;
        it.0.pending = it.0.f.call(ctx, (x,)).await.into_iter();
    }
}
