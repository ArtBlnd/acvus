//! The `Iterator` extension type: a lazy, move-only pipeline of stages.
//!
//! A stage is built by one constructor that takes its source and its
//! closure at one element type, then boxed as a `dyn` stage, which yields
//! the runtime's values. The box is the payload that crosses the boundary,
//! and it names no element type: the glue erases a `Regex`'s `Iter<String>`
//! and materializes the same box as a consumer's `Iter<T>`, so a payload
//! that mentioned `T` would be two Rust types on the two sides.
//!
//! `None` is the end of the source or a trap. The trap itself is on the
//! runtime, and the extern boundary that called the consumer is what reads
//! it back and fails the run (RFC-0044, stage 6).
//!
//! `T` is the declared element type and nothing more: no value is read as a
//! `T` on its account. The exit is `FromValue`, identity for the runtime's
//! value and a checked downcast for `Erased` and the container boxes, so the
//! pipeline's construction is never cited as a proof.

use std::collections::VecDeque;
use std::marker::PhantomData;

use acvus_extern::{
    BoxFuture, ClosureFn, Erased, ExternType, Fn1, FromValue, OneValue, Ref, Runtime, Stored, Var,
    kind,
};
use sync_wrapper::SyncWrapper;

#[derive(ExternType)]
#[extern_type(name = "Iterator")]
#[repr(transparent)]
pub struct Iter<T, E, I, Rt>(Stages<Rt>, PhantomData<(T, E, I)>)
where
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime;

pub enum Stages<Rt>
where
    Rt: Runtime,
{
    Sync(Box<dyn SyncStage<Rt>>),
    Async(Box<dyn AsyncStage<Rt>>),
}

pub trait SyncStage<Rt>: Send + Sync
where
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value>;
}

pub trait AsyncStage<Rt>: Send + Sync
where
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>>;
}

impl<Rt> SyncStage<Rt> for Box<dyn SyncStage<Rt>>
where
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        (**self).next(rt, frame)
    }
}

impl<Rt> AsyncStage<Rt> for Box<dyn AsyncStage<Rt>>
where
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>> {
        (**self).next(rt, frame)
    }
}

struct Lifted<S>(S);

impl<S, Rt> AsyncStage<Rt> for Lifted<S>
where
    S: SyncStage<Rt>,
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>> {
        Box::pin(std::future::ready(self.0.next(rt, frame)))
    }
}

impl<Rt> Stages<Rt>
where
    Rt: Runtime,
{
    fn into_async(self) -> Box<dyn AsyncStage<Rt>> {
        match self {
            Self::Sync(stage) => Box::new(Lifted(stage)),
            Self::Async(stage) => stage,
        }
    }

    /// The synchronous arm, for a consumer the solver chose the
    /// `Task::Sync` instance of. This is the one place that arm is read,
    /// so the guarantee is stated once (RFC-0046).
    pub fn sync_mut(&mut self) -> &mut Box<dyn SyncStage<Rt>> {
        match self {
            Self::Sync(stage) => stage,
            Self::Async(_) => unreachable!(
                "an Iterator whose effect's task is Sync was built from synchronous stages \
                 only: every constructor that takes an asynchronous source or a suspending \
                 closure raises the task above Sync, and the checker would then have chosen \
                 this consumer's asynchronous instance (RFC-0046)"
            ),
        }
    }
}

/// The body may await: both arms expand inside the consumer's own
/// `async fn`, and only the source's `next` differs between them.
macro_rules! drain {
    ($it:expr, $rt:expr, $frame:expr, |$value:pat_param| $body:block) => {
        match $it.stages_mut() {
            $crate::iter::Stages::Sync(stage) => {
                while let Some($value) = $crate::iter::SyncStage::next(stage, $rt, $frame) $body
            }
            $crate::iter::Stages::Async(stage) => {
                while let Some($value) =
                    $crate::iter::AsyncStage::next(stage, $rt, $frame).await $body
            }
        }
    };
}

/// The body cannot await: the consumer is the `Task::Sync` instance, and
/// `Stages::sync_mut` is where that is checked.
macro_rules! drain_now {
    ($it:expr, $rt:expr, $frame:expr, |$value:pat_param| $body:block) => {{
        let stage = $it.stages_mut().sync_mut();
        while let Some($value) = $crate::iter::SyncStage::next(stage, $rt, $frame) $body
    }};
}

pub(crate) use {drain, drain_now};

impl<T, E, I, Rt> Iter<T, E, I, Rt>
where
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    fn sealed(stage: impl SyncStage<Rt> + 'static) -> Self {
        Self(Stages::Sync(Box::new(stage)), PhantomData)
    }

    fn suspending(stage: impl AsyncStage<Rt> + 'static) -> Self {
        Self(Stages::Async(Box::new(stage)), PhantomData)
    }

    pub fn stages_mut(&mut self) -> &mut Stages<Rt> {
        &mut self.0
    }

    pub fn from_items(items: Vec<T>) -> Self
    where
        T: OneValue<Rt>,
    {
        let mut items = items.into_iter();
        Self::generate(move |_| items.next())
    }

    pub fn generate<F>(f: F) -> Self
    where
        F: FnMut(&Rt) -> Option<T> + Send + 'static,
        T: OneValue<Rt>,
    {
        Self::sealed(Generate(SyncWrapper::new(f)))
    }

    pub fn map<U>(self, f: Fn1<T, U, E, Rt>) -> Iter<U, E, I, Rt>
    where
        U: Var<kind::Type>,
    {
        match self.0 {
            Stages::Sync(source) if f.is_sync() => Iter::sealed(Map { source, f }),
            source => Iter::suspending(Map {
                source: source.into_async(),
                f,
            }),
        }
    }

    pub fn filter(self, f: Fn1<Ref<T, Rt>, bool, E, Rt>) -> Self {
        match self.0 {
            Stages::Sync(source) if f.is_sync() => Self::sealed(Filter { source, f }),
            source => Self::suspending(Filter {
                source: source.into_async(),
                f,
            }),
        }
    }

    pub fn take(self, n: u64) -> Self {
        match self.0 {
            Stages::Sync(source) => Self::sealed(Take {
                source,
                remaining: n,
            }),
            Stages::Async(source) => Self::suspending(Take {
                source,
                remaining: n,
            }),
        }
    }

    pub fn skip(self, n: u64) -> Self {
        match self.0 {
            Stages::Sync(source) => Self::sealed(Skip {
                source,
                remaining: n,
            }),
            Stages::Async(source) => Self::suspending(Skip {
                source,
                remaining: n,
            }),
        }
    }

    /// Every `step`th element, the first included. `step` is at least one:
    /// the ExternFn traps on zero before this is reached.
    pub fn step_by(self, step: u64) -> Self {
        match self.0 {
            Stages::Sync(source) => Self::sealed(StepBy {
                source,
                step,
                started: false,
            }),
            Stages::Async(source) => Self::suspending(StepBy {
                source,
                step,
                started: false,
            }),
        }
    }

    pub fn take_while(self, f: Fn1<Ref<T, Rt>, bool, E, Rt>) -> Self {
        match self.0 {
            Stages::Sync(source) if f.is_sync() => Self::sealed(TakeWhile {
                source,
                f,
                done: false,
            }),
            source => Self::suspending(TakeWhile {
                source: source.into_async(),
                f,
                done: false,
            }),
        }
    }

    pub fn skip_while(self, f: Fn1<Ref<T, Rt>, bool, E, Rt>) -> Self {
        match self.0 {
            Stages::Sync(source) if f.is_sync() => Self::sealed(SkipWhile {
                source,
                f,
                skipping: true,
            }),
            source => Self::suspending(SkipWhile {
                source: source.into_async(),
                f,
                skipping: true,
            }),
        }
    }

    /// Consecutive elements in `Vec`s of `size`, the last one shorter when
    /// the source runs out. `size` is at least one: the ExternFn traps on
    /// zero before this is reached. One chunk is the only buffer.
    pub fn chunks(self, size: u64) -> Iter<Vec<T>, E, I, Rt>
    where
        T: OneValue<Rt> + FromValue<Rt>,
    {
        match self.0 {
            Stages::Sync(source) => Iter::sealed(Chunks::<_, T> {
                source,
                size,
                item: PhantomData,
            }),
            Stages::Async(source) => Iter::suspending(Chunks::<_, T> {
                source,
                size,
                item: PhantomData,
            }),
        }
    }

    pub fn chain<J, K>(self, other: Iter<T, E, J, Rt>) -> Iter<T, E, K, Rt>
    where
        J: Var<kind::Identity>,
        K: Var<kind::Identity>,
    {
        Iter::chained(VecDeque::from([self.0, other.0]))
    }

    pub fn chain_all<K>(parts: Vec<Self>) -> Iter<T, E, K, Rt>
    where
        K: Var<kind::Identity>,
    {
        Iter::chained(parts.into_iter().map(|part| part.0).collect())
    }

    /// A chain suspends where any of its parts does: a later part is
    /// reached only after the ones before it end, but the box that holds
    /// them all is one stage and answers one way.
    fn chained(parts: VecDeque<Stages<Rt>>) -> Self {
        if parts.iter().all(|part| matches!(part, Stages::Sync(_))) {
            let parts = parts
                .into_iter()
                .map(|part| match part {
                    Stages::Sync(stage) => stage,
                    Stages::Async(_) => unreachable!("every part answered Sync"),
                })
                .collect();
            return Self::sealed(Chain { parts });
        }
        let parts = parts.into_iter().map(Stages::into_async).collect();
        Self::suspending(Chain { parts })
    }

    pub fn flatten<U>(self) -> Iter<U, E, I, Rt>
    where
        T: FromValue<Rt> + IntoIterator<Item = U>,
        T::IntoIter: Send + Sync,
        U: Var<kind::Type> + OneValue<Rt>,
    {
        match self.0 {
            Stages::Sync(source) => Iter::sealed(Flatten::<_, T> {
                source,
                pending: None,
            }),
            Stages::Async(source) => Iter::suspending(Flatten::<_, T> {
                source,
                pending: None,
            }),
        }
    }

    pub fn flat_map<S, U>(self, f: Fn1<T, S, E, Rt>) -> Iter<U, E, I, Rt>
    where
        S: Var<kind::Type> + FromValue<Rt> + IntoIterator<Item = U>,
        S::IntoIter: Send + Sync,
        U: Var<kind::Type> + OneValue<Rt>,
    {
        self.map(f).flatten()
    }

    pub async fn next_value(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        match &mut self.0 {
            Stages::Sync(stage) => stage.next(rt, frame),
            Stages::Async(stage) => stage.next(rt, frame).await,
        }
    }

    pub async fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<T>
    where
        T: FromValue<Rt>,
    {
        Some(T::from_value(rt, self.next_value(rt, frame).await?))
    }

    /// As `next`, for a consumer the solver chose the `Task::Sync`
    /// instance of.
    pub fn next_now(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<T>
    where
        T: FromValue<Rt>,
    {
        let value = self.0.sync_mut().next(rt, frame)?;
        Some(T::from_value(rt, value))
    }
}

impl<T, E, I, Rt> Iter<Erased<Rt, T>, E, I, Rt>
where
    T: Var<kind::Type> + Stored<Rt> + PartialEq + Clone,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    /// Consecutive equal elements collapsed to the first. The stage keeps a
    /// `T` clone of the last element it yielded, never a second value.
    pub fn dedup(self) -> Self {
        match self.0 {
            Stages::Sync(source) => Self::sealed(Dedup::<_, T> { source, last: None }),
            Stages::Async(source) => Self::suspending(Dedup::<_, T> { source, last: None }),
        }
    }
}

struct Generate<F>(SyncWrapper<F>);

impl<F, T, Rt> SyncStage<Rt> for Generate<F>
where
    F: FnMut(&Rt) -> Option<T> + Send,
    T: OneValue<Rt>,
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, _frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        Some(self.0.get_mut()(rt)?.erase(rt))
    }
}

struct Map<S, T, U, E, Rt>
where
    T: Var<kind::Type>,
    U: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    source: S,
    f: Fn1<T, U, E, Rt>,
}

impl<S, T, U, E, Rt> SyncStage<Rt> for Map<S, T, U, E, Rt>
where
    S: SyncStage<Rt>,
    T: Var<kind::Type>,
    U: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        let item = self.source.next(rt, frame)?;
        Some(self.f.call_value_now(rt, frame, item))
    }
}

impl<S, T, U, E, Rt> AsyncStage<Rt> for Map<S, T, U, E, Rt>
where
    S: AsyncStage<Rt>,
    T: Var<kind::Type>,
    U: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>> {
        Box::pin(async move {
            let Map { source, f } = self;
            let item = source.next(rt, frame).await?;
            Some(f.call_value(rt, frame, item).await)
        })
    }
}

struct Filter<S, T, E, Rt>
where
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    source: S,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
}

impl<S, T, E, Rt> SyncStage<Rt> for Filter<S, T, E, Rt>
where
    S: SyncStage<Rt>,
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        while let Some(value) = self.source.next(rt, frame) {
            if self.f.call_now(rt, frame, (Ref::lend(rt, &value),)) {
                return Some(value);
            }
        }
        None
    }
}

impl<S, T, E, Rt> AsyncStage<Rt> for Filter<S, T, E, Rt>
where
    S: AsyncStage<Rt>,
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>> {
        Box::pin(async move {
            let Filter { source, f } = self;
            while let Some(value) = source.next(rt, frame).await {
                if f.call(rt, frame, (Ref::lend(rt, &value),)).await {
                    return Some(value);
                }
            }
            None
        })
    }
}

struct Take<S> {
    source: S,
    remaining: u64,
}

impl<S, Rt> SyncStage<Rt> for Take<S>
where
    S: SyncStage<Rt>,
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        self.remaining = self.remaining.checked_sub(1)?;
        self.source.next(rt, frame)
    }
}

impl<S, Rt> AsyncStage<Rt> for Take<S>
where
    S: AsyncStage<Rt>,
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>> {
        Box::pin(async move {
            self.remaining = self.remaining.checked_sub(1)?;
            self.source.next(rt, frame).await
        })
    }
}

struct Skip<S> {
    source: S,
    remaining: u64,
}

impl<S, Rt> SyncStage<Rt> for Skip<S>
where
    S: SyncStage<Rt>,
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        while self.remaining > 0 {
            self.remaining -= 1;
            self.source.next(rt, frame)?;
        }
        self.source.next(rt, frame)
    }
}

impl<S, Rt> AsyncStage<Rt> for Skip<S>
where
    S: AsyncStage<Rt>,
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>> {
        Box::pin(async move {
            while self.remaining > 0 {
                self.remaining -= 1;
                self.source.next(rt, frame).await?;
            }
            self.source.next(rt, frame).await
        })
    }
}

struct StepBy<S> {
    source: S,
    step: u64,
    started: bool,
}

impl<S, Rt> SyncStage<Rt> for StepBy<S>
where
    S: SyncStage<Rt>,
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        if self.started {
            for _ in 1..self.step {
                self.source.next(rt, frame)?;
            }
        }
        self.started = true;
        self.source.next(rt, frame)
    }
}

impl<S, Rt> AsyncStage<Rt> for StepBy<S>
where
    S: AsyncStage<Rt>,
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>> {
        Box::pin(async move {
            if self.started {
                for _ in 1..self.step {
                    self.source.next(rt, frame).await?;
                }
            }
            self.started = true;
            self.source.next(rt, frame).await
        })
    }
}

struct TakeWhile<S, T, E, Rt>
where
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    source: S,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
    done: bool,
}

impl<S, T, E, Rt> SyncStage<Rt> for TakeWhile<S, T, E, Rt>
where
    S: SyncStage<Rt>,
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        if self.done {
            return None;
        }
        let value = self.source.next(rt, frame)?;
        if self.f.call_now(rt, frame, (Ref::lend(rt, &value),)) {
            return Some(value);
        }
        self.done = true;
        None
    }
}

impl<S, T, E, Rt> AsyncStage<Rt> for TakeWhile<S, T, E, Rt>
where
    S: AsyncStage<Rt>,
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>> {
        Box::pin(async move {
            let TakeWhile { source, f, done } = self;
            if *done {
                return None;
            }
            let value = source.next(rt, frame).await?;
            if f.call(rt, frame, (Ref::lend(rt, &value),)).await {
                return Some(value);
            }
            *done = true;
            None
        })
    }
}

struct SkipWhile<S, T, E, Rt>
where
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    source: S,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
    skipping: bool,
}

impl<S, T, E, Rt> SyncStage<Rt> for SkipWhile<S, T, E, Rt>
where
    S: SyncStage<Rt>,
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        while self.skipping {
            let value = self.source.next(rt, frame)?;
            if !self.f.call_now(rt, frame, (Ref::lend(rt, &value),)) {
                self.skipping = false;
                return Some(value);
            }
        }
        self.source.next(rt, frame)
    }
}

impl<S, T, E, Rt> AsyncStage<Rt> for SkipWhile<S, T, E, Rt>
where
    S: AsyncStage<Rt>,
    T: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>> {
        Box::pin(async move {
            let SkipWhile {
                source,
                f,
                skipping,
            } = self;
            while *skipping {
                let value = source.next(rt, frame).await?;
                if !f.call(rt, frame, (Ref::lend(rt, &value),)).await {
                    *skipping = false;
                    return Some(value);
                }
            }
            source.next(rt, frame).await
        })
    }
}

struct Chunks<S, T> {
    source: S,
    size: u64,
    item: PhantomData<T>,
}

impl<S, T, Rt> SyncStage<Rt> for Chunks<S, T>
where
    S: SyncStage<Rt>,
    T: OneValue<Rt> + FromValue<Rt> + Send + Sync,
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        let mut chunk: Vec<T> = Vec::new();
        while (chunk.len() as u64) < self.size {
            let Some(value) = self.source.next(rt, frame) else {
                break;
            };
            chunk.push(T::from_value(rt, value));
        }
        (!chunk.is_empty()).then(|| <Vec<T> as OneValue<Rt>>::erase(chunk, rt))
    }
}

impl<S, T, Rt> AsyncStage<Rt> for Chunks<S, T>
where
    S: AsyncStage<Rt>,
    T: OneValue<Rt> + FromValue<Rt> + Send + Sync,
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>> {
        Box::pin(async move {
            let mut chunk: Vec<T> = Vec::new();
            while (chunk.len() as u64) < self.size {
                let Some(value) = self.source.next(rt, frame).await else {
                    break;
                };
                chunk.push(T::from_value(rt, value));
            }
            (!chunk.is_empty()).then(|| <Vec<T> as OneValue<Rt>>::erase(chunk, rt))
        })
    }
}

struct Dedup<S, T> {
    source: S,
    last: Option<T>,
}

impl<S, T, Rt> SyncStage<Rt> for Dedup<S, T>
where
    S: SyncStage<Rt>,
    T: Stored<Rt> + PartialEq + Clone,
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        while let Some(value) = self.source.next(rt, frame) {
            let item = Erased::<Rt, T>::from_value(rt, value);
            let current = item.as_ref(rt);
            if self.last.as_ref() == Some(current) {
                continue;
            }
            self.last = Some(current.clone());
            return Some(item.into_value());
        }
        None
    }
}

impl<S, T, Rt> AsyncStage<Rt> for Dedup<S, T>
where
    S: AsyncStage<Rt>,
    T: Stored<Rt> + PartialEq + Clone,
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>> {
        Box::pin(async move {
            while let Some(value) = self.source.next(rt, frame).await {
                let item = Erased::<Rt, T>::from_value(rt, value);
                let current = item.as_ref(rt);
                if self.last.as_ref() == Some(current) {
                    continue;
                }
                self.last = Some(current.clone());
                return Some(item.into_value());
            }
            None
        })
    }
}

struct Chain<S> {
    parts: VecDeque<S>,
}

impl<S, Rt> SyncStage<Rt> for Chain<S>
where
    S: SyncStage<Rt> + Send + Sync,
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        while let Some(front) = self.parts.front_mut() {
            if let Some(value) = front.next(rt, frame) {
                return Some(value);
            }
            self.parts.pop_front();
        }
        None
    }
}

impl<S, Rt> AsyncStage<Rt> for Chain<S>
where
    S: AsyncStage<Rt> + Send + Sync,
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>> {
        Box::pin(async move {
            while let Some(front) = self.parts.front_mut() {
                if let Some(value) = front.next(rt, frame).await {
                    return Some(value);
                }
                self.parts.pop_front();
            }
            None
        })
    }
}

struct Flatten<S, T>
where
    T: IntoIterator,
{
    source: S,
    pending: Option<T::IntoIter>,
}

impl<S, T, U, Rt> SyncStage<Rt> for Flatten<S, T>
where
    S: SyncStage<Rt>,
    T: FromValue<Rt> + IntoIterator<Item = U> + Send + Sync,
    T::IntoIter: Send + Sync,
    U: OneValue<Rt>,
    Rt: Runtime,
{
    fn next(&mut self, rt: &Rt, frame: &mut Rt::Frame<'_>) -> Option<Rt::Value> {
        loop {
            if let Some(item) = self.pending.as_mut().and_then(Iterator::next) {
                return Some(item.erase(rt));
            }
            let value = self.source.next(rt, frame)?;
            self.pending = Some(T::from_value(rt, value).into_iter());
        }
    }
}

impl<S, T, U, Rt> AsyncStage<Rt> for Flatten<S, T>
where
    S: AsyncStage<Rt>,
    T: FromValue<Rt> + IntoIterator<Item = U> + Send + Sync,
    T::IntoIter: Send + Sync,
    U: OneValue<Rt>,
    Rt: Runtime,
{
    fn next<'a>(
        &'a mut self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
    ) -> BoxFuture<'a, Option<Rt::Value>> {
        Box::pin(async move {
            loop {
                if let Some(item) = self.pending.as_mut().and_then(Iterator::next) {
                    return Some(item.erase(rt));
                }
                let value = self.source.next(rt, frame).await?;
                self.pending = Some(T::from_value(rt, value).into_iter());
            }
        })
    }
}
