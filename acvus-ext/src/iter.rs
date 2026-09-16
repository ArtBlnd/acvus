//! The `Iterator` extension type: a lazy, move-only pipeline of stages.
//!
//! A stage is built by one constructor that takes its source and its
//! closure at one element type, then boxed as `dyn Stage<Rt>`, which yields
//! the runtime's values. The box is the payload that crosses the boundary,
//! and it names no element type: the glue erases a `Regex`'s `Iter<String>`
//! and materializes the same box as a consumer's `Iter<T>`, so a payload
//! that mentioned `T` would be two Rust types on the two sides.
//!
//! `T` is the declared element type and nothing more: no value is read as a
//! `T` on its account. The exit is `FromValue`, identity for the runtime's
//! value and a checked downcast for `Erased` and the container boxes, so the
//! pipeline's construction is never cited as a proof.

use std::collections::VecDeque;
use std::marker::PhantomData;

use acvus_extern::{
    BoxFuture, ClosureFn, Cross, EffectVar, ExternType, Fn1, FromValue, IdentityVar, Ref, Runtime,
    TyVar,
};
use sync_wrapper::SyncWrapper;

#[derive(ExternType)]
#[extern_type(name = "Iterator")]
#[repr(transparent)]
pub struct Iter<T, E, I, Rt>(Box<dyn Stage<Rt>>, PhantomData<(T, E, I)>)
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime;

type Pulled<'a, Rt> = BoxFuture<'a, Result<Option<<Rt as Runtime>::Value>, <Rt as Runtime>::Error>>;

trait Stage<Rt>: Send + Sync
where
    Rt: Runtime,
{
    fn next<'a>(&'a mut self, rt: &'a Rt) -> Pulled<'a, Rt>;
}

impl<T, E, I, Rt> Iter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    fn sealed(stage: impl Stage<Rt> + 'static) -> Self {
        Self(Box::new(stage), PhantomData)
    }

    pub fn from_items(items: Vec<T>) -> Self
    where
        T: Cross<Rt>,
    {
        let mut items = items.into_iter();
        Self::generate(move |_| items.next())
    }

    pub fn generate<F>(f: F) -> Self
    where
        F: FnMut(&Rt) -> Option<T> + Send + 'static,
        T: Cross<Rt>,
    {
        Self::sealed(Generate(SyncWrapper::new(f)))
    }

    pub fn map<U>(self, f: Fn1<T, U, E, Rt>) -> Iter<U, E, I, Rt>
    where
        U: TyVar,
    {
        Iter::sealed(Map { source: self, f })
    }

    pub fn filter(self, f: Fn1<Ref<T, Rt>, bool, E, Rt>) -> Self {
        Self::sealed(Filter { source: self, f })
    }

    pub fn take(self, n: u64) -> Self {
        Self::sealed(Take {
            source: self,
            remaining: n,
        })
    }

    pub fn skip(self, n: u64) -> Self {
        Self::sealed(Skip {
            source: self,
            remaining: n,
        })
    }

    pub fn chain<J, K>(self, other: Iter<T, E, J, Rt>) -> Iter<T, E, K, Rt>
    where
        J: IdentityVar,
        K: IdentityVar,
    {
        Iter::sealed(Chain {
            parts: VecDeque::from([self.0, other.0]),
        })
    }

    pub fn chain_all<K>(parts: Vec<Self>) -> Iter<T, E, K, Rt>
    where
        K: IdentityVar,
    {
        Iter::sealed(Chain {
            parts: parts.into_iter().map(|part| part.0).collect(),
        })
    }

    pub fn flatten<U>(self) -> Iter<U, E, I, Rt>
    where
        T: FromValue<Rt> + IntoIterator<Item = U>,
        T::IntoIter: Send + Sync,
        U: Cross<Rt>,
    {
        Iter::sealed(Flatten {
            source: self,
            pending: None,
        })
    }

    pub fn flat_map<S, U>(self, f: Fn1<T, S, E, Rt>) -> Iter<U, E, I, Rt>
    where
        S: TyVar + FromValue<Rt> + IntoIterator<Item = U>,
        S::IntoIter: Send + Sync,
        U: Cross<Rt>,
    {
        self.map(f).flatten()
    }

    pub async fn next_value(&mut self, rt: &Rt) -> Result<Option<Rt::Value>, Rt::Error> {
        self.0.next(rt).await
    }

    pub async fn next(&mut self, rt: &Rt) -> Result<Option<T>, Rt::Error>
    where
        T: FromValue<Rt>,
    {
        let Some(value) = self.next_value(rt).await? else {
            return Ok(None);
        };
        Ok(Some(T::from_value(rt, value)?))
    }
}

struct Generate<F>(SyncWrapper<F>);

impl<F, T, Rt> Stage<Rt> for Generate<F>
where
    F: FnMut(&Rt) -> Option<T> + Send,
    T: Cross<Rt>,
    Rt: Runtime,
{
    fn next<'a>(&'a mut self, rt: &'a Rt) -> Pulled<'a, Rt> {
        Box::pin(async move { Ok(self.0.get_mut()(rt).map(|item| item.erase(rt))) })
    }
}

struct Map<T, U, E, I, Rt>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    source: Iter<T, E, I, Rt>,
    f: Fn1<T, U, E, Rt>,
}

impl<T, U, E, I, Rt> Stage<Rt> for Map<T, U, E, I, Rt>
where
    T: TyVar,
    U: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    fn next<'a>(&'a mut self, rt: &'a Rt) -> Pulled<'a, Rt> {
        Box::pin(async move {
            let Some(value) = self.source.next_value(rt).await? else {
                return Ok(None);
            };
            Ok(Some(self.f.call_value(rt, value).await?))
        })
    }
}

struct Filter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    source: Iter<T, E, I, Rt>,
    f: Fn1<Ref<T, Rt>, bool, E, Rt>,
}

impl<T, E, I, Rt> Stage<Rt> for Filter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    fn next<'a>(&'a mut self, rt: &'a Rt) -> Pulled<'a, Rt> {
        Box::pin(async move {
            while let Some(value) = self.source.next_value(rt).await? {
                if self.f.call(rt, (Ref::lend(rt, &value),)).await? {
                    return Ok(Some(value));
                }
            }
            Ok(None)
        })
    }
}

struct Take<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    source: Iter<T, E, I, Rt>,
    remaining: u64,
}

impl<T, E, I, Rt> Stage<Rt> for Take<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    fn next<'a>(&'a mut self, rt: &'a Rt) -> Pulled<'a, Rt> {
        Box::pin(async move {
            if self.remaining == 0 {
                return Ok(None);
            }
            self.remaining -= 1;
            self.source.next_value(rt).await
        })
    }
}

struct Skip<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    source: Iter<T, E, I, Rt>,
    remaining: u64,
}

impl<T, E, I, Rt> Stage<Rt> for Skip<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    fn next<'a>(&'a mut self, rt: &'a Rt) -> Pulled<'a, Rt> {
        Box::pin(async move {
            while self.remaining > 0 {
                self.remaining -= 1;
                if self.source.next_value(rt).await?.is_none() {
                    return Ok(None);
                }
            }
            self.source.next_value(rt).await
        })
    }
}

struct Chain<Rt>
where
    Rt: Runtime,
{
    parts: VecDeque<Box<dyn Stage<Rt>>>,
}

impl<Rt> Stage<Rt> for Chain<Rt>
where
    Rt: Runtime,
{
    fn next<'a>(&'a mut self, rt: &'a Rt) -> Pulled<'a, Rt> {
        Box::pin(async move {
            while let Some(front) = self.parts.front_mut() {
                if let Some(value) = front.next(rt).await? {
                    return Ok(Some(value));
                }
                self.parts.pop_front();
            }
            Ok(None)
        })
    }
}

struct Flatten<S, E, I, Rt>
where
    S: TyVar + IntoIterator,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    source: Iter<S, E, I, Rt>,
    pending: Option<S::IntoIter>,
}

impl<S, U, E, I, Rt> Stage<Rt> for Flatten<S, E, I, Rt>
where
    S: TyVar + FromValue<Rt> + IntoIterator<Item = U>,
    S::IntoIter: Send + Sync,
    U: Cross<Rt>,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    fn next<'a>(&'a mut self, rt: &'a Rt) -> Pulled<'a, Rt> {
        Box::pin(async move {
            loop {
                if let Some(item) = self.pending.as_mut().and_then(Iterator::next) {
                    return Ok(Some(item.erase(rt)));
                }
                let Some(value) = self.source.next_value(rt).await? else {
                    return Ok(None);
                };
                self.pending = Some(S::from_value(rt, value)?.into_iter());
            }
        })
    }
}
