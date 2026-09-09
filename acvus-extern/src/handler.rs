//! Type-erased handlers, ready for a runtime to call.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_mir::ty::{PolyTy, Ty, matches_poly};
use acvus_utils::Interner;

use crate::convert::{FromValues, IntoValue};
use crate::error::ExternError;
use crate::runtime::Runtime;

type SyncFn<R> = dyn Fn(
        Vec<<R as Runtime>::Value>,
        &Interner,
    ) -> Result<<R as Runtime>::Value, <R as Runtime>::Error>
    + Send
    + Sync;
type AsyncFn<R> = dyn Fn(
        Vec<<R as Runtime>::Value>,
        Interner,
    )
        -> Pin<Box<dyn Future<Output = Result<<R as Runtime>::Value, <R as Runtime>::Error>> + Send>>
    + Send
    + Sync;

/// `Sync` may run on a blocking thread pool; `Async` runs on the async
/// runtime and owns its interner because it lives across await points.
pub enum ExternHandler<R: Runtime> {
    Sync(Arc<SyncFn<R>>),
    Async(Arc<AsyncFn<R>>),
}

impl<R: Runtime> Clone for ExternHandler<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Sync(f) => Self::Sync(Arc::clone(f)),
            Self::Async(f) => Self::Async(Arc::clone(f)),
        }
    }
}

impl<R: Runtime> ExternHandler<R> {
    pub fn is_sync(&self) -> bool {
        matches!(self, Self::Sync(_))
    }
}

pub fn into_sync_extern_handler<R, A, Ret, F>(f: F) -> ExternHandler<R>
where
    R: Runtime,
    F: Fn(&Interner, A) -> Result<Ret, R::Error> + Send + Sync + 'static,
    A: FromValues<R> + 'static,
    Ret: IntoValue<R> + 'static,
{
    ExternHandler::Sync(Arc::new(move |args, interner| {
        let a = A::from_values(args, interner)?;
        Ok(f(interner, a)?.into_value(interner))
    }))
}

pub fn into_async_extern_handler<R, A, Ret, F, Fut>(f: F) -> ExternHandler<R>
where
    R: Runtime,
    F: Fn(Interner, A) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Ret, R::Error>> + Send + 'static,
    A: FromValues<R> + 'static,
    Ret: IntoValue<R> + 'static,
{
    ExternHandler::Async(Arc::new(move |args, interner| {
        let a = match A::from_values(args, &interner) {
            Ok(v) => v,
            Err(e) => return Box::pin(std::future::ready(Err(e))),
        };
        let fut = f(interner.clone(), a);
        Box::pin(async move { Ok(fut.await?.into_value(&interner)) })
    }))
}

/// One instantiation of a monomorphized ExternFn: the function's type
/// with the bounded variable set to one member, and the handler compiled
/// for it.
pub struct MonoInstance<R: Runtime> {
    pub signature: PolyTy,
    pub handler: ExternHandler<R>,
}

/// The handlers of an ExternFn whose type variable ranges over a finite
/// set of concrete types. The instance whose signature matches the call's
/// resolved function type is the one to run.
pub struct MonoHandler<R: Runtime> {
    pub instances: Vec<MonoInstance<R>>,
}

/// What the registry holds for one ExternFn.
pub enum ExternEntry<R: Runtime> {
    Single(ExternHandler<R>),
    Mono(MonoHandler<R>),
}

impl<R: Runtime> Clone for ExternEntry<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Single(h) => Self::Single(h.clone()),
            Self::Mono(m) => Self::Mono(MonoHandler {
                instances: m
                    .instances
                    .iter()
                    .map(|i| MonoInstance {
                        signature: i.signature.clone(),
                        handler: i.handler.clone(),
                    })
                    .collect(),
            }),
        }
    }
}

impl<R: Runtime> ExternEntry<R> {
    /// The handler for a call whose resolved function type is `callee_ty`.
    pub fn select(&self, callee_ty: &Ty) -> Result<&ExternHandler<R>, ExternError> {
        match self {
            Self::Single(h) => Ok(h),
            Self::Mono(m) => m
                .instances
                .iter()
                .find(|i| matches_poly(callee_ty, &i.signature))
                .map(|i| &i.handler)
                .ok_or_else(|| {
                    ExternError::internal(format!(
                        "no instance of the ExternFn matches the call type {callee_ty:?}"
                    ))
                }),
        }
    }
}
