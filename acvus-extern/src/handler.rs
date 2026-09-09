//! Type-erased handlers, ready for a runtime to call.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_utils::Interner;

use crate::convert::{FromValues, IntoValue};
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
