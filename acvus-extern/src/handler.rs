//! Type-erased handlers, ready for a runtime to call.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_mir::ty::{PolyTy, Ty, matches_poly};

use crate::error::ExternError;
use crate::runtime::Runtime;

/// What a call gives back: its return value, and the value of every
/// place it borrowed, in argument order (RFC-0015).
pub struct Returned<V> {
    pub value: V,
    pub lent: Vec<V>,
}

impl<V> Returned<V> {
    pub fn value(value: V) -> Self {
        Returned {
            value,
            lent: Vec::new(),
        }
    }
}

type SyncFn<R> = dyn Fn(
        &R,
        Vec<<R as Runtime>::Value>,
    ) -> Result<Returned<<R as Runtime>::Value>, <R as Runtime>::Error>
    + Send
    + Sync;
type AsyncFn<R> = dyn Fn(
        R,
        Vec<<R as Runtime>::Value>,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Returned<<R as Runtime>::Value>, <R as Runtime>::Error>>
                + Send,
        >,
    > + Send
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
