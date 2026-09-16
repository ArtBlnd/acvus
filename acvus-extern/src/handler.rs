//! Type-erased handlers, ready for a runtime to call.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_mir::ty::PolyTy;

use crate::runtime::Runtime;

type SyncFn<R> = dyn Fn(&R, Vec<<R as Runtime>::Value>) -> Result<<R as Runtime>::Value, <R as Runtime>::Error>
    + Send
    + Sync;
type AsyncFn<R> = dyn Fn(
        R,
        Vec<<R as Runtime>::Value>,
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

pub struct Instance<R: Runtime> {
    pub signature: PolyTy,
    pub handler: ExternHandler<R>,
}

/// The number a call carries in `Callee::Extern` is an index into
/// `into_handlers`, and the compiler assigns it from `signatures`: the two
/// lists are the same list in the same order, and `acvus_mir::ty::Instances`
/// is the compiler's half of that contract.
pub struct Instances<R: Runtime> {
    pub concrete: Vec<Instance<R>>,
    pub generic: Option<ExternHandler<R>>,
}

impl<R: Runtime> Instances<R> {
    pub fn generic(handler: ExternHandler<R>) -> Self {
        Self {
            concrete: Vec::new(),
            generic: Some(handler),
        }
    }

    pub fn signatures(&self) -> acvus_mir::ty::Instances {
        acvus_mir::ty::Instances {
            concrete: self.concrete.iter().map(|i| i.signature.clone()).collect(),
            generic: self.generic.is_some(),
        }
    }

    pub fn into_handlers(self) -> Vec<ExternHandler<R>> {
        self.concrete
            .into_iter()
            .map(|i| i.handler)
            .chain(self.generic)
            .collect()
    }
}
