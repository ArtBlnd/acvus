//! Type-erased handlers, ready for a runtime to call.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_mir::ty::PolyTy;

use crate::runtime::Runtime;

/// A handler is lent the caller's argument slots and takes each value out
/// of the one it reads; the slice is the caller's, so what the handler
/// leaves behind is the caller's business (RFC-0044, stage 2b).
///
/// The return carries no failure. A handler that could not produce a
/// result leaves its trap on the runtime through `Runtime::trap` and
/// returns `Runtime::empty`, which is what the caller tests for.
type SyncFn<R> = dyn Fn(&R, &mut [<R as Runtime>::Value]) -> <R as Runtime>::Value + Send + Sync;
type AsyncFn<R> = dyn Fn(
        R,
        &mut [<R as Runtime>::Value],
    ) -> Pin<Box<dyn Future<Output = <R as Runtime>::Value> + Send>>
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

    /// Adds the instances of `more` whose signature is not already here:
    /// two declarations of one family cast at one member are one instance.
    pub fn add_concrete(&mut self, more: Vec<Instance<R>>) {
        for instance in more {
            let present = self
                .concrete
                .iter()
                .any(|existing| existing.signature == instance.signature);
            if !present {
                self.concrete.push(instance);
            }
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
