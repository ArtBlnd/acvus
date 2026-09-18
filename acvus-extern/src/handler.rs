//! Type-erased handlers, ready for a runtime to call.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_mir::ty::{PolyTy, Task};

use crate::runtime::Runtime;
use crate::slice::Elements;

/// A synchronous handler takes its arguments **by value, one Rust
/// parameter each, up to three** (RFC-0044, stage 2c). A `Value` is a
/// scalar pair, so `(&R, Value, Value, Value)` is seven scalars under the
/// `rust-call` ABI and the caller passes each argument in a register
/// rather than staging it into a contiguous run of its own registers.
///
/// Four or more arguments do not fit that ABI, so `ArityN` keeps the
/// window: the handler is lent the caller's argument slots and takes each
/// value out of the one it reads, and what it leaves behind is the
/// caller's business (stage 2b).
///
/// The return carries no failure. A handler that could not produce a
/// result leaves its trap on the runtime through `Runtime::trap` and
/// returns `Runtime::empty`, which is what the caller tests for.
type Sync0<R> = dyn Fn(&R) -> <R as Runtime>::Value + Send + Sync;
type Sync1<R> = dyn Fn(&R, <R as Runtime>::Value) -> <R as Runtime>::Value + Send + Sync;
type Sync2<R> =
    dyn Fn(&R, <R as Runtime>::Value, <R as Runtime>::Value) -> <R as Runtime>::Value + Send + Sync;
type Sync3<R> = dyn Fn(
        &R,
        <R as Runtime>::Value,
        <R as Runtime>::Value,
        <R as Runtime>::Value,
    ) -> <R as Runtime>::Value
    + Send
    + Sync;
type SyncN<R> = dyn Fn(&R, &mut [<R as Runtime>::Value]) -> <R as Runtime>::Value + Send + Sync;
type ElementsFn<R> = dyn Fn(&R, <R as Runtime>::Value) -> Elements<R> + Send + Sync;
type AsyncFn<R> = dyn Fn(
        R,
        &mut [<R as Runtime>::Value],
    ) -> Pin<Box<dyn Future<Output = <R as Runtime>::Value> + Send>>
    + Send
    + Sync;

/// A declaration that returns a slice, at the one arity such a
/// declaration has: its container reference in, the run's pointer and
/// length out as a Rust `ScalarPair`, in two machine registers with no
/// `Value` and no allocation (RFC-0047 §6).
pub struct SliceHandler<R>
where
    R: Runtime,
{
    elements: Arc<ElementsFn<R>>,
}

impl<R> SliceHandler<R>
where
    R: Runtime,
{
    pub fn new<F>(elements: F) -> Self
    where
        F: Fn(&R, R::Value) -> Elements<R> + Send + Sync + 'static,
    {
        Self {
            elements: Arc::new(elements),
        }
    }

    pub fn elements(&self, rt: &R, container: R::Value) -> Elements<R> {
        (self.elements)(rt, container)
    }

    pub fn boxed(&self, rt: &R, container: R::Value) -> R::Value {
        // SAFETY: `Slice::materialize` reads the box back as this same
        // `Elements<R>`, which is the only crossing either slice type has.
        unsafe { rt.erase::<Elements<R>>(self.elements(rt, container)) }
    }
}

impl<R> Clone for SliceHandler<R>
where
    R: Runtime,
{
    fn clone(&self) -> Self {
        Self {
            elements: Arc::clone(&self.elements),
        }
    }
}

/// The by-value ABI. The variant is fixed when the handler is built: the
/// macro reads the arity and the return type off the signature, and the
/// caller's operation is chosen to match.
pub enum SyncHandler<R>
where
    R: Runtime,
{
    Arity0(Arc<Sync0<R>>),
    Arity1(Arc<Sync1<R>>),
    Arity2(Arc<Sync2<R>>),
    Arity3(Arc<Sync3<R>>),
    ArityN(Arc<SyncN<R>>),
    Slice(SliceHandler<R>),
}

impl<R> Clone for SyncHandler<R>
where
    R: Runtime,
{
    fn clone(&self) -> Self {
        match self {
            Self::Arity0(f) => Self::Arity0(Arc::clone(f)),
            Self::Arity1(f) => Self::Arity1(Arc::clone(f)),
            Self::Arity2(f) => Self::Arity2(Arc::clone(f)),
            Self::Arity3(f) => Self::Arity3(Arc::clone(f)),
            Self::ArityN(f) => Self::ArityN(Arc::clone(f)),
            Self::Slice(f) => Self::Slice(f.clone()),
        }
    }
}

impl<R> SyncHandler<R>
where
    R: Runtime,
{
    /// The arity the ABI names, and `None` for the window form, whose
    /// arity is whatever the caller's slice holds.
    pub fn arity(&self) -> Option<usize> {
        match self {
            Self::Arity0(_) => Some(0),
            Self::Arity1(_) => Some(1),
            Self::Arity2(_) => Some(2),
            Self::Arity3(_) => Some(3),
            Self::Slice(_) => Some(1),
            Self::ArityN(_) => None,
        }
    }

    /// Calls the handler with its arguments in a slice, taking each out of
    /// its place. This is for a caller that does not know the arity where
    /// it stands — a spawn that must own the arguments, a test. The
    /// interpreter's call path does not come here: it reads each argument
    /// from the register the operation names and passes it by value.
    ///
    /// # Panics
    /// `args` is not as long as the arity the handler names.
    pub fn call_taking(&self, rt: &R, args: &mut [R::Value]) -> R::Value {
        if let Some(arity) = self.arity() {
            assert_eq!(
                args.len(),
                arity,
                "a handler of arity {arity} was called with {} arguments",
                args.len()
            );
        }
        match self {
            Self::Arity0(f) => f(rt),
            Self::Arity1(f) => {
                let a0 = std::mem::take(&mut args[0]);
                f(rt, a0)
            }
            Self::Arity2(f) => {
                let a0 = std::mem::take(&mut args[0]);
                let a1 = std::mem::take(&mut args[1]);
                f(rt, a0, a1)
            }
            Self::Arity3(f) => {
                let a0 = std::mem::take(&mut args[0]);
                let a1 = std::mem::take(&mut args[1]);
                let a2 = std::mem::take(&mut args[2]);
                f(rt, a0, a1, a2)
            }
            Self::Slice(f) => {
                let a0 = std::mem::take(&mut args[0]);
                f.boxed(rt, a0)
            }
            Self::ArityN(f) => f(rt, args),
        }
    }
}

/// One handler per rung of `Task` (RFC-0046). `Sync` runs to its result
/// in the caller's frame; `Heavy` is a Rust `fn` all the same, but one the
/// runtime hands to `Executor::spawn_blocking` and awaits; `Async` runs on
/// the async runtime and owns its interner because it lives across await
/// points.
pub enum ExternHandler<R: Runtime> {
    Sync(SyncHandler<R>),
    Heavy(SyncHandler<R>),
    Async(Arc<AsyncFn<R>>),
}

impl<R: Runtime> Clone for ExternHandler<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Sync(f) => Self::Sync(f.clone()),
            Self::Heavy(f) => Self::Heavy(f.clone()),
            Self::Async(f) => Self::Async(Arc::clone(f)),
        }
    }
}

impl<R: Runtime> ExternHandler<R> {
    /// Whether the call reaches its result without the caller suspending.
    /// A `Heavy` handler does not: it is offloaded and awaited.
    pub fn is_sync(&self) -> bool {
        match self {
            Self::Sync(_) => true,
            Self::Heavy(_) | Self::Async(_) => false,
        }
    }

    /// The task this handler runs at, which is the task its declaration
    /// named.
    pub fn task(&self) -> Task {
        match self {
            Self::Sync(_) => Task::Sync,
            Self::Async(_) => Task::Async,
            Self::Heavy(_) => Task::Heavy,
        }
    }
}

pub struct Instance<R: Runtime> {
    pub signature: PolyTy,
    pub handler: ExternHandler<R>,
    /// The greatest task this instance runs — a ceiling, "at most", not
    /// the instance's own task. The `async fn` glue admits `Heavy` as well
    /// as `Async`, because it awaits either; the plain `fn` glue admits
    /// only `Sync`.
    pub admits: Task,
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
            concrete: self
                .concrete
                .iter()
                .map(|i| acvus_mir::ty::InstanceSig {
                    ty: i.signature.clone(),
                    admits: i.admits,
                })
                .collect(),
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
