//! Type-erased handlers, ready for a runtime to call.

use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_mir::ty::{PolyTy, Task};

use crate::runtime::Runtime;
use crate::slice::Elements;

/// A `#[state]` parameter, supplied when the registry is built and read
/// by the handler as its first argument. The call operation holds the
/// `Arc` as a field, so the operation's type says a state exists and the
/// handler runs without testing for one (RFC-0052 §6).
pub type State = Arc<dyn Any + Send + Sync>;
pub type StateRef<'a> = &'a (dyn Any + Send + Sync);

pub type Sync0<R> = fn(&R) -> <R as Runtime>::Value;
pub type Sync1<R> = fn(&R, <R as Runtime>::Value) -> <R as Runtime>::Value;
pub type Sync2<R> = fn(&R, <R as Runtime>::Value, <R as Runtime>::Value) -> <R as Runtime>::Value;
pub type Sync3<R> = fn(
    &R,
    <R as Runtime>::Value,
    <R as Runtime>::Value,
    <R as Runtime>::Value,
) -> <R as Runtime>::Value;
pub type SyncWindow<R> = fn(&R, &[<R as Runtime>::Value]) -> <R as Runtime>::Value;
pub type SyncSlice<R> = fn(&R, <R as Runtime>::Value) -> Elements<R>;

pub type State0<R> = fn(StateRef<'_>, &R) -> <R as Runtime>::Value;
pub type State1<R> = fn(StateRef<'_>, &R, <R as Runtime>::Value) -> <R as Runtime>::Value;
pub type State2<R> =
    fn(StateRef<'_>, &R, <R as Runtime>::Value, <R as Runtime>::Value) -> <R as Runtime>::Value;
pub type State3<R> = fn(
    StateRef<'_>,
    &R,
    <R as Runtime>::Value,
    <R as Runtime>::Value,
    <R as Runtime>::Value,
) -> <R as Runtime>::Value;
pub type StateWindow<R> = fn(StateRef<'_>, &R, &[<R as Runtime>::Value]) -> <R as Runtime>::Value;
pub type StateSlice<R> = fn(StateRef<'_>, &R, <R as Runtime>::Value) -> Elements<R>;

pub type Async<R> =
    fn(R, &[<R as Runtime>::Value]) -> Pin<Box<dyn Future<Output = <R as Runtime>::Value> + Send>>;
pub type AsyncState<R> = fn(
    State,
    R,
    &[<R as Runtime>::Value],
) -> Pin<Box<dyn Future<Output = <R as Runtime>::Value> + Send>>;

/// Which `fn` shape a declaration compiled to. A declaration of three or
/// fewer parameters takes each in a register; four or more take the
/// caller's argument window, and a slice return takes its run back in two
/// registers with no boxing (RFC-0047 §6).
///
/// `prepare` reads this once, to pick the call operation's type; the
/// operation then holds the bare `fn` pointer and calls it with no
/// decision in between.
pub enum SyncAbi<R>
where
    R: Runtime,
{
    Arity0(Sync0<R>),
    Arity1(Sync1<R>),
    Arity2(Sync2<R>),
    Arity3(Sync3<R>),
    Window(SyncWindow<R>),
    Slice(SyncSlice<R>),
}

pub enum StateAbi<R>
where
    R: Runtime,
{
    Arity0(State0<R>),
    Arity1(State1<R>),
    Arity2(State2<R>),
    Arity3(State3<R>),
    Window(StateWindow<R>),
    Slice(StateSlice<R>),
}

pub enum SyncCall<R>
where
    R: Runtime,
{
    Plain(SyncAbi<R>),
    Stateful { state: State, abi: StateAbi<R> },
}

pub enum AsyncCall<R>
where
    R: Runtime,
{
    Plain(Async<R>),
    Stateful { state: State, f: AsyncState<R> },
}

impl<R> Clone for SyncAbi<R>
where
    R: Runtime,
{
    fn clone(&self) -> Self {
        match *self {
            Self::Arity0(f) => Self::Arity0(f),
            Self::Arity1(f) => Self::Arity1(f),
            Self::Arity2(f) => Self::Arity2(f),
            Self::Arity3(f) => Self::Arity3(f),
            Self::Window(f) => Self::Window(f),
            Self::Slice(f) => Self::Slice(f),
        }
    }
}

impl<R> Clone for StateAbi<R>
where
    R: Runtime,
{
    fn clone(&self) -> Self {
        match *self {
            Self::Arity0(f) => Self::Arity0(f),
            Self::Arity1(f) => Self::Arity1(f),
            Self::Arity2(f) => Self::Arity2(f),
            Self::Arity3(f) => Self::Arity3(f),
            Self::Window(f) => Self::Window(f),
            Self::Slice(f) => Self::Slice(f),
        }
    }
}

impl<R> Clone for SyncCall<R>
where
    R: Runtime,
{
    fn clone(&self) -> Self {
        match self {
            Self::Plain(abi) => Self::Plain(abi.clone()),
            Self::Stateful { state, abi } => Self::Stateful {
                state: Arc::clone(state),
                abi: abi.clone(),
            },
        }
    }
}

impl<R> Clone for AsyncCall<R>
where
    R: Runtime,
{
    fn clone(&self) -> Self {
        match self {
            Self::Plain(f) => Self::Plain(*f),
            Self::Stateful { state, f } => Self::Stateful {
                state: Arc::clone(state),
                f: *f,
            },
        }
    }
}

impl<R> SyncCall<R>
where
    R: Runtime,
{
    /// The arity the ABI names, and `None` for the window form, whose
    /// arity is whatever the caller's slice holds.
    pub fn arity(&self) -> Option<usize> {
        let at = |plain: &SyncAbi<R>| match plain {
            SyncAbi::Arity0(_) => Some(0),
            SyncAbi::Arity1(_) | SyncAbi::Slice(_) => Some(1),
            SyncAbi::Arity2(_) => Some(2),
            SyncAbi::Arity3(_) => Some(3),
            SyncAbi::Window(_) => None,
        };
        match self {
            Self::Plain(abi) => at(abi),
            Self::Stateful { abi, .. } => match abi {
                StateAbi::Arity0(_) => Some(0),
                StateAbi::Arity1(_) | StateAbi::Slice(_) => Some(1),
                StateAbi::Arity2(_) => Some(2),
                StateAbi::Arity3(_) => Some(3),
                StateAbi::Window(_) => None,
            },
        }
    }

    /// Calls the handler with its arguments in a slice, taking each out of
    /// its place: for a caller that does not know the arity where it
    /// stands — a spawn that must own the arguments, a test. The
    /// interpreter's call path does not come here; it reads each argument
    /// from the register the operation names and passes it by value.
    ///
    /// # Panics
    /// `args` is not as long as the arity the handler names.
    pub fn call_taking(&self, rt: &R, args: &[R::Value]) -> R::Value {
        if let Some(arity) = self.arity() {
            assert_eq!(
                args.len(),
                arity,
                "a handler of arity {arity} was called with {} arguments",
                args.len()
            );
        }
        let take = |args: &[R::Value], at: usize| args[at];
        match self {
            Self::Plain(abi) => match abi {
                SyncAbi::Arity0(f) => f(rt),
                SyncAbi::Arity1(f) => f(rt, take(args, 0)),
                SyncAbi::Arity2(f) => {
                    let a0 = take(args, 0);
                    let a1 = take(args, 1);
                    f(rt, a0, a1)
                }
                SyncAbi::Arity3(f) => {
                    let a0 = take(args, 0);
                    let a1 = take(args, 1);
                    let a2 = take(args, 2);
                    f(rt, a0, a1, a2)
                }
                SyncAbi::Window(f) => f(rt, args),
                SyncAbi::Slice(f) => erase_elements(rt, f(rt, take(args, 0))),
            },
            Self::Stateful { state, abi } => {
                let s: StateRef<'_> = state.as_ref();
                match abi {
                    StateAbi::Arity0(f) => f(s, rt),
                    StateAbi::Arity1(f) => f(s, rt, take(args, 0)),
                    StateAbi::Arity2(f) => {
                        let a0 = take(args, 0);
                        let a1 = take(args, 1);
                        f(s, rt, a0, a1)
                    }
                    StateAbi::Arity3(f) => {
                        let a0 = take(args, 0);
                        let a1 = take(args, 1);
                        let a2 = take(args, 2);
                        f(s, rt, a0, a1, a2)
                    }
                    StateAbi::Window(f) => f(s, rt, args),
                    StateAbi::Slice(f) => erase_elements(rt, f(s, rt, take(args, 0))),
                }
            }
        }
    }
}

/// The run a slice-returning declaration produced, as a value: the form a
/// caller takes when it wants a `Value` rather than the two registers.
pub fn erase_elements<R>(rt: &R, elements: Elements<R>) -> R::Value
where
    R: Runtime,
{
    // SAFETY: `Slice::materialize` reads the box back as this same
    // `Elements<R>`, which is the only crossing either slice type has.
    unsafe { rt.erase::<Elements<R>>(elements) }
}

/// One handler per rung of `Task` (RFC-0046). `Sync` runs to its result
/// in the caller's frame; `Heavy` is a Rust `fn` all the same, but one the
/// runtime hands to `Executor::spawn_blocking` and awaits; `Async` runs on
/// the async runtime and owns its interner because it lives across await
/// points.
pub enum ExternHandler<R: Runtime> {
    Sync(SyncCall<R>),
    Heavy(SyncCall<R>),
    Async(AsyncCall<R>),
}

impl<R: Runtime> Clone for ExternHandler<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Sync(f) => Self::Sync(f.clone()),
            Self::Heavy(f) => Self::Heavy(f.clone()),
            Self::Async(f) => Self::Async(f.clone()),
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
