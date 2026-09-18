//! Calls: the extern instances, the named functions, the closures, the
//! spawns, and the closure constructor.
//!
//! A synchronous extern is an operation and runs inside the block. Everything
//! that can suspend — an asynchronous extern, a heavy extern, a call into
//! another body, an `Eval` — is a terminator: it hands a `'static` future to
//! the driver and leaves the block at `SUSPEND`.
//!
//! `prepare` reads a handler's `SyncAbi`/`StateAbi` once, picks the operation
//! by the arity it names, and moves the bare `fn` in as a field; a `run` here
//! calls it with no decision in between (RFC-0052 §6).

use std::sync::Arc;

use acvus_extern::{AsyncCall, Elements, Owned, State, StateRef, erase_elements};
use acvus_mir::graph::QualifiedRef;
use futures::future::BoxFuture;
use smallvec::SmallVec;

use crate::code::{BlockId, Deref, Off, Op, SUSPEND, successor};
use crate::interpreter::lookup_module;
use crate::machine::{Machine, call_module, call_module_sync, fn_value_call};
use crate::runtime::{AcvusRuntime, SyncCall};
use crate::value::{FnValue, HandleValue, Value};

pub type Sync0 = acvus_extern::Sync0<AcvusRuntime>;
pub type Sync1 = acvus_extern::Sync1<AcvusRuntime>;
pub type Sync2 = acvus_extern::Sync2<AcvusRuntime>;
pub type Sync3 = acvus_extern::Sync3<AcvusRuntime>;
pub type SyncWindow = acvus_extern::SyncWindow<AcvusRuntime>;
pub type SyncSlice = acvus_extern::SyncSlice<AcvusRuntime>;

/// `acvus_extern::handler` declares these six as `State0<R>..StateSlice<R>`
/// but exports neither them nor its module, so they are spelled here at
/// `R = AcvusRuntime`. A `StateAbi` destructured in `prepare` is assigned
/// straight into these fields, which is where a divergence from the
/// declarations would surface.
pub type State0 = fn(StateRef<'_>, &AcvusRuntime) -> Value;
pub type State1 = fn(StateRef<'_>, &AcvusRuntime, Value) -> Value;
pub type State2 = fn(StateRef<'_>, &AcvusRuntime, Value, Value) -> Value;
pub type State3 = fn(StateRef<'_>, &AcvusRuntime, Value, Value, Value) -> Value;
pub type StateWindow = fn(StateRef<'_>, &AcvusRuntime, &[Value]) -> Value;
pub type StateSlice = fn(StateRef<'_>, &AcvusRuntime, Value) -> Elements<AcvusRuntime>;

pub type Async = acvus_extern::Async<AcvusRuntime>;
pub type AsyncState = acvus_extern::AsyncState<AcvusRuntime>;

/// A body is entered with the arguments it owns, so a call into one stages
/// them: each read where it stands, and the frame's claim on the ones this
/// call consumes dropped in one mask.
#[inline]
fn staged(m: &mut Machine<'_>, slots: &[Off], takes: u64) -> Vec<Value> {
    let regs = m.regs();
    let args = slots.iter().map(|slot| regs.read(*slot)).collect();
    regs.take_mask(takes);
    args
}

/// The run of registers a call's arguments sit in: `prepare` laid them out
/// contiguously and emitted the `Mov`s that put them there as operations
/// before this call, so what is left here is where the run starts, how wide
/// it is, and the frame's claim on the arguments the call consumes.
pub struct ArgWindow {
    pub at: Off,
    pub arity: u16,
    pub takes: u64,
}

impl ArgWindow {
    /// The registers themselves, lent to a handler that runs before this
    /// frame moves on (RFC-0044, stage 2b).
    #[inline]
    fn lend<'r>(&self, m: &'r mut Machine<'_>) -> &'r [Value] {
        let regs = m.regs();
        regs.take_mask(self.takes);
        regs.run_of(self.at, self.arity)
    }

    /// The arguments owned, for work that outlives this frame.
    fn own(&self, m: &mut Machine<'_>) -> Vec<Value> {
        self.lend(m).to_vec()
    }
}

pub struct CallExtern0<const LARGE: bool> {
    pub dst: Off,
    pub f: Sync0,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for CallExtern0<LARGE> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let value = (self.f)(m.rt);
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallExtern1<const LARGE: bool, const WORD: bool> {
    pub dst: Off,
    pub a: Off,
    pub takes: u64,
    pub f: Sync1,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool, const WORD: bool> Op for CallExtern1<LARGE, WORD> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let regs = m.regs();
        let a = regs.read(self.a);
        regs.take_mask(self.takes);
        let value = (self.f)(m.rt, a);
        m.regs().store::<LARGE, WORD>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallExtern2<const LARGE: bool> {
    pub dst: Off,
    pub a: Off,
    pub b: Off,
    pub takes: u64,
    pub f: Sync2,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for CallExtern2<LARGE> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        regs.take_mask(self.takes);
        let value = (self.f)(m.rt, a, b);
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallExtern3<const LARGE: bool> {
    pub dst: Off,
    pub a: Off,
    pub b: Off,
    pub c: Off,
    pub takes: u64,
    pub f: Sync3,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for CallExtern3<LARGE> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        let c = regs.read(self.c);
        regs.take_mask(self.takes);
        let value = (self.f)(m.rt, a, b, c);
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

/// A declaration of four or more parameters is called through its window.
pub struct CallWindow<const LARGE: bool> {
    pub dst: Off,
    pub window: ArgWindow,
    pub f: SyncWindow,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for CallWindow<LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let rt = m.rt;
        let value = (self.f)(rt, self.window.lend(m));
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

/// A slice return comes back in two registers and is boxed here (RFC-0047 §6).
/// The boxed `Elements` is a `Large`, which is why this call carries no `LARGE`
/// parameter.
pub struct CallSlice {
    pub dst: Off,
    pub a: Off,
    pub takes: u64,
    pub f: SyncSlice,
    pub next: Box<dyn Op>,
}

impl Op for CallSlice {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let regs = m.regs();
        let a = regs.read(self.a);
        regs.take_mask(self.takes);
        let run = (self.f)(m.rt, a);
        let value = erase_elements(m.rt, run);
        m.regs().define::<true>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallState0<const LARGE: bool> {
    pub dst: Off,
    pub state: State,
    pub f: State0,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for CallState0<LARGE> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let value = (self.f)(&*self.state, m.rt);
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallState1<const LARGE: bool> {
    pub dst: Off,
    pub a: Off,
    pub takes: u64,
    pub state: State,
    pub f: State1,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for CallState1<LARGE> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let regs = m.regs();
        let a = regs.read(self.a);
        regs.take_mask(self.takes);
        let value = (self.f)(&*self.state, m.rt, a);
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallState2<const LARGE: bool> {
    pub dst: Off,
    pub a: Off,
    pub b: Off,
    pub takes: u64,
    pub state: State,
    pub f: State2,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for CallState2<LARGE> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        regs.take_mask(self.takes);
        let value = (self.f)(&*self.state, m.rt, a, b);
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallState3<const LARGE: bool> {
    pub dst: Off,
    pub a: Off,
    pub b: Off,
    pub c: Off,
    pub takes: u64,
    pub state: State,
    pub f: State3,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for CallState3<LARGE> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        let c = regs.read(self.c);
        regs.take_mask(self.takes);
        let value = (self.f)(&*self.state, m.rt, a, b, c);
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallStateWindow<const LARGE: bool> {
    pub dst: Off,
    pub window: ArgWindow,
    pub state: State,
    pub f: StateWindow,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for CallStateWindow<LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let rt = m.rt;
        let value = (self.f)(&*self.state, rt, self.window.lend(m));
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallStateSlice {
    pub dst: Off,
    pub a: Off,
    pub takes: u64,
    pub state: State,
    pub f: StateSlice,
    pub next: Box<dyn Op>,
}

impl Op for CallStateSlice {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let regs = m.regs();
        let a = regs.read(self.a);
        regs.take_mask(self.takes);
        let run = (self.f)(&*self.state, m.rt, a);
        let value = erase_elements(m.rt, run);
        m.regs().define::<true>(self.dst, value);
        self.next.run(m, r0)
    }
}

/// The argument of a fused call that reads what the call before it produced
/// rather than a register (RFC-0044, stage 6).
pub const PREVIOUS: Off = Off::PREVIOUS;

/// One call of a fused run, at the shapes `prepare::FusableCall` admits:
/// three or fewer arguments including the held value, and a slice return.
pub enum Call {
    Nullary { f: Sync0 },
    Unary { f: Sync1, a: Off },
    Binary { f: Sync2, a: Off, b: Off },
    Slice { f: SyncSlice, a: Off },
}

impl Call {
    #[inline]
    fn invoke(&self, m: &mut Machine<'_>, held: &mut Value) -> Value {
        let rt = m.rt;
        match *self {
            Call::Nullary { f } => f(rt),
            Call::Unary { f, a } => {
                let a = arg(m, held, a);
                f(rt, a)
            }
            Call::Binary { f, a, b } => {
                let a = arg(m, held, a);
                let b = arg(m, held, b);
                f(rt, a, b)
            }
            Call::Slice { f, a } => {
                let a = arg(m, held, a);
                erase_elements(rt, f(rt, a))
            }
        }
    }
}

#[inline]
fn arg(m: &mut Machine<'_>, held: &mut Value, at: Off) -> Value {
    match at {
        PREVIOUS => std::mem::take(held),
        at => m.regs().read(at),
    }
}

/// RFC-0044, stage 6.
///
/// `CALLS` and `TAIL` are the run's shape, which `prepare` resolved like every
/// other static fact: a body of this instance holds no loop bound.
pub struct Fused<const CALLS: usize, const TAIL: bool, const LARGE: bool> {
    pub dst: Off,
    pub calls: SmallVec<[Call; 2]>,
    pub tail: Option<Deref>,
    pub takes: u64,
    pub next: Box<dyn Op>,
}

impl<const CALLS: usize, const TAIL: bool, const LARGE: bool> Op for Fused<CALLS, TAIL, LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        debug_assert_eq!(
            self.calls.len(),
            CALLS,
            "a fused instance of another length"
        );
        m.regs().take_mask(self.takes);
        let mut held = Value::UNDEF;
        for call in &self.calls[..CALLS] {
            held = call.invoke(m, &mut held);
        }
        let value = match TAIL {
            true => {
                let read = self.tail.expect("a fused instance with a tail holds none");
                read(&held)
            }
            false => held,
        };
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

/// The longest run `fused` holds an instance for.
pub const MAX_CALLS: usize = 3;

/// The instance a run of `calls.len()` calls, with or without a deref, runs
/// as. `prepare::FusedRegion` builds no other shape: a lone call with no deref
/// is not a run, and the recognizer stops at `MAX_CALLS`.
pub fn fused(
    large: bool,
    dst: Off,
    calls: SmallVec<[Call; 2]>,
    tail: Option<Deref>,
    takes: u64,
    next: Box<dyn Op>,
) -> Box<dyn Op> {
    match large {
        true => shape::<true>(dst, calls, tail, takes, next),
        false => shape::<false>(dst, calls, tail, takes, next),
    }
}

/// The two facts that pick a fused instance, read off the run `prepare` built.
struct Instance {
    calls: usize,
    tail: bool,
}

fn shape<const LARGE: bool>(
    dst: Off,
    calls: SmallVec<[Call; 2]>,
    tail: Option<Deref>,
    takes: u64,
    next: Box<dyn Op>,
) -> Box<dyn Op> {
    let instance = Instance {
        calls: calls.len(),
        tail: tail.is_some(),
    };
    match instance {
        Instance {
            calls: 1,
            tail: true,
        } => Box::new(Fused::<1, true, LARGE> {
            dst,
            calls,
            tail,
            takes,
            next,
        }),
        Instance {
            calls: 2,
            tail: true,
        } => Box::new(Fused::<2, true, LARGE> {
            dst,
            calls,
            tail,
            takes,
            next,
        }),
        Instance {
            calls: 3,
            tail: true,
        } => Box::new(Fused::<3, true, LARGE> {
            dst,
            calls,
            tail,
            takes,
            next,
        }),
        Instance {
            calls: 2,
            tail: false,
        } => Box::new(Fused::<2, false, LARGE> {
            dst,
            calls,
            tail,
            takes,
            next,
        }),
        Instance {
            calls: 3,
            tail: false,
        } => Box::new(Fused::<3, false, LARGE> {
            dst,
            calls,
            tail,
            takes,
            next,
        }),
        Instance {
            calls: 1,
            tail: false,
        } => {
            panic!("a lone call with no deref is not a fused run")
        }
        Instance { calls, .. } => panic!("a fused run of {calls} calls has no instance"),
    }
}

pub struct CallExternAsync<const LARGE: bool> {
    pub dst: Off,
    pub window: ArgWindow,
    pub f: Async,
    pub next: BlockId,
}

impl<const LARGE: bool> Op for CallExternAsync<LARGE> {
    fn run(&self, m: &mut Machine<'_>, _: u64) -> BlockId {
        let rt = m.rt.clone();
        let fut = (self.f)(rt, self.window.lend(m));
        m.suspend::<LARGE>(self.dst, self.next, fut);
        SUSPEND
    }
}

pub struct CallStateAsync<const LARGE: bool> {
    pub dst: Off,
    pub window: ArgWindow,
    pub state: State,
    pub f: AsyncState,
    pub next: BlockId,
}

impl<const LARGE: bool> Op for CallStateAsync<LARGE> {
    fn run(&self, m: &mut Machine<'_>, _: u64) -> BlockId {
        let rt = m.rt.clone();
        let state = Arc::clone(&self.state);
        let fut = (self.f)(state, rt, self.window.lend(m));
        m.suspend::<LARGE>(self.dst, self.next, fut);
        SUSPEND
    }
}

/// A `heavy` extern (RFC-0046): a Rust `fn`, but one worth another thread.
/// The work outlives this frame, so it owns its arguments as a spawn's does;
/// the call then awaits the handle, so the site suspends exactly as an
/// `async fn` extern's does.
pub struct CallHeavy<const LARGE: bool> {
    pub dst: Off,
    pub window: ArgWindow,
    pub f: SyncCall,
    pub next: BlockId,
}

impl<const LARGE: bool> Op for CallHeavy<LARGE> {
    fn run(&self, m: &mut Machine<'_>, _: u64) -> BlockId {
        let args = self.window.own(m);
        let rt = m.rt.clone();
        let f = self.f.clone();
        let executor = Arc::clone(&m.shared().executor);
        let handle = executor.spawn_blocking(Box::new(move || f.call_taking(&rt, &args)));
        m.suspend::<LARGE>(
            self.dst,
            self.next,
            Box::pin(async move { executor.eval(handle).await }),
        );
        SUSPEND
    }
}

/// A call into another body whose type says `Sync` (RFC-0046): the callee is
/// reached through the module table at run time, as it always was, and run to
/// its value on the window above this frame (RFC-0052 rule 7). Whether the
/// callee *may* suspend is not asked — the checker settled it.
pub struct CallDirect<const LARGE: bool, const WORD: bool> {
    pub dst: Off,
    pub callee: QualifiedRef,
    pub args: Box<[Off]>,
    pub takes: u64,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool, const WORD: bool> Op for CallDirect<LARGE, WORD> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let args = staged(m, &self.args, self.takes);
        let prepared = Arc::clone(lookup_module(m.shared(), &self.callee));
        let value = call_module_sync(m, &prepared, self.callee, args);
        m.regs().store::<LARGE, WORD>(self.dst, value);
        self.next.run(m, r0)
    }
}

/// The same call where the callee's task is above `Sync`: it hands the driver
/// a future and leaves the block, which is why this one is a terminator and
/// `CallDirect` is not.
pub struct CallDirectAsync<const LARGE: bool> {
    pub dst: Off,
    pub callee: QualifiedRef,
    pub args: Box<[Off]>,
    pub takes: u64,
    pub next: BlockId,
}

impl<const LARGE: bool> Op for CallDirectAsync<LARGE> {
    fn run(&self, m: &mut Machine<'_>, _: u64) -> BlockId {
        let args = staged(m, &self.args, self.takes);
        let shared = Arc::clone(m.shared());
        let page = Arc::clone(m.page);
        let fut = Box::pin(call_module(shared, page, self.callee, args));
        m.suspend::<LARGE>(self.dst, self.next, fut);
        SUSPEND
    }
}

/// How a call reaches the closure its callee register holds, which `prepare`
/// read off that register's type: `THROUGH` is a reference the caller keeps,
/// its absence a value this call consumes — which is why the value form takes
/// the register and `takes` covers the arguments alone.
///
/// # Safety
/// The type checker admits only a closure in the callee register, and only a
/// live reference to one under `THROUGH`: the closure's register is not
/// written during the call, and the machine holding it outlives the call.
#[inline(always)]
unsafe fn call_closure<const THROUGH: bool>(
    m: &mut Machine<'_>,
    callee: Off,
    args: &mut [Value],
) -> Value {
    match THROUGH {
        true => {
            let closure: &FnValue = unsafe {
                let target = m.regs().peek(callee).target();
                &*(target.as_fn() as *const FnValue)
            };
            m.call_fn_sync(closure, args)
        }
        false => {
            let closure = unsafe { m.regs().take::<true>(callee).materialize::<FnValue>() };
            m.call_fn_sync(&closure, args)
        }
    }
}

/// A closure call whose type says `Sync`: as `CallDirect`, run to its value
/// inside the block, with no test of the closure's own `Code`.
pub struct CallIndirect<const LARGE: bool, const WORD: bool, const THROUGH: bool> {
    pub dst: Off,
    pub callee: Off,
    pub args: Box<[Off]>,
    pub takes: u64,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool, const WORD: bool, const THROUGH: bool> Op
    for CallIndirect<LARGE, WORD, THROUGH>
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let mut args = staged(m, &self.args, self.takes);
        // SAFETY: as `call_closure` states.
        let value = unsafe { call_closure::<THROUGH>(m, self.callee, &mut args) };
        m.regs().store::<LARGE, WORD>(self.dst, value);
        self.next.run(m, r0)
    }
}

/// A closure call whose task is above `Sync`: the future goes to the driver.
pub struct CallIndirectAsync<const LARGE: bool, const THROUGH: bool> {
    pub dst: Off,
    pub callee: Off,
    pub args: Box<[Off]>,
    pub takes: u64,
    pub next: BlockId,
}

impl<const LARGE: bool, const THROUGH: bool> Op for CallIndirectAsync<LARGE, THROUGH> {
    fn run(&self, m: &mut Machine<'_>, _: u64) -> BlockId {
        let mut args = staged(m, &self.args, self.takes);
        let fut: BoxFuture<'static, Value> = match THROUGH {
            // SAFETY: the type checker admits only a live reference to a
            // closure here; the closure's register is not written during the
            // call, and the machine holding it outlives the future the driver
            // awaits.
            true => {
                let closure: &'static FnValue = unsafe {
                    let target = m.regs().peek(self.callee).target();
                    &*(target.as_fn() as *const FnValue)
                };
                Box::pin(fn_value_call(closure, &mut args))
            }
            // SAFETY: the type checker admits only a closure value here.
            false => {
                let closure =
                    unsafe { m.regs().take::<true>(self.callee).materialize::<FnValue>() };
                Box::pin(async move { fn_value_call(&closure, &mut args).await })
            }
        };
        m.suspend::<LARGE>(self.dst, self.next, fut);
        SUSPEND
    }
}

pub struct Eval<const LARGE: bool> {
    pub dst: Off,
    pub handle: Off,
    pub next: BlockId,
}

impl<const LARGE: bool> Op for Eval<LARGE> {
    fn run(&self, m: &mut Machine<'_>, _: u64) -> BlockId {
        // SAFETY: the type checker admits only a handle value here.
        let handle = unsafe {
            m.regs()
                .take::<true>(self.handle)
                .materialize::<HandleValue>()
        };
        let executor = Arc::clone(&m.shared().executor);
        m.suspend::<LARGE>(
            self.dst,
            self.next,
            Box::pin(async move { executor.eval(handle).await }),
        );
        SUSPEND
    }
}

/// A spawn's work outlives this frame, so it owns its arguments rather than
/// borrowing registers; it keeps the window at every arity, and the handler
/// takes each argument out of it.
pub struct SpawnExternSync {
    pub dst: Off,
    pub window: ArgWindow,
    pub f: SyncCall,
    pub next: Box<dyn Op>,
}

impl Op for SpawnExternSync {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let args = self.window.own(m);
        let rt = m.rt.clone();
        let f = self.f.clone();
        let handle = m
            .shared()
            .executor
            .spawn_blocking(Box::new(move || f.call_taking(&rt, &args)));
        m.regs().define::<true>(self.dst, Value::handle(handle));
        self.next.run(m, r0)
    }
}

pub struct SpawnExternAsync {
    pub dst: Off,
    pub window: ArgWindow,
    pub f: AsyncCall<AcvusRuntime>,
    pub next: Box<dyn Op>,
}

impl Op for SpawnExternAsync {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let mut args = self.window.own(m);
        let rt = m.rt.clone();
        let fut = match &self.f {
            AsyncCall::Plain(f) => f(rt, &mut args),
            AsyncCall::Stateful { state, f } => f(Arc::clone(state), rt, &mut args),
        };
        let handle = m.shared().executor.spawn_async(fut);
        m.regs().define::<true>(self.dst, Value::handle(handle));
        self.next.run(m, r0)
    }
}

pub struct SpawnModule {
    pub dst: Off,
    pub callee: QualifiedRef,
    pub args: Box<[Off]>,
    pub takes: u64,
    pub next: Box<dyn Op>,
}

impl Op for SpawnModule {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let args = staged(m, &self.args, self.takes);
        let child = crate::interpreter::Interpreter::spawned(
            Arc::clone(m.shared()),
            self.callee,
            Arc::clone(m.page),
            args,
        );
        let handle = m.shared().executor.spawn_interpreter(child);
        m.regs().define::<true>(self.dst, Value::handle(handle));
        self.next.run(m, r0)
    }
}

pub struct MakeClosure {
    pub dst: Off,
    pub entry: Arc<dyn crate::machine::Callable>,
    pub captures: Box<[Off]>,
    pub takes: u64,
    pub next: Box<dyn Op>,
}

impl Op for MakeClosure {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> BlockId {
        let captures: Arc<[Owned<AcvusRuntime>]> = {
            let regs = m.regs();
            let captures = self
                .captures
                .iter()
                .map(|slot| Owned::from_value(regs.read(*slot)))
                .collect();
            regs.take_mask(self.takes);
            captures
        };
        let closure = FnValue {
            shared: Arc::clone(m.shared()),
            page: Arc::clone(m.page),
            entry: Arc::clone(&self.entry),
            captures,
        };
        m.regs().define::<true>(self.dst, Value::closure(closure));
        self.next.run(m, r0)
    }
}
