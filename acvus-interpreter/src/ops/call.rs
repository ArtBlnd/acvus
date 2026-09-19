//! Calls: the extern instances, the named functions, the closures, the
//! spawns, and the closure constructor.
//!
//! A synchronous extern is an operation and runs inside the block. Everything
//! that can suspend — an asynchronous extern, a heavy extern, a call into
//! another body, an `Eval` — is a terminator: it hands a `'static` future to
//! the driver and leaves the block at `SUSPEND`.
//!
//! `prepare` reads a handler's `Width` once, picks the operation by the form
//! it names, and moves the handler in as a field; a `run` here calls through
//! it with no decision in between (RFC-0052 §6, RFC-0059 rule 7).

use std::sync::Arc;

use acvus_extern::{Owned, Words};
use acvus_mir::graph::QualifiedRef;
use futures::future::BoxFuture;
use smallvec::SmallVec;

use crate::code::{BlockId, Deref, Exit, Off, Op, SUSPEND, SlicePair, successor};
use crate::interpreter::lookup_module;
use crate::machine::{Machine, call_module, call_module_sync, fn_value_call};
use crate::runtime::AcvusRuntime;
use crate::value::{FnValue, HandleValue, Value};

/// The declared instance a call reaches, cloned out of the module table for
/// this site alone: a box's data pointer is the handler, so `run` loads it
/// and jumps. The operation asked `Handler::width` once, at preparation, and
/// calls the one form that answer names.
pub type Handler = Box<dyn acvus_extern::Handler<AcvusRuntime>>;
pub type AsyncHandler = Box<dyn acvus_extern::AsyncHandler<AcvusRuntime>>;

/// A call the driver runs hands its arguments to a future that outlives this
/// frame, so it owns them rather than lending registers.
#[inline]
fn staged(m: &mut Machine<'_>, slots: &[Off], takes: u64) -> Vec<Value> {
    let regs = m.regs();
    let args = slots.iter().map(|slot| regs.read(*slot)).collect();
    regs.take_mask(takes);
    args
}

/// One argument of a synchronous call into a body, moved into the register the
/// callee reads it from. `prepare` gives a body's parameters its first
/// registers, and those are the registers the window above this frame begins
/// with, so the callee enters with its arguments in place (RFC-0052 rule 7).
pub struct LayArg {
    pub at: Off,
    pub src: Off,
    pub next: Box<dyn Op>,
}

impl Op for LayArg {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let value = regs.read(self.src);
        regs.lay(self.at, value);
        self.next.run(m, r0)
    }
}

/// A slice argument of the same call: two adjacent registers (RFC-0047
/// amended, rule 4).
pub struct LayPair {
    pub at: SlicePair,
    pub src: SlicePair,
    pub next: Box<dyn Op>,
}

impl Op for LayPair {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        m.regs().lay_pair(self.at, self.src);
        self.next.run(m, r0)
    }
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
    pub f: Handler,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for CallExtern0<LARGE> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        // SAFETY: `prepare` read this handler's width and built this
        // operation for the form it named.
        let value = unsafe { self.f.call0(m.rt) };
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallExtern1<const LARGE: bool, const WORD: bool> {
    pub dst: Off,
    pub a: Off,
    pub takes: u64,
    pub f: Handler,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool, const WORD: bool> Op for CallExtern1<LARGE, WORD> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        regs.take_mask(self.takes);
        // SAFETY: as `CallExtern0`'s, at one argument.
        let value = unsafe { self.f.call1(m.rt, a) };
        m.regs().store::<LARGE, WORD>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallExtern2<const LARGE: bool> {
    pub dst: Off,
    pub a: Off,
    pub b: Off,
    pub takes: u64,
    pub f: Handler,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for CallExtern2<LARGE> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        regs.take_mask(self.takes);
        // SAFETY: as `CallExtern0`'s, at two arguments.
        let value = unsafe { self.f.call2(m.rt, a, b) };
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
    pub f: Handler,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for CallExtern3<LARGE> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        let c = regs.read(self.c);
        regs.take_mask(self.takes);
        // SAFETY: as `CallExtern0`'s, at three arguments.
        let value = unsafe { self.f.call3(m.rt, a, b, c) };
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

/// A declaration whose arguments are wider than the register forms is called
/// through its window.
pub struct CallWindow<const LARGE: bool> {
    pub dst: Off,
    pub window: ArgWindow,
    pub f: Handler,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool> Op for CallWindow<LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let rt = m.rt;
        // SAFETY: as `CallExtern0`'s; the run is the window `prepare` laid.
        let value = unsafe { self.f.call_run(rt, self.window.lend(m)) };
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

/// `AsSlice` (RFC-0047 amended): the handler's result is two values wide and
/// this stores their words to `dst`, the first register of the pair
/// `prepare::assign_slots` gave the slice. A slice is no `Value`, so there is
/// no `LARGE` here and no drop anywhere.
pub struct AsSlice {
    pub dst: SlicePair,
    pub a: Off,
    pub takes: u64,
    pub f: Handler,
    pub next: Box<dyn Op>,
}

impl Op for AsSlice {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        regs.take_mask(self.takes);
        // SAFETY: as `CallExtern0`'s; `prepare` admits only a handler whose
        // result is the two values a slice is.
        let Words { ptr, len } = unsafe { self.f.call_slice(m.rt, a) }.words();
        let regs = m.regs();
        regs.set_word(self.dst.ptr, ptr);
        regs.set_word(self.dst.len, len);
        self.next.run(m, r0)
    }
}

/// The argument of a fused call that reads what the call before it produced
/// rather than a register (RFC-0044, stage 6).
pub const PREVIOUS: Off = Off::PREVIOUS;

/// One call of a fused run, at the shapes `prepare::FusableCall` admits:
/// three or fewer arguments including the held value. A slice-returning
/// handler is not among them — its result is a register pair, and a run
/// hands one `Value` from each call to the next.
pub enum Call {
    Nullary { f: Handler },
    Unary { f: Handler, a: Off },
    Binary { f: Handler, a: Off, b: Off },
}

impl Call {
    #[inline]
    fn invoke(&self, m: &mut Machine<'_>, held: &mut Value) -> Value {
        let rt = m.rt;
        // SAFETY: `prepare::fusable_call` admits a call into this run only at
        // the form each arm names.
        match self {
            Call::Nullary { f } => unsafe { f.call0(rt) },
            Call::Unary { f, a } => {
                let a = arg(m, held, *a);
                unsafe { f.call1(rt, a) }
            }
            Call::Binary { f, a, b } => {
                let a = arg(m, held, *a);
                let b = arg(m, held, *b);
                unsafe { f.call2(rt, a, b) }
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

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
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
    pub f: AsyncHandler,
    pub next: BlockId,
}

impl<const LARGE: bool> Op for CallExternAsync<LARGE> {
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        let rt = m.rt.clone();
        // SAFETY: `prepare` built this operation from this handler's width,
        // and the future owns the arguments it is given.
        let fut = unsafe { self.f.call(rt, self.window.lend(m)) };
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
    pub f: Handler,
    pub next: BlockId,
}

impl<const LARGE: bool> Op for CallHeavy<LARGE> {
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        let args = self.window.own(m);
        let rt = m.rt.clone();
        let f = self.f.clone();
        let executor = Arc::clone(&m.shared().executor);
        // SAFETY: the window is this call's whole argument run, owned by the
        // closure the pool runs.
        let handle = executor.spawn_blocking(Box::new(move || unsafe { f.call_run(&rt, &args) }));
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
    pub arity: u16,
    pub takes: u64,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool, const WORD: bool> Op for CallDirect<LARGE, WORD> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        m.regs().take_mask(self.takes);
        let prepared = Arc::clone(lookup_module(m.shared(), &self.callee));
        let value = call_module_sync(m, &prepared, self.callee, self.arity);
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
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
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
unsafe fn call_closure<const THROUGH: bool>(m: &mut Machine<'_>, callee: Off, arity: u16) -> Value {
    match THROUGH {
        true => {
            let closure: &FnValue = unsafe {
                let target = m.regs().peek(callee).target();
                &*(target.as_fn() as *const FnValue)
            };
            m.call_fn_sync(closure, arity)
        }
        false => {
            let closure = unsafe { m.regs().take::<true>(callee).materialize::<FnValue>() };
            m.call_fn_sync(&closure, arity)
        }
    }
}

/// A closure call whose type says `Sync`: as `CallDirect`, run to its value
/// inside the block, with no test of the closure's own `Code`.
pub struct CallIndirect<const LARGE: bool, const WORD: bool, const THROUGH: bool> {
    pub dst: Off,
    pub callee: Off,
    pub arity: u16,
    pub takes: u64,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool, const WORD: bool, const THROUGH: bool> Op
    for CallIndirect<LARGE, WORD, THROUGH>
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        m.regs().take_mask(self.takes);
        // SAFETY: as `call_closure` states.
        let value = unsafe { call_closure::<THROUGH>(m, self.callee, self.arity) };
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
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
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
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
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
    pub f: Handler,
    pub next: Box<dyn Op>,
}

impl Op for SpawnExternSync {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let args = self.window.own(m);
        let rt = m.rt.clone();
        let f = self.f.clone();
        // SAFETY: as `CallHeavy`'s.
        let handle = m
            .shared()
            .executor
            .spawn_blocking(Box::new(move || unsafe { f.call_run(&rt, &args) }));
        m.regs().define::<true>(self.dst, Value::handle(handle));
        self.next.run(m, r0)
    }
}

pub struct SpawnExternAsync {
    pub dst: Off,
    pub window: ArgWindow,
    pub f: AsyncHandler,
    pub next: Box<dyn Op>,
}

impl Op for SpawnExternAsync {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let args = self.window.own(m);
        let rt = m.rt.clone();
        // SAFETY: as `CallExternAsync`'s; the spawned future owns `args`.
        let fut = unsafe { self.f.call(rt, &args) };
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

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
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

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
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
