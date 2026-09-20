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

use acvus_extern::{Ctx, ObjectShape, Owned, Release, Runtime, Words};
use acvus_mir::graph::QualifiedRef;
use futures::future::BoxFuture;
use smallvec::SmallVec;

use crate::code::{BlockId, CodeRef, Deref, Exit, Marked, Off, Op, SUSPEND, SlicePair, successor};
use crate::interpreter::lookup_module;
use crate::machine::{
    Lent, LentCall, LentOut, Machine, call_module, call_module_sync, fn_value_call,
};
use crate::runtime::AcvusRuntime;
use crate::value::{FnValue, HandleValue, Value};

/// The declared instance a call reaches, cloned out of the module table for
/// this site alone. `prepare` asked `HandlerFactory::width` once and hands
/// the factory the shape it decided; the factory builds the operation, which
/// holds the handler by value and calls its body statically.
pub type Handler = Box<dyn acvus_extern::HandlerFactory<AcvusRuntime>>;

/// The same instance with its site table filled from the settled types of
/// this site's arguments, which is what builds the operation.
pub type Sited = Box<dyn acvus_extern::AtSite<AcvusRuntime>>;

/// Where a synchronous call site's result goes, where its arguments are, and
/// what runs after it — everything `prepare` settles before the handler's
/// type is known (RFC-0059 rule 4 amended).
pub enum CallShape {
    Registers0 {
        dst: Marked,
        large: bool,
        next: Box<dyn Op>,
    },
    Registers1 {
        dst: Marked,
        a: Off,
        takes: u64,
        large: bool,
        word: bool,
        next: Box<dyn Op>,
    },
    Registers2 {
        dst: Marked,
        a: Off,
        b: Off,
        takes: u64,
        large: bool,
        next: Box<dyn Op>,
    },
    Registers3 {
        dst: Marked,
        a: Off,
        b: Off,
        c: Off,
        takes: u64,
        large: bool,
        next: Box<dyn Op>,
    },
    Registers4 {
        dst: Marked,
        a: Off,
        b: Off,
        c: Off,
        d: Off,
        takes: u64,
        large: bool,
        next: Box<dyn Op>,
    },
    Window {
        dst: Marked,
        window: ArgWindow,
        large: bool,
        next: Box<dyn Op>,
    },
    Pair1 {
        dst: SlicePair,
        a: Off,
        takes: u64,
        next: Box<dyn Op>,
    },
    Pair2 {
        dst: SlicePair,
        a: Off,
        b: Off,
        takes: u64,
        next: Box<dyn Op>,
    },
    Pair3 {
        dst: SlicePair,
        a: Off,
        b: Off,
        c: Off,
        takes: u64,
        next: Box<dyn Op>,
    },
    Pair4 {
        dst: SlicePair,
        a: Off,
        b: Off,
        c: Off,
        d: Off,
        takes: u64,
        next: Box<dyn Op>,
    },
    PairWindow {
        dst: SlicePair,
        window: ArgWindow,
        next: Box<dyn Op>,
    },
    Run0 {
        dst: RunDest,
        next: Box<dyn Op>,
    },
    Run1 {
        dst: RunDest,
        a: Off,
        takes: u64,
        next: Box<dyn Op>,
    },
    Run2 {
        dst: RunDest,
        a: Off,
        b: Off,
        takes: u64,
        next: Box<dyn Op>,
    },
    Run3 {
        dst: RunDest,
        a: Off,
        b: Off,
        c: Off,
        takes: u64,
        next: Box<dyn Op>,
    },
    Run4 {
        dst: RunDest,
        a: Off,
        b: Off,
        c: Off,
        d: Off,
        takes: u64,
        next: Box<dyn Op>,
    },
    RunWindow {
        dst: RunDest,
        window: ArgWindow,
        next: Box<dyn Op>,
    },
    Heavy {
        dst: Marked,
        window: ArgWindow,
        large: bool,
        resume: BlockId,
    },
    Spawn {
        dst: Marked,
        window: ArgWindow,
        next: Box<dyn Op>,
    },
}

/// As `CallShape`, for the sites whose handler is an `async fn`.
pub enum AsyncShape {
    Await {
        dst: Marked,
        window: ArgWindow,
        large: bool,
        resume: BlockId,
    },
    Spawn {
        dst: Marked,
        window: ArgWindow,
        next: Box<dyn Op>,
    },
}

/// Where one call of a fused run reads its arguments (RFC-0044 stage 6).
pub enum FusedShape {
    Nullary,
    Unary { a: Off },
    Binary { a: Off, b: Off },
}

/// The shapes a form does not take, which `prepare`'s reading of the
/// handler's width keeps it from building.
fn not_this_form(form: &str) -> ! {
    panic!("a handler of {form} was given a call shape of another form")
}

/// The window, and the two ways a call leaves this thread. None of the three
/// depends on the register form, and the thread-crossing pair holds the
/// handler behind a `dyn`: the value is sent, not called (RFC-0059 rule 3
/// amended).
pub fn off_the_register_forms<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Window {
            dst,
            window,
            large,
            next,
        } => match large {
            true => Box::new(CallWindow::<H, true> {
                dst,
                window,
                f,
                next,
            }),
            false => Box::new(CallWindow::<H, false> {
                dst,
                window,
                f,
                next,
            }),
        },
        CallShape::Heavy {
            dst,
            window,
            large,
            resume,
        } => {
            let f: Arc<dyn SentCall> = Arc::new(f);
            match large {
                true => Box::new(CallHeavy::<true> {
                    dst,
                    window,
                    f,
                    next: resume,
                }),
                false => Box::new(CallHeavy::<false> {
                    dst,
                    window,
                    f,
                    next: resume,
                }),
            }
        }
        CallShape::Spawn { dst, window, next } => Box::new(SpawnExternSync {
            dst,
            window,
            f: Arc::new(f),
            next,
        }),
        _ => not_this_form("this arity"),
    }
}

pub fn op_no_argument<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Registers0 { dst, large, next } => match large {
            true => Box::new(CallExtern0::<H, true> { dst, f, next }),
            false => Box::new(CallExtern0::<H, false> { dst, f, next }),
        },
        shape => off_the_register_forms(f, shape),
    }
}

pub fn op_one_argument<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Registers1 {
            dst,
            a,
            takes,
            large,
            word,
            next,
        } => match large {
            true => Box::new(CallExtern1::<H, true, false> {
                dst,
                a,
                takes,
                f,
                next,
            }),
            false => match word {
                true => Box::new(CallExtern1::<H, false, true> {
                    dst,
                    a,
                    takes,
                    f,
                    next,
                }),
                false => Box::new(CallExtern1::<H, false, false> {
                    dst,
                    a,
                    takes,
                    f,
                    next,
                }),
            },
        },
        shape => off_the_register_forms(f, shape),
    }
}

pub fn op_two_arguments<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Registers2 {
            dst,
            a,
            b,
            takes,
            large,
            next,
        } => match large {
            true => Box::new(CallExtern2::<H, true> {
                dst,
                a,
                b,
                takes,
                f,
                next,
            }),
            false => Box::new(CallExtern2::<H, false> {
                dst,
                a,
                b,
                takes,
                f,
                next,
            }),
        },
        shape => off_the_register_forms(f, shape),
    }
}

pub fn op_three_arguments<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Registers3 {
            dst,
            a,
            b,
            c,
            takes,
            large,
            next,
        } => match large {
            true => Box::new(CallExtern3::<H, true> {
                dst,
                a,
                b,
                c,
                takes,
                f,
                next,
            }),
            false => Box::new(CallExtern3::<H, false> {
                dst,
                a,
                b,
                c,
                takes,
                f,
                next,
            }),
        },
        shape => off_the_register_forms(f, shape),
    }
}

pub fn op_four_arguments<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Registers4 {
            dst,
            a,
            b,
            c,
            d,
            takes,
            large,
            next,
        } => match large {
            true => Box::new(CallExtern4::<H, true> {
                dst,
                a,
                b,
                c,
                d,
                takes,
                f,
                next,
            }),
            false => Box::new(CallExtern4::<H, false> {
                dst,
                a,
                b,
                c,
                d,
                takes,
                f,
                next,
            }),
        },
        shape => off_the_register_forms(f, shape),
    }
}

/// This family has no `Heavy` and no `Spawn` arm, which is a decision and not
/// an omission: a result two values wide is a borrow of the frame the call
/// laid its arguments on, and both of those tasks resume after that frame is
/// gone. `ExternHandler::heavy` takes `impl ValuesOnly<R>` and
/// `AsyncCall::WIDTH` fixes `ret: 1`, so no such handler can reach here.
fn off_the_pair_register_forms<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    let CallShape::PairWindow { dst, window, next } = shape else {
        not_this_form("a result two values wide at this arity")
    };
    Box::new(CallPairWindow::<H> {
        dst,
        window,
        f,
        next,
    })
}

pub fn op_pair_one_argument<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Pair1 {
            dst,
            a,
            takes,
            next,
        } => Box::new(CallPair1::<H> {
            dst,
            a,
            takes,
            f,
            next,
        }),
        shape => off_the_pair_register_forms(f, shape),
    }
}

pub fn op_pair_two_arguments<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Pair2 {
            dst,
            a,
            b,
            takes,
            next,
        } => Box::new(CallPair2::<H> {
            dst,
            a,
            b,
            takes,
            f,
            next,
        }),
        shape => off_the_pair_register_forms(f, shape),
    }
}

pub fn op_pair_three_arguments<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Pair3 {
            dst,
            a,
            b,
            c,
            takes,
            next,
        } => Box::new(CallPair3::<H> {
            dst,
            a,
            b,
            c,
            takes,
            f,
            next,
        }),
        shape => off_the_pair_register_forms(f, shape),
    }
}

pub fn op_pair_four_arguments<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Pair4 {
            dst,
            a,
            b,
            c,
            d,
            takes,
            next,
        } => Box::new(CallPair4::<H> {
            dst,
            a,
            b,
            c,
            d,
            takes,
            f,
            next,
        }),
        shape => off_the_pair_register_forms(f, shape),
    }
}

pub fn op_pair_wide<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    off_the_pair_register_forms(f, shape)
}

/// The aggregate family's fallback arm. It has no `Heavy` and no `Spawn` arm
/// for the reason the pair family has none: both tasks resume after the frame
/// the destination run lives in is gone, and `ExternHandler::heavy` takes
/// `impl ValuesOnly<R>` while `AsyncCall::WIDTH` fixes `ret: 1`.
fn off_the_run_register_forms<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    let CallShape::RunWindow { dst, window, next } = shape else {
        not_this_form("an aggregate result at this arity")
    };
    Box::new(CallRunWindow::<H> {
        dst,
        window,
        f,
        next,
    })
}

pub fn op_run_no_argument<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Run0 { dst, next } => Box::new(CallRun0::<H> { dst, f, next }),
        shape => off_the_run_register_forms(f, shape),
    }
}

pub fn op_run_one_argument<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Run1 {
            dst,
            a,
            takes,
            next,
        } => Box::new(CallRun1::<H> {
            dst,
            a,
            takes,
            f,
            next,
        }),
        shape => off_the_run_register_forms(f, shape),
    }
}

pub fn op_run_two_arguments<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Run2 {
            dst,
            a,
            b,
            takes,
            next,
        } => Box::new(CallRun2::<H> {
            dst,
            a,
            b,
            takes,
            f,
            next,
        }),
        shape => off_the_run_register_forms(f, shape),
    }
}

pub fn op_run_three_arguments<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Run3 {
            dst,
            a,
            b,
            c,
            takes,
            next,
        } => Box::new(CallRun3::<H> {
            dst,
            a,
            b,
            c,
            takes,
            f,
            next,
        }),
        shape => off_the_run_register_forms(f, shape),
    }
}

pub fn op_run_four_arguments<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    match shape {
        CallShape::Run4 {
            dst,
            a,
            b,
            c,
            d,
            takes,
            next,
        } => Box::new(CallRun4::<H> {
            dst,
            a,
            b,
            c,
            d,
            takes,
            f,
            next,
        }),
        shape => off_the_run_register_forms(f, shape),
    }
}

pub fn op_run_wide<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    off_the_run_register_forms(f, shape)
}

pub fn fused_no_argument<H>(f: H, shape: FusedShape) -> Call
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    let FusedShape::Nullary = shape else {
        not_this_form("no argument")
    };
    Box::new(Nullary::<H> { f })
}

pub fn fused_one_argument<H>(f: H, shape: FusedShape) -> Call
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    let FusedShape::Unary { a } = shape else {
        not_this_form("one argument")
    };
    Box::new(Unary::<H> { f, a })
}

pub fn fused_two_arguments<H>(f: H, shape: FusedShape) -> Call
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    let FusedShape::Binary { a, b } = shape else {
        not_this_form("two arguments")
    };
    Box::new(Binary::<H> { f, a, b })
}

pub fn async_extern_op<H>(f: H, shape: AsyncShape) -> Box<dyn Op>
where
    H: acvus_extern::AsyncCall<AcvusRuntime>,
{
    let f: Arc<dyn SentAsync> = Arc::new(f);
    match shape {
        AsyncShape::Await {
            dst,
            window,
            large,
            resume,
        } => match large {
            true => Box::new(CallExternAsync::<true> {
                dst,
                window,
                f,
                next: resume,
            }),
            false => Box::new(CallExternAsync::<false> {
                dst,
                window,
                f,
                next: resume,
            }),
        },
        AsyncShape::Spawn { dst, window, next } => Box::new(SpawnExternAsync {
            dst,
            window,
            f,
            next,
        }),
    }
}

/// The future of an `async fn` handler is boxed at the call (RFC-0059 rule
/// 6), so the handler behind it is reached once per call through a `dyn` and
/// the box is what the driver holds.
pub trait SentAsync: Send + Sync {
    /// # Safety
    /// As `AsyncCall::call`: `run` is the call's whole argument run and the
    /// future owns it.
    unsafe fn call_async(&self, rt: AcvusRuntime, run: &[Value]) -> BoxFuture<'static, Value>;
}

impl<H> SentAsync for H
where
    H: acvus_extern::AsyncCall<AcvusRuntime>,
{
    unsafe fn call_async(&self, rt: AcvusRuntime, run: &[Value]) -> BoxFuture<'static, Value> {
        // SAFETY: the caller's contract, which is `AsyncCall::call`'s.
        unsafe { self.call(rt, run) }
    }
}

/// A handler whose call crosses a thread: the value is sent to the pool and
/// called there, so a `Heavy` call and a spawn keep the `dyn` a `Sync` call
/// no longer has (RFC-0059 rule 4 amended).
pub trait SentCall: Send + Sync {
    /// # Safety
    /// `run` is the call's whole argument run, owned by the caller of this
    /// method for as long as the call, and `ctx`'s frame is a window bound to
    /// no other frame.
    unsafe fn call_owned(&self, ctx: &mut Ctx<'_, AcvusRuntime>, run: &[Value]) -> Value;
}

impl<H> SentCall for H
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    unsafe fn call_owned(&self, ctx: &mut Ctx<'_, AcvusRuntime>, run: &[Value]) -> Value {
        // SAFETY: the caller's contract, which is `call_run`'s.
        unsafe { self.call_run(ctx, run) }
    }
}

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
        let value = m.regs().read(self.src);
        m.window().lay(self.at, value);
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
        let regs = m.regs();
        let ptr = regs.read(self.src.ptr);
        let len = regs.read(self.src.len);
        let window = m.window();
        window.lay(self.at.ptr, ptr);
        window.lay(self.at.len, len);
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
    /// frame moves on, with the window it calls a closure in (RFC-0044,
    /// stage 2b; RFC-0050 rule 6).
    #[inline]
    fn lend<'r, 'c>(&self, m: &'r mut Machine<'c>) -> Lent<'r, 'c> {
        m.regs().take_mask(self.takes);
        m.lend_and_window(self.at, self.arity)
    }

    /// The arguments owned, for work that outlives this frame.
    fn own(&self, m: &mut Machine<'_>) -> Vec<Value> {
        let regs = m.regs();
        regs.take_mask(self.takes);
        regs.run_of(self.at, self.arity).to_vec()
    }
}

pub struct CallExtern0<H, const LARGE: bool> {
    pub dst: Marked,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H, const LARGE: bool> Op for CallExtern0<H, LARGE>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        // SAFETY: `prepare` read this handler's width and built this
        // operation for the form it named.
        let value = unsafe { self.f.call0(&mut m.ctx) };
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallExtern1<H, const LARGE: bool, const WORD: bool> {
    pub dst: Marked,
    pub a: Off,
    pub takes: u64,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H, const LARGE: bool, const WORD: bool> Op for CallExtern1<H, LARGE, WORD>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        regs.take_mask(self.takes);
        // SAFETY: as `CallExtern0`'s, at one argument.
        let value = unsafe { self.f.call1(&mut m.ctx, a) };
        m.regs().store::<LARGE, WORD>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallExtern2<H, const LARGE: bool> {
    pub dst: Marked,
    pub a: Off,
    pub b: Off,
    pub takes: u64,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H, const LARGE: bool> Op for CallExtern2<H, LARGE>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        regs.take_mask(self.takes);
        // SAFETY: as `CallExtern0`'s, at two arguments.
        let value = unsafe { self.f.call2(&mut m.ctx, a, b) };
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallExtern3<H, const LARGE: bool> {
    pub dst: Marked,
    pub a: Off,
    pub b: Off,
    pub c: Off,
    pub takes: u64,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H, const LARGE: bool> Op for CallExtern3<H, LARGE>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        let c = regs.read(self.c);
        regs.take_mask(self.takes);
        // SAFETY: as `CallExtern0`'s, at three arguments.
        let value = unsafe { self.f.call3(&mut m.ctx, a, b, c) };
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct CallExtern4<H, const LARGE: bool> {
    pub dst: Marked,
    pub a: Off,
    pub b: Off,
    pub c: Off,
    pub d: Off,
    pub takes: u64,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H, const LARGE: bool> Op for CallExtern4<H, LARGE>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        let c = regs.read(self.c);
        let d = regs.read(self.d);
        regs.take_mask(self.takes);
        // SAFETY: as `CallExtern0`'s, at four arguments.
        let value = unsafe { self.f.call4(&mut m.ctx, a, b, c, d) };
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

/// A register form is read once per call off the operation itself, so the
/// widest one stays inside a cache line. At a zero-sized handler — which is
/// every declaration in the workspace — `CallExtern1` through `CallExtern4`
/// all measure 48 bytes, the `Off`s fitting in the padding `Marked` and the
/// `u64` mask leave, so this bound is not what fixed the cut at four;
/// `acvus_extern::REGISTER_FORM`'s own note says what did. It is asserted
/// because a host whose handler carries state, or a wider `Off`, would make it
/// binding.
const CACHE_LINE: usize = 64;
const _: () = assert!(
    size_of::<CallExtern4<(), false>>() <= CACHE_LINE,
    "the widest register form is past a cache line: lower acvus_extern::REGISTER_FORM"
);
const _: () = assert!(
    acvus_extern::REGISTER_FORM == 4,
    "REGISTER_FORM names the widest CallExtern this module defines"
);

/// A declaration whose arguments are wider than the register forms is called
/// through its window.
pub struct CallWindow<H, const LARGE: bool> {
    pub dst: Marked,
    pub window: ArgWindow,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H, const LARGE: bool> Op for CallWindow<H, LARGE>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let Lent { run, ctx } = self.window.lend(m);
        // SAFETY: as `CallExtern0`'s; the run is the window `prepare` laid.
        let value = unsafe { self.f.call_run(ctx, run) };
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

/// The pair forms (RFC-0047 amended rule 2, RFC-0062 Decision 4): the
/// handler's result is two of the runtime's values and this stores their
/// words to the two adjacent registers `prepare::assign_slots` gave the
/// call's result. Neither register holds a `Value`, so there is no `LARGE`
/// here and no drop anywhere.
#[inline]
fn land_pair(m: &mut Machine<'_>, dst: SlicePair, out: [Value; 2]) {
    // SAFETY: `out` is what the handler's `Ret::into_run` wrote at
    // `Form = Pair`, which is `slice_into_run`'s own output.
    let words = unsafe { m.ctx.rt.slice_from_run(&out) };
    land_words(m, dst, words);
}

#[inline]
fn land_words(m: &mut Machine<'_>, dst: SlicePair, words: Words) {
    let regs = m.regs();
    regs.set_word(dst.ptr, words.ptr);
    regs.set_word(dst.len, words.len);
}

pub struct CallPair1<H> {
    pub dst: SlicePair,
    pub a: Off,
    pub takes: u64,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H> Op for CallPair1<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        regs.take_mask(self.takes);
        // SAFETY: `prepare` read this handler's width and built this
        // operation for the form it named.
        let out = unsafe { self.f.call_pair1(&mut m.ctx, a) };
        land_pair(m, self.dst, out);
        self.next.run(m, r0)
    }
}

pub struct CallPair2<H> {
    pub dst: SlicePair,
    pub a: Off,
    pub b: Off,
    pub takes: u64,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H> Op for CallPair2<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        regs.take_mask(self.takes);
        // SAFETY: as `CallPair1`'s, at two arguments.
        let out = unsafe { self.f.call_pair2(&mut m.ctx, a, b) };
        land_pair(m, self.dst, out);
        self.next.run(m, r0)
    }
}

pub struct CallPair3<H> {
    pub dst: SlicePair,
    pub a: Off,
    pub b: Off,
    pub c: Off,
    pub takes: u64,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H> Op for CallPair3<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        let c = regs.read(self.c);
        regs.take_mask(self.takes);
        // SAFETY: as `CallPair1`'s, at three arguments.
        let out = unsafe { self.f.call_pair3(&mut m.ctx, a, b, c) };
        land_pair(m, self.dst, out);
        self.next.run(m, r0)
    }
}

pub struct CallPair4<H> {
    pub dst: SlicePair,
    pub a: Off,
    pub b: Off,
    pub c: Off,
    pub d: Off,
    pub takes: u64,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H> Op for CallPair4<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        let c = regs.read(self.c);
        let d = regs.read(self.d);
        regs.take_mask(self.takes);
        // SAFETY: as `CallPair1`'s, at four arguments.
        let out = unsafe { self.f.call_pair4(&mut m.ctx, a, b, c, d) };
        land_pair(m, self.dst, out);
        self.next.run(m, r0)
    }
}

/// The registers `prepare/runs.rs` placed for an aggregate that stays in its
/// frame (RFC-0050 rules 2 and 3).
pub struct RunAt {
    pub at: Off,
    pub width: u16,
    /// The registers of the run whose value owns a `Large`, which
    /// `prepare::runs::Layout::releases` names. The frame drops its claim on
    /// each before lending the run and takes it again after.
    pub releases: Box<[Marked]>,
}

/// Where an aggregate-returning call writes the components of its result. A
/// handler writes the same run either way, which is why one operation family
/// serves both (RFC-0050 rule 6).
pub enum RunDest {
    Frame(RunAt),
    /// Rule 4's realization, for a result that outlives the frame: the flat
    /// body of the heap object the result becomes is the run.
    Heap {
        dst: Marked,
        shape: Arc<ObjectShape>,
        width: usize,
    },
}

/// The body every register form of the aggregate family shares: the
/// destination run is lent, the handler writes its components, and the frame
/// takes its claim on the `Large`s that landed.
#[inline]
fn land_run<F>(m: &mut Machine<'_>, dst: &RunDest, call: F)
where
    F: FnOnce(&mut Ctx<'_, AcvusRuntime>, &mut [Value]),
{
    match dst {
        RunDest::Frame(run) => {
            let regs = m.regs();
            for at in &run.releases {
                regs.assign::<false>(*at, Value::UNDEF);
            }
            let LentOut { out, ctx } = m.lend_out_and_window(run.at, run.width);
            call(ctx, out);
            let regs = m.regs();
            for at in &run.releases {
                regs.claim(*at);
            }
        }
        RunDest::Heap { dst, shape, width } => {
            let mut values: Box<[Owned<AcvusRuntime>]> =
                (0..*width).map(|_| Owned::default()).collect();
            // SAFETY: every slot is `Owned::default()`, which owns nothing, and
            // the handler writes each at most once.
            let out = unsafe { acvus_extern::lend_run(&mut values) };
            call(&mut m.ctx, out);
            let object = Value::object(Arc::clone(shape), values);
            m.regs().define::<true>(*dst, object);
        }
    }
}

pub struct CallRun0<H> {
    pub dst: RunDest,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H> Op for CallRun0<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        // SAFETY: `prepare` read this handler's width and built this
        // operation for the form it named; `land_run` lends a run of
        // `WIDTH.ret` values the caller owns.
        land_run(m, &self.dst, |ctx, out| unsafe {
            self.f.call_out0(ctx, out)
        });
        self.next.run(m, r0)
    }
}

pub struct CallRun1<H> {
    pub dst: RunDest,
    pub a: Off,
    pub takes: u64,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H> Op for CallRun1<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        regs.take_mask(self.takes);
        // SAFETY: as `CallRun0`'s, at one argument.
        land_run(m, &self.dst, |ctx, out| unsafe {
            self.f.call_out1(ctx, a, out)
        });
        self.next.run(m, r0)
    }
}

pub struct CallRun2<H> {
    pub dst: RunDest,
    pub a: Off,
    pub b: Off,
    pub takes: u64,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H> Op for CallRun2<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        regs.take_mask(self.takes);
        // SAFETY: as `CallRun0`'s, at two arguments.
        land_run(m, &self.dst, |ctx, out| unsafe {
            self.f.call_out2(ctx, a, b, out)
        });
        self.next.run(m, r0)
    }
}

pub struct CallRun3<H> {
    pub dst: RunDest,
    pub a: Off,
    pub b: Off,
    pub c: Off,
    pub takes: u64,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H> Op for CallRun3<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        let c = regs.read(self.c);
        regs.take_mask(self.takes);
        // SAFETY: as `CallRun0`'s, at three arguments.
        land_run(m, &self.dst, |ctx, out| unsafe {
            self.f.call_out3(ctx, a, b, c, out)
        });
        self.next.run(m, r0)
    }
}

pub struct CallRun4<H> {
    pub dst: RunDest,
    pub a: Off,
    pub b: Off,
    pub c: Off,
    pub d: Off,
    pub takes: u64,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H> Op for CallRun4<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        let c = regs.read(self.c);
        let d = regs.read(self.d);
        regs.take_mask(self.takes);
        // SAFETY: as `CallRun0`'s, at four arguments.
        land_run(m, &self.dst, |ctx, out| unsafe {
            self.f.call_out4(ctx, a, b, c, d, out)
        });
        self.next.run(m, r0)
    }
}

/// The window form. It does not go through `land_run`, because the argument
/// run and the destination run are both this frame's registers and the two
/// borrows have to be split.
pub struct CallRunWindow<H> {
    pub dst: RunDest,
    pub window: ArgWindow,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H> Op for CallRunWindow<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        m.regs().take_mask(self.window.takes);
        match &self.dst {
            RunDest::Frame(run) => {
                let regs = m.regs();
                for at in &run.releases {
                    regs.assign::<false>(*at, Value::UNDEF);
                }
                // SAFETY: `Prepare::call_into_run` asserts the destination run
                // lies above every register an argument window is coloured in.
                let LentCall {
                    run: args,
                    out,
                    ctx,
                } = unsafe { m.lend_call(self.window.at, self.window.arity, run.at, run.width) };
                // SAFETY: as `CallRun0`'s; the run is the window `prepare` laid.
                unsafe { self.f.call(ctx, args, out) };
                let regs = m.regs();
                for at in &run.releases {
                    regs.claim(*at);
                }
            }
            RunDest::Heap { dst, shape, width } => {
                let mut values: Box<[Owned<AcvusRuntime>]> =
                    (0..*width).map(|_| Owned::default()).collect();
                // SAFETY: every slot is `Owned::default()`, which owns nothing.
                let out = unsafe { acvus_extern::lend_run(&mut values) };
                let Lent { run, ctx } = m.lend_and_window(self.window.at, self.window.arity);
                // SAFETY: as `CallRun0`'s; the run is the window `prepare` laid.
                unsafe { self.f.call(ctx, run, out) };
                let object = Value::object(Arc::clone(shape), values);
                m.regs().define::<true>(*dst, object);
            }
        }
        self.next.run(m, r0)
    }
}

pub struct CallPairWindow<H> {
    pub dst: SlicePair,
    pub window: ArgWindow,
    pub f: H,
    pub next: Box<dyn Op>,
}

impl<H> Op for CallPairWindow<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let Lent { run, ctx } = self.window.lend(m);
        // SAFETY: as `CallPair1`'s; the run is the window `prepare` laid.
        let out = unsafe { self.f.call_pair_run(ctx, run) };
        land_pair(m, self.dst, out);
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
///
/// A run of one handler type is not built: the calls of a run reach
/// different declarations, so the run holds nodes and the `dyn` is over the
/// node rather than over the handler (RFC-0059 rule 4 amended).
pub type Call = Box<dyn Invoke>;

pub trait Invoke: Send + Sync {
    fn invoke(&self, m: &mut Machine<'_>, held: Value) -> Value;
}

pub struct Nullary<H> {
    f: H,
}

pub struct Unary<H> {
    f: H,
    a: Off,
}

pub struct Binary<H> {
    f: H,
    a: Off,
    b: Off,
}

impl<H> Invoke for Nullary<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    #[inline]
    fn invoke(&self, m: &mut Machine<'_>, _: Value) -> Value {
        // SAFETY: `prepare::fusable_call` admits a call into this run only at
        // the form each node names.
        unsafe { self.f.call0(&mut m.ctx) }
    }
}

impl<H> Invoke for Unary<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    #[inline]
    fn invoke(&self, m: &mut Machine<'_>, held: Value) -> Value {
        let a = arg(m, held, self.a);
        // SAFETY: as `Nullary`'s, at one argument.
        unsafe { self.f.call1(&mut m.ctx, a) }
    }
}

impl<H> Invoke for Binary<H>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    #[inline]
    fn invoke(&self, m: &mut Machine<'_>, held: Value) -> Value {
        let a = arg(m, held, self.a);
        let b = arg(m, held, self.b);
        // SAFETY: as `Nullary`'s, at two arguments.
        unsafe { self.f.call2(&mut m.ctx, a, b) }
    }
}

#[inline]
fn arg(m: &mut Machine<'_>, held: Value, at: Off) -> Value {
    match at {
        PREVIOUS => held,
        at => m.regs().read(at),
    }
}

/// RFC-0044, stage 6.
///
/// `CALLS` and `TAIL` are the run's shape, which `prepare` resolved like every
/// other static fact: a body of this instance holds no loop bound.
pub struct Fused<const CALLS: usize, const TAIL: bool, const LARGE: bool> {
    pub dst: Marked,
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
            held = call.invoke(m, held);
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
    dst: Marked,
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
    dst: Marked,
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
    pub dst: Marked,
    pub window: ArgWindow,
    pub f: Arc<dyn SentAsync>,
    pub next: BlockId,
}

impl<const LARGE: bool> Op for CallExternAsync<LARGE> {
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        let rt = m.ctx.rt.clone();
        let Lent { run, .. } = self.window.lend(m);
        // SAFETY: `prepare` built this operation from this handler's width,
        // and the future owns the arguments it is given.
        let fut = unsafe { self.f.call_async(rt, run) };
        m.suspend::<LARGE>(self.dst, self.next, fut);
        SUSPEND
    }
}

/// A `heavy` extern (RFC-0046): a Rust `fn`, but one worth another thread.
/// The work outlives this frame, so it owns its arguments as a spawn's does;
/// the call then awaits the handle, so the site suspends exactly as an
/// `async fn` extern's does.
pub struct CallHeavy<const LARGE: bool> {
    pub dst: Marked,
    pub window: ArgWindow,
    pub f: Arc<dyn SentCall>,
    pub next: BlockId,
}

impl<const LARGE: bool> Op for CallHeavy<LARGE> {
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        let args = self.window.own(m);
        let rt = m.ctx.rt.clone();
        let f = Arc::clone(&self.f);
        let executor = Arc::clone(&m.shared().executor);
        // SAFETY: the window is this call's whole argument run, owned by the
        // closure the pool runs.
        let handle = executor.spawn_blocking(Box::new(move || {
            let mut rooted = rt.rooted();
            unsafe { f.call_owned(AcvusRuntime::ctx_of(&mut rooted), &args) }
        }));
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
pub struct CallDirect<const LARGE: bool, const WORD: bool, const PAIR: bool> {
    pub dst: Marked,
    pub callee: QualifiedRef,
    pub arity: u16,
    pub takes: u64,
    pub next: Box<dyn Op>,
}

impl<const LARGE: bool, const WORD: bool, const PAIR: bool> Op for CallDirect<LARGE, WORD, PAIR> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        const {
            assert!(
                !(PAIR && (LARGE || WORD)),
                "a view's two registers hold neither a Large nor a kind the frame opened"
            )
        }
        m.regs().take_mask(self.takes);
        let prepared = Arc::clone(lookup_module(m.shared(), &self.callee));
        match PAIR {
            true => {
                let out: Words = call_module_sync(m, &prepared, self.callee, self.arity);
                land_words(m, SlicePair::at(self.dst.at), out);
            }
            false => {
                let value: Value = call_module_sync(m, &prepared, self.callee, self.arity);
                m.regs().store::<LARGE, WORD>(self.dst, value);
            }
        }
        self.next.run(m, r0)
    }
}

/// The same call where the callee's task is above `Sync`: it hands the driver
/// a future and leaves the block, which is why this one is a terminator and
/// `CallDirect` is not.
pub struct CallDirectAsync<const LARGE: bool, const PAIR: bool> {
    pub dst: Marked,
    pub callee: QualifiedRef,
    pub args: Box<[Off]>,
    pub takes: u64,
    pub next: BlockId,
}

impl<const LARGE: bool, const PAIR: bool> Op for CallDirectAsync<LARGE, PAIR> {
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        const {
            assert!(
                !(PAIR && LARGE),
                "a view's two registers hold no Large the frame owns"
            )
        }
        let args = staged(m, &self.args, self.takes);
        let rt = m.ctx.rt.clone();
        match PAIR {
            true => {
                let fut = Box::pin(call_module::<Words>(rt, self.callee, args));
                m.suspend_pair(SlicePair::at(self.dst.at), self.next, fut);
            }
            false => {
                let fut = Box::pin(call_module::<Value>(rt, self.callee, args));
                m.suspend::<LARGE>(self.dst, self.next, fut);
            }
        }
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
    callee: Marked,
    arity: u16,
) -> Value {
    match THROUGH {
        true => {
            let closure: &FnValue = unsafe {
                let target = m.regs().peek(callee.at).target();
                &*(target.as_fn() as *const FnValue)
            };
            m.call_fn_sync(closure, arity)
        }
        false => {
            let held = m.regs().take::<true>(callee);
            // SAFETY: the type checker admits only a closure value here.
            let value = m.call_fn_sync(unsafe { held.as_fn() }, arity);
            held.release();
            value
        }
    }
}

/// A closure call whose type says `Sync`: as `CallDirect`, run to its value
/// inside the block. It asks nothing of the closure but the two words its
/// own record's head holds.
pub struct CallIndirect<const LARGE: bool, const WORD: bool, const THROUGH: bool> {
    pub dst: Marked,
    pub callee: Marked,
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
    pub dst: Marked,
    pub callee: Marked,
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
                let (closure, rt): (&'static FnValue, &'static AcvusRuntime) = unsafe {
                    let rt = &*(m.ctx.rt as *const AcvusRuntime);
                    let target = m.regs().peek(self.callee.at).target();
                    (&*(target.as_fn() as *const FnValue), rt)
                };
                Box::pin(fn_value_call(closure, rt, &mut args))
            }
            // SAFETY: as the `true` arm's, for the runtime; the closure value
            // is taken out of the register and the future owns the record.
            false => {
                let rt = unsafe { &*(m.ctx.rt as *const AcvusRuntime) };
                let held = m.regs().take::<true>(self.callee);
                Box::pin(async move {
                    // SAFETY: the type checker admits only a closure value here.
                    let value = fn_value_call(unsafe { held.as_fn() }, rt, &mut args).await;
                    held.release();
                    value
                })
            }
        };
        m.suspend::<LARGE>(self.dst, self.next, fut);
        SUSPEND
    }
}

pub struct Eval<const LARGE: bool> {
    pub dst: Marked,
    pub handle: Marked,
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
    pub dst: Marked,
    pub window: ArgWindow,
    pub f: Arc<dyn SentCall>,
    pub next: Box<dyn Op>,
}

impl Op for SpawnExternSync {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let args = self.window.own(m);
        let rt = m.ctx.rt.clone();
        let f = Arc::clone(&self.f);
        // SAFETY: as `CallHeavy`'s.
        let handle = m.shared().executor.spawn_blocking(Box::new(move || {
            let mut rooted = rt.rooted();
            unsafe { f.call_owned(AcvusRuntime::ctx_of(&mut rooted), &args) }
        }));
        m.regs().define::<true>(self.dst, Value::handle(handle));
        self.next.run(m, r0)
    }
}

pub struct SpawnExternAsync {
    pub dst: Marked,
    pub window: ArgWindow,
    pub f: Arc<dyn SentAsync>,
    pub next: Box<dyn Op>,
}

impl Op for SpawnExternAsync {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let args = self.window.own(m);
        let rt = m.ctx.rt.clone();
        // SAFETY: as `CallExternAsync`'s; the spawned future owns `args`.
        let fut = unsafe { self.f.call_async(rt, &args) };
        let handle = m.shared().executor.spawn_async(fut);
        m.regs().define::<true>(self.dst, Value::handle(handle));
        self.next.run(m, r0)
    }
}

pub struct SpawnModule {
    pub dst: Marked,
    pub callee: QualifiedRef,
    pub args: Box<[Off]>,
    pub takes: u64,
    pub next: Box<dyn Op>,
}

impl Op for SpawnModule {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let args = staged(m, &self.args, self.takes);
        let child = crate::interpreter::Interpreter::spawned(m.ctx.rt.clone(), self.callee, args);
        let handle = m.shared().executor.spawn_interpreter(child);
        m.regs().define::<true>(self.dst, Value::handle(handle));
        self.next.run(m, r0)
    }
}

/// One allocation and no atomic: the record is written once, with the code
/// word `prepare` fixed and the captures read out of this frame's registers.
pub struct MakeClosure {
    pub dst: Marked,
    pub code: CodeRef,
    pub captures: Box<[Off]>,
    pub takes: u64,
    pub next: Box<dyn Op>,
}

impl Op for MakeClosure {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let closure = {
            let regs = m.regs();
            let closure =
                Value::closure(self.code, self.captures.iter().map(|slot| regs.read(*slot)));
            regs.take_mask(self.takes);
            closure
        };
        m.regs().define::<true>(self.dst, closure);
        self.next.run(m, r0)
    }
}
