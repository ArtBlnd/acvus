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
//! it with no decision in between (RFC-0052 rule 6, RFC-0059 rule 7).

use std::marker::PhantomData;
use std::sync::Arc;

use acvus_extern::{
    ArgRun, CallForms, Ctx, InRegisters, InWindow, ObjectShape, One, OneRegister, OptionOf, Owned,
    Pair, RetForms, Returned, Run, Runtime, Words,
};
use acvus_mir::graph::QualifiedRef;
use futures::future::BoxFuture;
use smallvec::SmallVec;

use crate::code::{BlockId, Deref, Exit, Marked, Off, Op, SUSPEND, SlicePair, successor};
use crate::executor::{AsyncJob, BlockingJob};
use crate::flight::{Aloft, Launched};
use crate::interpreter::lookup_module;
use crate::machine::{
    Lent, LentCall, LentOut, Machine, call_module, call_module_sync, fn_value_call,
};
use crate::ops::control::{self, Ends, Escapes, Rejoins};
use crate::runtime::AcvusRuntime;
use crate::value::Value;
use acvus_extern::Release;

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
/// type is known (RFC-0059 rule 4).
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
    /// The call is a loop's whole head: `prepare::loop_op` recognized
    /// `while let Some(x) = f(&mut it)` and lowered the body without the
    /// `UnwrapOption` that stood at its head, so this shape carries the body
    /// and the ending as well as the destination.
    ForCall {
        it: Off,
        x: Marked,
        large: bool,
        word: bool,
        body: Box<dyn Op>,
        next: Box<dyn Op>,
        ends: Ends,
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

/// Where one call of a fused run reads its arguments (RFC-0044 rule 7).
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

/// The operation a call of `H`'s form is, by the two types its `Handler`
/// impl names (RFC-0044 rule 3, RFC-0047 rule 6): its arguments in
/// registers where the register forms cover them, else lent its window; its
/// result one value, the pair a view is, or the run an aggregate's
/// components fill. Both runs are types of `H`, so each instantiation is the
/// one arm and the body it builds holds no length to check.
pub fn op<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    <H::Args as ArgRun>::select(Shaped(shape), f)
}

/// Everything `prepare` settled about the call site, waiting on the argument
/// form `ArgRun::select` names.
struct Shaped(CallShape);

impl CallForms<AcvusRuntime> for Shaped {
    type Out = Box<dyn Op>;

    fn registers0<H>(self, f: H) -> Box<dyn Op>
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<0>>,
    {
        <H::Ret as Returned>::select(Registers0(self.0), f)
    }

    fn registers1<H>(self, f: H) -> Box<dyn Op>
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<1>>,
    {
        <H::Ret as Returned>::select(Registers1(self.0), f)
    }

    fn registers2<H>(self, f: H) -> Box<dyn Op>
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<2>>,
    {
        <H::Ret as Returned>::select(Registers2(self.0), f)
    }

    fn registers3<H>(self, f: H) -> Box<dyn Op>
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<3>>,
    {
        <H::Ret as Returned>::select(Registers3(self.0), f)
    }

    fn registers4<H>(self, f: H) -> Box<dyn Op>
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<4>>,
    {
        <H::Ret as Returned>::select(Registers4(self.0), f)
    }

    fn window<H>(self, f: H) -> Box<dyn Op>
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InWindow>,
    {
        <H::Ret as Returned>::select(Windowed(self.0), f)
    }
}

/// One call of a fused run: at most two of the runtime's values in, one out
/// (RFC-0044 rule 7). Any other form is refused by `prepare`'s recognizer
/// (`fusable_call`) before it reaches here.
pub fn fused_call<H>(f: H, shape: FusedShape) -> Call
where
    H: acvus_extern::Handler<AcvusRuntime>,
{
    <H::Args as ArgRun>::select(FusedShaped(shape), f)
}

struct FusedShaped(FusedShape);

impl CallForms<AcvusRuntime> for FusedShaped {
    type Out = Call;

    fn registers0<H>(self, f: H) -> Call
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<0>>,
    {
        <H::Ret as Returned>::select(FusedNullary(self.0), f)
    }

    fn registers1<H>(self, f: H) -> Call
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<1>>,
    {
        <H::Ret as Returned>::select(FusedUnary(self.0), f)
    }

    fn registers2<H>(self, f: H) -> Call
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<2>>,
    {
        <H::Ret as Returned>::select(FusedBinary(self.0), f)
    }

    fn registers3<H>(self, _: H) -> Call
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<3>>,
    {
        no_fused_call_of(3)
    }

    fn registers4<H>(self, _: H) -> Call
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<4>>,
    {
        no_fused_call_of(4)
    }

    fn window<H>(self, _: H) -> Call
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InWindow>,
    {
        panic!("a fused run holds no call whose arguments are lent their window")
    }
}

fn no_fused_call_of(arguments: usize) -> ! {
    panic!("a fused run holds no call of {arguments} arguments")
}

fn no_fused_pair() -> ! {
    panic!(
        "a fused run hands one value from each call to the next and holds no call whose \
         result is a pair"
    )
}

fn no_fused_run() -> ! {
    panic!(
        "a fused run hands one value from each call to the next and holds no call whose \
         result is an aggregate's components"
    )
}

/// One argument form of a fused run: the node it builds where the result is
/// one value, and the two refusals `prepare`'s recognizer already made.
macro_rules! fused_form {
    ($selector:ident, $args:ty, $shape:pat, $node:expr, $form:literal) => {
        struct $selector(FusedShape);

        impl RetForms<AcvusRuntime> for $selector {
            type Args = $args;
            type Out = Call;

            fn one<H>(self, f: H) -> Call
            where
                H: acvus_extern::Handler<AcvusRuntime, Args = $args, Ret: OneRegister>,
            {
                let $shape = self.0 else { not_this_form($form) };
                $node(f)
            }

            fn pair<H>(self, _: H) -> Call
            where
                H: acvus_extern::Handler<AcvusRuntime, Args = $args, Ret = Pair>,
            {
                no_fused_pair()
            }

            fn run<const W: usize, H>(self, _: H) -> Call
            where
                H: acvus_extern::Handler<AcvusRuntime, Args = $args, Ret = Run<W>>,
            {
                no_fused_run()
            }
        }
    };
}

fused_form!(
    FusedNullary,
    InRegisters<0>,
    FusedShape::Nullary,
    |f| Box::new(Nullary::<H> { f }),
    "no argument"
);
fused_form!(
    FusedUnary,
    InRegisters<1>,
    FusedShape::Unary { a },
    |f| Box::new(Unary::<H> { f, a }),
    "one argument"
);
fused_form!(
    FusedBinary,
    InRegisters<2>,
    FusedShape::Binary { a, b },
    |f| Box::new(Binary::<H> { f, a, b }),
    "two arguments"
);

/// A call whose result lands in one value: the handler writes the register
/// and answers the verdict, and the operation lands the two (RFC-0069).
///
/// # Safety
/// `Handler::call`'s, at `run`.
#[inline(always)]
unsafe fn one<H>(
    f: &H,
    ctx: &mut Ctx<'_, AcvusRuntime>,
    run: <H::Args as ArgRun>::Run<'_, AcvusRuntime>,
) -> Value
where
    H: acvus_extern::Handler<AcvusRuntime, Ret: OneRegister>,
{
    let rt = ctx.rt;
    let mut out = [Value::default()];
    // SAFETY: the caller's contract.
    let verdict = unsafe { f.call(ctx, run, <H::Ret as OneRegister>::slot(&mut out)) };
    let [written] = out;
    <H::Ret as OneRegister>::land(rt, verdict, written)
}

/// A call whose result is the pair a view occupies.
///
/// # Safety
/// As `one`.
#[inline(always)]
unsafe fn pair<H>(
    f: &H,
    ctx: &mut Ctx<'_, AcvusRuntime>,
    run: <H::Args as ArgRun>::Run<'_, AcvusRuntime>,
) -> [Value; 2]
where
    H: acvus_extern::Handler<AcvusRuntime, Ret = Pair>,
{
    let mut out = [Value::default(); 2];
    // SAFETY: the caller's contract.
    let () = unsafe { f.call(ctx, run, &mut out) };
    out
}

/// The two tasks that leave this thread. Neither depends on the argument
/// form — both own the window — so every register form and the window form
/// end here, and both hold the handler behind a `dyn`: the value is sent, not
/// called (RFC-0059 rule 4).
fn sent_to_another_thread<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime, Ret: OneRegister>,
{
    match shape {
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

/// This family has no `Heavy` and no `Spawn` arm, which is a decision and not
/// an omission: a result two values wide is a borrow of the frame the call
/// laid its arguments on, and both of those tasks resume after that frame is
/// gone. `ExternHandler::heavy` takes `impl ValuesOnly<R>` and
/// `AsyncCall::WIDTH` fixes `ret: 1`, so no such handler can reach here.
fn pair_in_window<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime, Args = InWindow, Ret = Pair>,
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

/// The aggregate family's window arm. It has no `Heavy` and no `Spawn` arm
/// for the reason the pair family has none: both tasks resume after the frame
/// the destination run lives in is gone, and `ExternHandler::heavy` takes
/// `impl ValuesOnly<R>` while `AsyncCall::WIDTH` fixes `ret: 1`.
fn run_in_window<H>(f: H, shape: CallShape) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime, Args = InWindow, Ret: Returned<Verdict = ()>>,
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

/// `CallShape::ForCall` past the two facts the operation takes as constants.
struct ForCallSite {
    it: Off,
    x: Marked,
    body: Box<dyn Op>,
    next: Box<dyn Op>,
    ends: Ends,
}

/// The loop a head call drives, at the two endings its body can have.
fn for_call<H, const LARGE: bool, const WORD: bool>(f: H, site: ForCallSite) -> Box<dyn Op>
where
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<1>, Ret = OptionOf<One>>,
{
    let ForCallSite {
        it,
        x,
        body,
        next,
        ends,
    } = site;
    let src = control::Call::<H, LARGE, WORD> { it, x, f };
    match ends {
        Ends::Word => Box::new(control::For::<_, Rejoins> {
            src,
            body,
            next,
            ends: PhantomData,
        }),
        Ends::Verdict => Box::new(control::For::<_, Escapes> {
            src,
            body,
            next,
            ends: PhantomData,
        }),
    }
}

struct Registers0(CallShape);

impl RetForms<AcvusRuntime> for Registers0 {
    type Args = InRegisters<0>;
    type Out = Box<dyn Op>;

    fn one<H>(self, f: H) -> Box<dyn Op>
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<0>, Ret: OneRegister>,
    {
        match self.0 {
            CallShape::Registers0 { dst, large, next } => match large {
                true => Box::new(CallExtern0::<H, true> { dst, f, next }),
                false => Box::new(CallExtern0::<H, false> { dst, f, next }),
            },
            shape => sent_to_another_thread(f, shape),
        }
    }

    /// A declaration of no parameter returns a view of nothing, and the macro
    /// refuses it where the declaration is written: a result that borrows has
    /// no parameter to be a borrow of, which is the
    /// `view_returned_without_a_loan` compile-fail case (RFC-0047 rule 3).
    /// `prepare::call_into_pair` states the same refusal at its own
    /// `Registers(0)` arm, so no `CallPair0` exists for this to build.
    fn pair<H>(self, _: H) -> Box<dyn Op>
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<0>, Ret = Pair>,
    {
        panic!(
            "a declaration of no parameter returns a view of nothing: a result that borrows \
             has no parameter it can be a borrow of, and `#[extern_fn]` refuses the \
             declaration ahead of this (RFC-0047 rule 3)"
        )
    }

    fn run<const W: usize, H>(self, f: H) -> Box<dyn Op>
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<0>, Ret = Run<W>>,
    {
        match self.0 {
            CallShape::Run0 { dst, next } => Box::new(CallRun0::<H> { dst, f, next }),
            _ => not_this_form("an aggregate result at this arity"),
        }
    }
}

/// One register form of the three result families. The argument count is a
/// type, so each of the three methods matches the one shape `prepare` builds
/// for it and hands every other shape to the arm that owns it.
macro_rules! registers {
    (
        $selector:ident, $n:literal,
        $value_shape:pat, $value_node:expr,
        $pair_shape:pat, $pair_node:expr,
        $run_shape:pat, $run_node:expr
        $(, option: $option_shape:pat, $option_node:expr)?
    ) => {
        struct $selector(CallShape);

        impl RetForms<AcvusRuntime> for $selector {
            type Args = InRegisters<$n>;
            type Out = Box<dyn Op>;

            fn one<H>(self, f: H) -> Box<dyn Op>
            where
                H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<$n>, Ret: OneRegister>,
            {
                match self.0 {
                    $value_shape => $value_node(f),
                    CallShape::ForCall { .. } => not_this_form("a result that is always written"),
                    shape => sent_to_another_thread(f, shape),
                }
            }

            $(
                fn option<H>(self, f: H) -> Box<dyn Op>
                where
                    H: acvus_extern::Handler<
                        AcvusRuntime,
                        Args = InRegisters<$n>,
                        Ret = OptionOf<One>,
                    >,
                {
                    match self.0 {
                        $option_shape => $option_node(f),
                        shape => $selector(shape).one(f),
                    }
                }
            )?

            fn pair<H>(self, f: H) -> Box<dyn Op>
            where
                H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<$n>, Ret = Pair>,
            {
                match self.0 {
                    $pair_shape => $pair_node(f),
                    _ => not_this_form("a result two values wide at this arity"),
                }
            }

            fn run<const W: usize, H>(self, f: H) -> Box<dyn Op>
            where
                H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<$n>, Ret = Run<W>>,
            {
                match self.0 {
                    $run_shape => $run_node(f),
                    _ => not_this_form("an aggregate result at this arity"),
                }
            }
        }
    };
}

registers!(
    Registers1,
    1,
    CallShape::Registers1 {
        dst,
        a,
        takes,
        large,
        word,
        next,
    },
    |f| match large {
        true => Box::new(CallExtern1::<H, true, false> {
            dst,
            a,
            takes,
            f,
            next,
        }) as Box<dyn Op>,
        false => match word {
            true => Box::new(CallExtern1::<H, false, true> {
                dst,
                a,
                takes,
                f,
                next,
            }) as Box<dyn Op>,
            false => Box::new(CallExtern1::<H, false, false> {
                dst,
                a,
                takes,
                f,
                next,
            }),
        },
    },
    CallShape::Pair1 {
        dst,
        a,
        takes,
        next,
    },
    |f| Box::new(CallPair1::<H> {
        dst,
        a,
        takes,
        f,
        next,
    }),
    CallShape::Run1 {
        dst,
        a,
        takes,
        next,
    },
    |f| Box::new(CallRun1::<H> {
        dst,
        a,
        takes,
        f,
        next,
    }),
    option: CallShape::ForCall {
        it,
        x,
        large,
        word,
        body,
        next,
        ends,
    },
    |f| {
        let site = ForCallSite {
            it,
            x,
            body,
            next,
            ends,
        };
        match large {
            true => for_call::<H, true, false>(f, site),
            false => match word {
                true => for_call::<H, false, true>(f, site),
                false => for_call::<H, false, false>(f, site),
            },
        }
    }
);

registers!(
    Registers2,
    2,
    CallShape::Registers2 {
        dst,
        a,
        b,
        takes,
        large,
        next,
    },
    |f| match large {
        true => Box::new(CallExtern2::<H, true> {
            dst,
            a,
            b,
            takes,
            f,
            next,
        }) as Box<dyn Op>,
        false => Box::new(CallExtern2::<H, false> {
            dst,
            a,
            b,
            takes,
            f,
            next,
        }),
    },
    CallShape::Pair2 {
        dst,
        a,
        b,
        takes,
        next,
    },
    |f| Box::new(CallPair2::<H> {
        dst,
        a,
        b,
        takes,
        f,
        next,
    }),
    CallShape::Run2 {
        dst,
        a,
        b,
        takes,
        next,
    },
    |f| Box::new(CallRun2::<H> {
        dst,
        a,
        b,
        takes,
        f,
        next,
    })
);

registers!(
    Registers3,
    3,
    CallShape::Registers3 {
        dst,
        a,
        b,
        c,
        takes,
        large,
        next,
    },
    |f| match large {
        true => Box::new(CallExtern3::<H, true> {
            dst,
            a,
            b,
            c,
            takes,
            f,
            next,
        }) as Box<dyn Op>,
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
    CallShape::Pair3 {
        dst,
        a,
        b,
        c,
        takes,
        next,
    },
    |f| Box::new(CallPair3::<H> {
        dst,
        a,
        b,
        c,
        takes,
        f,
        next,
    }),
    CallShape::Run3 {
        dst,
        a,
        b,
        c,
        takes,
        next,
    },
    |f| Box::new(CallRun3::<H> {
        dst,
        a,
        b,
        c,
        takes,
        f,
        next,
    })
);

registers!(
    Registers4,
    4,
    CallShape::Registers4 {
        dst,
        a,
        b,
        c,
        d,
        takes,
        large,
        next,
    },
    |f| match large {
        true => Box::new(CallExtern4::<H, true> {
            dst,
            a,
            b,
            c,
            d,
            takes,
            f,
            next,
        }) as Box<dyn Op>,
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
    CallShape::Pair4 {
        dst,
        a,
        b,
        c,
        d,
        takes,
        next,
    },
    |f| Box::new(CallPair4::<H> {
        dst,
        a,
        b,
        c,
        d,
        takes,
        f,
        next,
    }),
    CallShape::Run4 {
        dst,
        a,
        b,
        c,
        d,
        takes,
        next,
    },
    |f| Box::new(CallRun4::<H> {
        dst,
        a,
        b,
        c,
        d,
        takes,
        f,
        next,
    })
);

/// The form whose arguments the register forms do not cover: the call is lent
/// the window they already sit in.
struct Windowed(CallShape);

impl RetForms<AcvusRuntime> for Windowed {
    type Args = InWindow;
    type Out = Box<dyn Op>;

    fn one<H>(self, f: H) -> Box<dyn Op>
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InWindow, Ret: OneRegister>,
    {
        match self.0 {
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
            shape => sent_to_another_thread(f, shape),
        }
    }

    fn pair<H>(self, f: H) -> Box<dyn Op>
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InWindow, Ret = Pair>,
    {
        pair_in_window(f, self.0)
    }

    fn run<const W: usize, H>(self, f: H) -> Box<dyn Op>
    where
        H: acvus_extern::Handler<AcvusRuntime, Args = InWindow, Ret = Run<W>>,
    {
        run_in_window(f, self.0)
    }
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
/// no longer has (RFC-0059 rule 4).
pub trait SentCall: Send + Sync {
    /// # Safety
    /// `run` is the call's whole argument run, owned by the caller of this
    /// method for as long as the call, and `ctx`'s frame is a window bound to
    /// no other frame.
    unsafe fn call_owned(&self, ctx: &mut Ctx<'_, AcvusRuntime>, run: &[Value]) -> Value;
}

impl<H> SentCall for H
where
    H: acvus_extern::Handler<AcvusRuntime, Ret: OneRegister>,
{
    unsafe fn call_owned(&self, ctx: &mut Ctx<'_, AcvusRuntime>, run: &[Value]) -> Value {
        // SAFETY: the caller's contract, which is `Handler::call`'s: the
        // window it owns is this declaration's whole argument run.
        unsafe { one(self, ctx, <H::Args as ArgRun>::from_slice(run)) }
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
/// rule 6).
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
    /// frame moves on, with the window it calls a closure in (RFC-0044
    /// rule 2; RFC-0050 rule 6).
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<0>, Ret: OneRegister>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        // SAFETY: `prepare` read this handler's width and built this
        // operation for the form it named.
        let value = unsafe { one(&self.f, &mut m.ctx, &[]) };
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<1>, Ret: OneRegister>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        regs.take_mask(self.takes);
        // SAFETY: as `CallExtern0`'s, at one argument.
        let value = unsafe { one(&self.f, &mut m.ctx, &[a]) };
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<2>, Ret: OneRegister>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        regs.take_mask(self.takes);
        // SAFETY: as `CallExtern0`'s, at two arguments.
        let value = unsafe { one(&self.f, &mut m.ctx, &[a, b]) };
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<3>, Ret: OneRegister>,
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
        let value = unsafe { one(&self.f, &mut m.ctx, &[a, b, c]) };
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<4>, Ret: OneRegister>,
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
        let value = unsafe { one(&self.f, &mut m.ctx, &[a, b, c, d]) };
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InWindow, Ret: OneRegister>,
{
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let Lent { run, ctx } = self.window.lend(m);
        // SAFETY: as `CallExtern0`'s; the run is the window `prepare` laid.
        let value = unsafe { one(&self.f, ctx, run) };
        m.regs().define::<LARGE>(self.dst, value);
        self.next.run(m, r0)
    }
}

/// The pair forms (RFC-0047 rule 6, RFC-0062 rule 4): the
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<1>, Ret = Pair>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        regs.take_mask(self.takes);
        // SAFETY: `prepare` read this handler's width and built this
        // operation for the form it named.
        let out = unsafe { pair(&self.f, &mut m.ctx, &[a]) };
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<2>, Ret = Pair>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        regs.take_mask(self.takes);
        // SAFETY: as `CallPair1`'s, at two arguments.
        let out = unsafe { pair(&self.f, &mut m.ctx, &[a, b]) };
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<3>, Ret = Pair>,
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
        let out = unsafe { pair(&self.f, &mut m.ctx, &[a, b, c]) };
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<4>, Ret = Pair>,
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
        let out = unsafe { pair(&self.f, &mut m.ctx, &[a, b, c, d]) };
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
                (0..*width).map(|_| Owned::vacant(unsafe { acvus_extern::Holding::new() })).collect();
            // SAFETY: every slot is `Owned::vacant(unsafe { acvus_extern::Holding::new() })`, which owns nothing, and
            // the handler writes each at most once.
            let out = unsafe { acvus_extern::lend_run(acvus_extern::Holding::new(), &mut values) };
            call(&mut m.ctx, out);
            let object = Value::object_with(shape, || values);
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<0>, Ret: Returned<Verdict = ()>>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        // SAFETY: `prepare` read this handler's width and built this
        // operation for the form it named; `land_run` lends a run of
        // `WIDTH.ret` values the caller owns.
        land_run(m, &self.dst, |ctx, out| {
            let () = unsafe { self.f.call(ctx, &[], <H::Ret as Returned>::from_slice(out)) };
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<1>, Ret: Returned<Verdict = ()>>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        regs.take_mask(self.takes);
        // SAFETY: as `CallRun0`'s, at one argument.
        land_run(m, &self.dst, |ctx, out| {
            let () = unsafe {
                self.f
                    .call(ctx, &[a], <H::Ret as Returned>::from_slice(out))
            };
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<2>, Ret: Returned<Verdict = ()>>,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let a = regs.read(self.a);
        let b = regs.read(self.b);
        regs.take_mask(self.takes);
        // SAFETY: as `CallRun0`'s, at two arguments.
        land_run(m, &self.dst, |ctx, out| {
            let () = unsafe {
                self.f
                    .call(ctx, &[a, b], <H::Ret as Returned>::from_slice(out))
            };
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<3>, Ret: Returned<Verdict = ()>>,
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
        land_run(m, &self.dst, |ctx, out| {
            let () = unsafe {
                self.f
                    .call(ctx, &[a, b, c], <H::Ret as Returned>::from_slice(out))
            };
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<4>, Ret: Returned<Verdict = ()>>,
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
        land_run(m, &self.dst, |ctx, out| {
            let () = unsafe {
                self.f
                    .call(ctx, &[a, b, c, d], <H::Ret as Returned>::from_slice(out))
            };
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InWindow, Ret: Returned<Verdict = ()>>,
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
                let () = unsafe {
                    self.f
                        .call(ctx, args, <H::Ret as Returned>::from_slice(out))
                };
                let regs = m.regs();
                for at in &run.releases {
                    regs.claim(*at);
                }
            }
            RunDest::Heap { dst, shape, width } => {
                let mut values: Box<[Owned<AcvusRuntime>]> =
                    (0..*width).map(|_| Owned::vacant(unsafe { acvus_extern::Holding::new() })).collect();
                // SAFETY: every slot is `Owned::vacant(unsafe { acvus_extern::Holding::new() })`, which owns nothing.
                let out = unsafe { acvus_extern::lend_run(acvus_extern::Holding::new(), &mut values) };
                let Lent { run, ctx } = m.lend_and_window(self.window.at, self.window.arity);
                // SAFETY: as `CallRun0`'s; the run is the window `prepare` laid.
                let () = unsafe { self.f.call(ctx, run, <H::Ret as Returned>::from_slice(out)) };
                let object = Value::object_with(shape, || values);
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InWindow, Ret = Pair>,
{
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let Lent { run, ctx } = self.window.lend(m);
        // SAFETY: as `CallPair1`'s; the run is the window `prepare` laid.
        let out = unsafe { pair(&self.f, ctx, run) };
        land_pair(m, self.dst, out);
        self.next.run(m, r0)
    }
}

/// The argument of a fused call that reads what the call before it produced
/// rather than a register (RFC-0044 rule 7).
pub const PREVIOUS: Off = Off::PREVIOUS;

/// One call of a fused run, at the shapes `prepare::FusableCall` admits:
/// three or fewer arguments including the held value. A slice-returning
/// handler is not among them — its result is a register pair, and a run
/// hands one `Value` from each call to the next.
///
/// A run of one handler type is not built: the calls of a run reach
/// different declarations, so the run holds nodes and the `dyn` is over the
/// node rather than over the handler (RFC-0059 rule 4).
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
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<0>, Ret: OneRegister>,
{
    #[inline]
    fn invoke(&self, m: &mut Machine<'_>, _: Value) -> Value {
        // SAFETY: `prepare::fusable_call` admits a call into this run only at
        // the form each node names.
        unsafe { one(&self.f, &mut m.ctx, &[]) }
    }
}

impl<H> Invoke for Unary<H>
where
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<1>, Ret: OneRegister>,
{
    #[inline]
    fn invoke(&self, m: &mut Machine<'_>, held: Value) -> Value {
        let a = arg(m, held, self.a);
        // SAFETY: as `Nullary`'s, at one argument.
        unsafe { one(&self.f, &mut m.ctx, &[a]) }
    }
}

impl<H> Invoke for Binary<H>
where
    H: acvus_extern::Handler<AcvusRuntime, Args = InRegisters<2>, Ret: OneRegister>,
{
    #[inline]
    fn invoke(&self, m: &mut Machine<'_>, held: Value) -> Value {
        let a = arg(m, held, self.a);
        let b = arg(m, held, self.b);
        // SAFETY: as `Nullary`'s, at two arguments.
        unsafe { one(&self.f, &mut m.ctx, &[a, b]) }
    }
}

#[inline]
fn arg(m: &mut Machine<'_>, held: Value, at: Off) -> Value {
    match at {
        PREVIOUS => held,
        at => m.regs().read(at),
    }
}

/// RFC-0044 rule 7.
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
            true => match self.tail {
                Some(read) => read(&held),
                // `shape` picks `TAIL` out of `tail.is_some()`, so this arm is
                // the pair having come apart.
                None => {
                    debug_assert!(false, "a fused instance with a tail holds none");
                    held
                }
            },
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
        let flying = rt.flight.start();
        // SAFETY: the window is this call's whole argument run, owned by the
        // closure the pool runs.
        let work = Aloft::new(
            move || {
                let mut rooted = rt.rooted();
                // SAFETY: the `Ctx` is lent to the handler alone, and safe code
                // reaches no second `Ctx` to exchange it with: `ctx_of`,
                // `Ctx::new` and `Ctx::frame_mut` are `unsafe`.
                unsafe { f.call_owned(AcvusRuntime::ctx_of(&mut rooted), &args) }
            },
            flying,
        );
        let unevaluated = m.ctx.rt.tally.spawned();
        let job = BlockingJob::new(Box::new(move || work.run()));
        let id = job.id();
        let handle = executor.spawn_blocking(job);
        m.suspend::<LARGE>(
            self.dst,
            self.next,
            Box::pin(async move {
                // The value goes to `dst`, a register the checker typed at the
                // spawn's result, which owns it from then on.
                let value = executor.eval(handle).await.of_job(id);
                unevaluated.evaluated();
                value
            }),
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
    let rt = m.ctx.rt;
    match THROUGH {
        true => {
            let closure: &Value = unsafe {
                let target = m.regs().peek(callee.at).target();
                &*(target as *const Value)
            };
            // SAFETY: the caller's contract: the register holds a closure, so
            // its code word names the `Code` this enters and the captures the
            // entry reads.
            unsafe {
                closure
                    .code_of()
                    .code()
                    .call(*closure, rt, m.window(), arity)
            }
        }
        false => {
            let taken = m.regs().take::<true>(callee);
            // SAFETY: as the `THROUGH` arm, for the closure this call took.
            let value = unsafe { taken.code_of().code().call(taken, rt, m.window(), arity) };
            taken.release();
            value
        }
    }
}

/// A closure call whose type says `Sync`: as `CallDirect`, run to its value
/// inside the block, with no test of the closure's own `Code`.
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
        let rt = m.ctx.rt.clone();
        // A closure value is one word beside its kind: copied into the future
        // it names the same record, which under `THROUGH` the register keeps
        // live for the call and otherwise the future owns.
        let closure: Value = match THROUGH {
            // SAFETY: the type checker admits only a live reference to a
            // closure here, and the register it names is not written during
            // the call.
            true => unsafe { *m.regs().peek(self.callee.at).target() },
            false => m.regs().take::<true>(self.callee),
        };
        let fut: BoxFuture<'static, Value> = Box::pin(async move {
            let value = fn_value_call(&closure, &rt, &mut args).await;
            if !THROUGH {
                closure.release();
            }
            value
        });
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
        let Launched {
            handle,
            job,
            unevaluated,
        } = unsafe { m.regs().take::<true>(self.handle).materialize::<Launched>() };
        let executor = Arc::clone(&m.shared().executor);
        m.suspend::<LARGE>(
            self.dst,
            self.next,
            Box::pin(async move {
                // The value goes to `dst`, a register the checker typed at the
                // spawn's result, which owns it from then on.
                let value = executor.eval(handle).await.of_job(job);
                unevaluated.evaluated();
                value
            }),
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
        let flying = rt.flight.start();
        // SAFETY: as `CallHeavy`'s.
        let work = Aloft::new(
            move || {
                let mut rooted = rt.rooted();
                // SAFETY: the `Ctx` is lent to the handler alone, and safe code
                // reaches no second `Ctx` to exchange it with: `ctx_of`,
                // `Ctx::new` and `Ctx::frame_mut` are `unsafe`.
                unsafe { f.call_owned(AcvusRuntime::ctx_of(&mut rooted), &args) }
            },
            flying,
        );
        let unevaluated = m.ctx.rt.tally.spawned();
        let job = BlockingJob::new(Box::new(move || work.run()));
        let launched = Launched {
            job: job.id(),
            handle: m.shared().executor.spawn_blocking(job),
            unevaluated,
        };
        m.regs().define::<true>(self.dst, Value::handle(launched));
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
        let flying = rt.flight.start();
        // SAFETY: as `CallExternAsync`'s; the spawned future owns `args`.
        let fut = unsafe { self.f.call_async(rt, &args) };
        let unevaluated = m.ctx.rt.tally.spawned();
        let job = AsyncJob::new(Box::pin(Aloft::new(fut, flying)));
        let launched = Launched {
            job: job.id(),
            handle: m.shared().executor.spawn_async(job),
            unevaluated,
        };
        m.regs().define::<true>(self.dst, Value::handle(launched));
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
        let job = crate::interpreter::Interpreter::spawned(m.ctx.rt, self.callee, args);
        let unevaluated = m.ctx.rt.tally.spawned();
        let launched = Launched {
            job: job.id(),
            handle: m.shared().executor.spawn_async(job),
            unevaluated,
        };
        m.regs().define::<true>(self.dst, Value::handle(launched));
        self.next.run(m, r0)
    }
}

pub struct MakeClosure {
    pub dst: Marked,
    pub code: crate::code::CodeRef,
    pub captures: Box<[Off]>,
    pub takes: u64,
    pub next: Box<dyn Op>,
}

impl Op for MakeClosure {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        if self.captures.is_empty() {
            m.regs().define::<true>(self.dst, Value::code(self.code));
            return self.next.run(m, r0);
        }
        let closure = {
            let regs = m.regs();
            let mut captures = self
                .captures
                .iter()
                // SAFETY: every register read here that owns a word is in the mask
                // `take_mask` clears below, so its word moves here; the others
                // hold words that own nothing (`prepare` builds the mask from the
                // operands that own).
                .map(|slot| unsafe { Owned::from_value(acvus_extern::Holding::new(), regs.read(*slot)) });
            let closure = Value::closure(self.code, &mut captures);
            regs.take_mask(self.takes);
            closure
        };
        m.regs().define::<true>(self.dst, closure);
        self.next.run(m, r0)
    }
}
