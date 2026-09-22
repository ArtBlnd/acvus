//! The machine that runs a prepared body (RFC-0052 §2).
//!
//! `Machine::run` is synchronous and is one loop with one compare: a block
//! runs its operations straight through and its terminator names the next
//! block. `drive` is the one `async fn` around it — it awaits the future a
//! `Suspend` left behind, stores the result in the slot that terminator named,
//! and re-enters the loop at the block after it. A run-time failure is a panic
//! and leaves through the unwinder, not through this loop.

use std::fmt::Debug;
use std::future::Future;
use std::mem::MaybeUninit;
use std::slice;
use std::sync::Arc;

use acvus_extern::{Ctx, Words};
use acvus_mir::graph::QualifiedRef;
use acvus_utils::Interner;
use futures::future::BoxFuture;

use crate::code::{
    BlockId, Body, Code, EntryKonst, Exit, Expr, ExprBody, ExprChain, Marked, Off, Op, Pending,
    Prepared, RETURN, SENTINEL, SUSPEND, SlicePair,
};
use crate::interpreter::{InterpreterContext, lookup_module};
use crate::journal::RuntimeContext;
use crate::regs::{FrameState, Regs, Store};
use crate::runtime::AcvusRuntime;
use crate::value::Value;

/// What an extern call is lent of the frame it runs in: the run its arguments
/// sit in, and the context it runs in (RFC-0050 rule 6).
pub struct Lent<'r, 'c> {
    pub run: &'r [Value],
    pub ctx: &'r mut Ctx<'c, AcvusRuntime>,
}

pub struct LentOut<'r, 'c> {
    pub out: &'r mut [Value],
    pub ctx: &'r mut Ctx<'c, AcvusRuntime>,
}

pub struct LentCall<'r, 'c> {
    pub run: &'r [Value],
    pub out: &'r mut [Value],
    pub ctx: &'r mut Ctx<'c, AcvusRuntime>,
}

/// Obligation across artifacts: the two-`Value` run a `Words` is read out of
/// here is the one `acvus_extern`'s `Ret::into_run` writes and
/// `Runtime::slice_from_run` reads, so a view crossing a body call and a view
/// crossing an extern call are one representation (RFC-0062 Decision 1).
pub trait Returned {
    fn of(exit: [Value; 2]) -> Self;
}

impl Returned for Value {
    #[inline(always)]
    fn of(exit: [Value; 2]) -> Value {
        exit[0]
    }
}

impl Returned for Words {
    #[inline(always)]
    fn of(exit: [Value; 2]) -> Words {
        Words {
            ptr: exit[0].bits(),
            len: exit[1].bits(),
        }
    }
}

pub struct Machine<'c> {
    body: &'c Body,
    regs: Regs<'c>,
    pub ctx: Ctx<'c, AcvusRuntime>,
    exit: [Value; 2],
    /// The block `run` enters at. A suspension writes the block after
    /// itself here, so the driver's next `run` resumes rather than
    /// re-entering the body.
    at: BlockId,
    pending: Option<Pending>,
}

impl<'c> Machine<'c> {
    pub fn new(body: &'c Body, mut regs: Regs<'c>, rt: &'c AcvusRuntime) -> Machine<'c> {
        let frame = regs.take_window();
        Machine {
            body,
            regs,
            ctx: Ctx::new(rt, frame),
            exit: [Value::unit(); 2],
            at: body.entry,
            pending: None,
        }
    }

    /// One compare per joint. The index is unchecked because every
    /// terminator's target was resolved by `prepare` against this same
    /// `heads` array, and the two sentinels are what the compare catches.
    pub fn run(&mut self) -> Exit {
        let body = self.body;
        let mut at: Exit = self.at.into();
        loop {
            debug_assert!(
                (at as usize) < body.heads.len(),
                "a terminator names block {at}, which this body does not have"
            );
            // SAFETY: `prepare::link` resolves every terminator target to an
            // index of this array or to a sentinel, and a sentinel leaves the
            // loop at the compare below before it is used as an index.
            let head: &Box<dyn Op> = unsafe { body.heads.get_unchecked(at as usize) };
            at = head.run(self, 0);
            if at >= SENTINEL {
                return at;
            }
        }
    }

    pub fn body(&self) -> &'c Body {
        self.body
    }

    #[inline(always)]
    pub fn regs(&mut self) -> &mut Regs<'c> {
        &mut self.regs
    }

    pub fn shared(&self) -> &Arc<InterpreterContext> {
        &self.ctx.rt.shared
    }

    pub fn interner(&self) -> &Interner {
        &self.ctx.rt.shared.interner
    }

    /// The body returns this value; the `Return` terminator leaves the loop.
    #[inline]
    pub fn finish(&mut self, value: Value) {
        self.exit[0] = value;
    }

    #[inline]
    pub fn finish_pair(&mut self, ptr: u64, len: u64) {
        self.exit = [crate::runtime::word(ptr), crate::runtime::word(len)];
    }

    /// The future the driver awaits, the slot its result lands in, and the
    /// block the body goes on at once it has (RFC-0046).
    #[inline]
    pub fn suspend<const LARGE: bool>(
        &mut self,
        dst: Marked,
        resume: BlockId,
        fut: BoxFuture<'static, Value>,
    ) {
        self.at = resume;
        self.pending = Some(Pending::Word {
            dst,
            owns_large: LARGE,
            fut,
        });
    }

    #[inline]
    pub fn suspend_pair(
        &mut self,
        dst: SlicePair,
        resume: BlockId,
        fut: BoxFuture<'static, Words>,
    ) {
        self.at = resume;
        self.pending = Some(Pending::Pair { dst, fut });
    }

    /// The cells a call out of this frame takes its callee's frame from, and
    /// the body this frame's window is bound to (RFC-0050 rule 6).
    #[inline(always)]
    pub fn window(&mut self) -> &mut FrameState {
        &mut self.ctx.frame
    }

    /// The argument window, the destination run and the window above, for an
    /// aggregate-returning call whose arguments do not fit the register forms.
    ///
    /// # Safety
    /// The two runs are disjoint ranges of this frame's registers.
    /// `prepare::plan_runs` places every destination run above the scalar
    /// registers an argument window is coloured in, and
    /// `Prepare::call_into_run` asserts that of the run it names.
    #[inline(always)]
    pub unsafe fn lend_call<'r>(
        &'r mut self,
        args: Off,
        arity: u16,
        at: Off,
        width: u16,
    ) -> LentCall<'r, 'c> {
        debug_assert!(
            args.index() + usize::from(arity) <= at.index()
                || at.index() + usize::from(width) <= args.index(),
            "an argument run and a destination run overlap"
        );
        let run = self.regs.run_of(args, arity);
        // SAFETY: the caller's contract, which the `debug_assert!` above
        // re-checks in a debug build: the two slices name disjoint registers.
        let run = unsafe { std::slice::from_raw_parts(run.as_ptr(), run.len()) };
        LentCall {
            run,
            out: self.regs.run_of_mut(at, width),
            ctx: &mut self.ctx,
        }
    }

    /// Lent together for the reason `lend_and_window` lends its pair: the
    /// destination run is this frame's own registers and the window is the
    /// cells above them.
    #[inline(always)]
    pub fn lend_out_and_window<'r>(&'r mut self, at: Off, width: u16) -> LentOut<'r, 'c> {
        LentOut {
            out: self.regs.run_of_mut(at, width),
            ctx: &mut self.ctx,
        }
    }

    /// Lent together because the run is this frame's own registers and the
    /// window is the cells above them: two fields, two disjoint borrows.
    #[inline(always)]
    pub fn lend_and_window<'r>(&'r mut self, at: Off, arity: u16) -> Lent<'r, 'c> {
        Lent {
            run: self.regs.run_of(at, arity),
            ctx: &mut self.ctx,
        }
    }

    /// Run `callee` to its result in the window above this frame (RFC-0052
    /// rule 7): no allocation, one mask store to enter and one sweep to leave.
    /// The arguments are already in the window's first registers, where the
    /// call's `LayArg` operations put them.
    pub fn call_sync<F, R>(&mut self, callee: &Body, named: &dyn Debug, arity: u16, fill: F) -> R
    where
        F: FnOnce(&mut Machine<'_>),
        R: Returned,
    {
        let rt = self.ctx.rt;
        let window = &mut self.ctx.frame;
        if window.fits(callee) {
            let (regs, opened) = window.bind(callee);
            return run_frame(callee, named, regs, rt, opened, fill);
        }
        run_rooted(window, callee, named, arity, rt, fill)
    }

    pub fn call_fn_sync(&mut self, f: &Value, arity: u16) -> Value {
        // SAFETY: the caller's contract: `f` is a closure.
        unsafe { f.code_of() }.code().call_in(f, arity, self)
    }
}

/// A `Store` of the callee's own, the argument run copied into it.
#[cold]
#[inline(never)]
fn run_rooted<F, R>(
    window: &mut FrameState,
    callee: &Body,
    named: &dyn Debug,
    arity: u16,
    rt: &AcvusRuntime,
    fill: F,
) -> R
where
    F: FnOnce(&mut Machine<'_>),
    R: Returned,
{
    let mut store = Store::new();
    let (mut regs, _) = store.bind(callee);
    open_frame(callee, &mut regs);
    for (at, arg) in window.laid(arity).iter().enumerate() {
        let slot = u16::try_from(at).expect("an argument run is at most one cell wide");
        regs.open(Off::of(slot), *arg);
    }
    run_frame(callee, named, regs, rt, true, fill)
}

/// What a frame holds for as long as it is bound to one body: the kind byte
/// of every word-typed register and the body's entry constants. A `set_word`
/// leaves the kind byte and an entry constant's register has no writer, so a
/// second call on the same frame reads what this wrote (RFC-0052 §5, §6).
fn open_frame(body: &Body, regs: &mut Regs<'_>) {
    regs.open_marks(body.mark_words);
    for kind in &body.slot_kinds {
        regs.open(kind.slot, Value::inline(kind.kind, 0));
    }
    for EntryKonst { slot, value } in &body.entry_konsts {
        regs.open(*slot, *value);
    }
}

/// # Panics
/// `body` is typed pure and its prepared body can suspend, which is a
/// disagreement between the effect the checker read and the body `prepare`
/// produced.
fn run_frame<F, R>(
    body: &Body,
    named: &dyn Debug,
    mut regs: Regs<'_>,
    rt: &AcvusRuntime,
    opened: bool,
    fill: F,
) -> R
where
    F: FnOnce(&mut Machine<'_>),
    R: Returned,
{
    debug_assert!(
        !body.may_suspend,
        "{named:?} is typed pure, and its prepared body can suspend"
    );
    if !opened {
        open_frame(body, &mut regs);
    }
    let mut machine = Machine::new(body, regs, rt);
    fill(&mut machine);
    let stop = machine.run();
    debug_assert_eq!(
        stop, RETURN,
        "{named:?} is typed pure, and its body left the machine at {stop}"
    );
    let exit = machine.exit;
    machine.regs.sweep(body.mark_words);
    R::of(exit)
}

async fn drive<R>(mut machine: Machine<'_>) -> R
where
    R: Returned,
{
    loop {
        let stop = machine.run();
        if stop == RETURN {
            let exit = machine.exit;
            machine.regs.sweep(machine.body.mark_words);
            return R::of(exit);
        }
        debug_assert_eq!(stop, SUSPEND, "a body left the machine at {stop}");
        let pending = machine
            .pending
            .take()
            .expect("a Suspend terminator left no future for the driver");
        match pending {
            Pending::Word {
                dst,
                owns_large,
                fut,
            } => {
                let value = fut.await;
                match owns_large {
                    true => machine.regs.define::<true>(dst, value),
                    false => machine.regs.put(dst.at, value),
                }
            }
            Pending::Pair { dst, fut } => {
                let Words { ptr, len } = fut.await;
                machine.regs.set_word(dst.ptr, ptr);
                machine.regs.set_word(dst.len, len);
            }
        }
    }
}

/// Run the entry body of the module `id` names.
pub async fn call_module<R>(
    shared: Arc<InterpreterContext>,
    page: Arc<dyn RuntimeContext>,
    id: QualifiedRef,
    args: Vec<Value>,
) -> R
where
    R: Returned,
{
    let prepared: Arc<Prepared> = Arc::clone(lookup_module(&shared, &id));
    let rt = AcvusRuntime::new(shared, page);
    let body = prepared.main.as_ref();
    let mut store = Store::new();
    let (mut regs, _) = store.bind(body);
    open_frame(body, &mut regs);
    for (slot, arg) in body.params.iter().zip(args) {
        regs.put(*slot, arg);
    }
    if let Some(order) = body.order_param {
        regs.put(order, Value::unit());
    }
    drive(Machine::new(body, regs, &rt)).await
}

pub fn call_module_sync<R>(
    machine: &mut Machine<'_>,
    prepared: &Prepared,
    id: QualifiedRef,
    arity: u16,
) -> R
where
    R: Returned,
{
    let body = prepared.main.as_ref();
    machine.call_sync(body, &id, arity, |callee| {
        if let Some(order) = body.order_param {
            callee.regs.put(order, Value::unit());
        }
    })
}

impl Code {
    /// Run in the window a handler was lent, on the argument run the handler
    /// laid in its first `arity` registers (RFC-0050 rule 6), in the run the
    /// handler's `Ctx` names.
    pub fn call_in_window(
        &self,
        f: &Value,
        rt: &AcvusRuntime,
        window: &mut FrameState,
        arity: u16,
    ) -> Value {
        match self {
            Code::Body(body) => body.call_in_window(f, rt, window, arity),
            Code::Expr(expr) => expr_value(expr, window.laid(arity)),
        }
    }

    /// Run in the window above the calling frame, on the argument run the
    /// caller laid in that window's first `arity` registers (RFC-0052 rule 7).
    pub fn call_in(&self, f: &Value, arity: u16, m: &mut Machine<'_>) -> Value {
        match self {
            Code::Body(body) => m.call_sync(body, &body.span, arity, |callee| {
                bind_captures(body, f, &mut callee.regs)
            }),
            Code::Expr(expr) => expr_value(expr, m.window().laid(arity)),
        }
    }

    /// The arguments read into the callee's frame before the future exists,
    /// so the caller may lend registers that die at the call.
    pub fn start<'c>(&'c self, f: &Value, args: &mut [Value]) -> Resume<'c> {
        match self {
            Code::Body(body) => body.start(f, args),
            Code::Expr(expr) => Resume::Done(expr_value(expr, args)),
        }
    }
}

/// What a closure call has to do after its arguments are read: run a body on a
/// frame, or — for an `Expr` — nothing, because the chain has already produced
/// the value.
pub enum Resume<'c> {
    Frame { body: &'c Body, store: Store },
    Done(Value),
}

impl Body {
    /// A chain (`Code::Expr`) runs with no frame; this arm keeps its own.
    #[inline(never)]
    fn call_in_window(
        &self,
        f: &Value,
        rt: &AcvusRuntime,
        window: &mut FrameState,
        arity: u16,
    ) -> Value {
        if !window.fits(self) {
            return run_rooted(window, self, &self.span, arity, rt, |callee| {
                bind_captures(self, f, &mut callee.regs)
            });
        }
        let (mut regs, bound) = window.bind(self);
        if !bound {
            open_frame(self, &mut regs);
        }
        bind_captures(self, f, &mut regs);
        let mut machine = Machine::new(self, regs, rt);
        let stop = machine.run();
        debug_assert_eq!(
            stop, RETURN,
            "{:?} is typed pure, and its body left the machine at {stop}",
            self.span
        );
        let exit = machine.exit;
        machine.regs.sweep(self.mark_words);
        Value::of(exit)
    }

    fn start<'c>(&'c self, f: &Value, args: &mut [Value]) -> Resume<'c> {
        let mut store = Store::new();
        {
            let (mut regs, _) = store.bind(self);
            open_frame(self, &mut regs);
            fill(self, f, args, &mut regs);
        }
        Resume::Frame { body: self, store }
    }
}

pub fn fn_value_call<'f>(
    f: &'f Value,
    rt: &'f AcvusRuntime,
    args: &mut [Value],
) -> impl Future<Output = Value> + Send + use<'f> {
    // SAFETY: the caller's contract: `f` is a closure.
    let resume = unsafe { f.code_of() }.code().start(f, args);
    async move {
        match resume {
            Resume::Frame { body, mut store } => {
                let (regs, _) = store.bind(body);
                let machine = Machine::new(body, regs, rt);
                drive(machine).await
            }
            Resume::Done(value) => value,
        }
    }
}

pub fn fn_value_call_in_window(
    f: &Value,
    rt: &AcvusRuntime,
    window: &mut FrameState,
    arity: u16,
) -> Value {
    // SAFETY: the caller's contract: `f` is a closure.
    unsafe { f.code_of() }
        .code()
        .call_in_window(f, rt, window, arity)
}

/// The closure owns its captures; the body sees each through a reference
/// (RFC-0018).
fn bind_captures(body: &Body, f: &Value, regs: &mut Regs<'_>) {
    // SAFETY: the caller's contract on every entry: `f` is a closure.
    let captures = unsafe { f.captures_of() };
    for (slot, capture) in body.captures.iter().zip(captures) {
        regs.put(*slot, Value::reference(capture));
    }
    if let Some(order) = body.order_param {
        regs.put(order, Value::unit());
    }
}

/// The same, for the one path handed its arguments as values rather than as a
/// run of the caller's registers: the frame a suspending call runs on, which
/// exists before the future does (`Code::start`).
fn fill(body: &Body, f: &Value, args: &mut [Value], regs: &mut Regs<'_>) {
    bind_captures(body, f, regs);
    for (slot, arg) in body.params.iter().zip(args) {
        regs.put(*slot, *arg);
    }
}

/// A body that is one chain runs with no registers, no `Machine` and no
/// dispatch loop (RFC-0044, stage 4).
///
/// # Panics
/// The call brought an arity the body was not prepared with.
#[inline]
fn expr_value(expr: &Expr, args: &[Value]) -> Value {
    debug_assert_eq!(
        args.len(),
        expr.arity as usize,
        "an expression body is called with the arguments it reads"
    );
    match &expr.body {
        ExprBody::Argument(at) => args[*at as usize],
        ExprBody::Chain(chain) => chain_value(chain, args),
    }
}

/// The `#[inline(never)]` is a measurement, not a taste. The operand space is
/// `MAX_OPERANDS` `Value`s wide; while this was inlined, its frame was part of
/// every frameless call, and `map(|x| -> x) | sum` — which builds no operand
/// space at all — measured 4.8 ns per iteration instead of 4.1, and
/// `range | sum` 2.0 instead of 1.8. Splitting it restored both.
#[inline(never)]
fn chain_value(chain: &ExprChain, args: &[Value]) -> Value {
    let space = OperandSpace::of(args, &chain.konsts);
    Value::inline(chain.kind, (chain.eval)(chain, space.as_slice()))
}

struct OperandSpace {
    values: [MaybeUninit<Value>; ExprChain::MAX_OPERANDS],
    len: usize,
}

impl OperandSpace {
    /// The arguments, then the constants, which is the order
    /// `prepare::expression_body` assigned the chain's leaf offsets in.
    fn of(args: &[Value], konsts: &[Value]) -> OperandSpace {
        let len = args.len() + konsts.len();
        debug_assert!(
            len <= ExprChain::MAX_OPERANDS,
            "an expression body reads {len} operands, past the {} a frameless call builds",
            ExprChain::MAX_OPERANDS
        );
        let mut values = [const { MaybeUninit::uninit() }; ExprChain::MAX_OPERANDS];
        for (slot, value) in values.iter_mut().zip(args.iter().chain(konsts)) {
            slot.write(*value);
        }
        OperandSpace { values, len }
    }

    fn as_slice(&self) -> &[Value] {
        // SAFETY: `prepare::expression_body` refuses a body whose operands
        // outnumber `ExprChain::MAX_OPERANDS`, and `expr_value` asserts the
        // call brought the arity that body was prepared with, so `of` wrote
        // exactly `len` values into an array that holds them.
        unsafe { slice::from_raw_parts(self.values.as_ptr().cast::<Value>(), self.len) }
    }
}
