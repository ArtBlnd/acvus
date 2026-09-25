//! The machine that runs a prepared body (RFC-0052 rule 2).
//!
//! `Machine::run` is synchronous and is one loop with one compare: a block
//! runs its operations straight through and its terminator names the next
//! block. `drive` is the one `async fn` around it — it awaits the future a
//! `Suspend` left behind, stores the result in the slot that terminator named,
//! and re-enters the loop at the block after it. A run-time failure is a panic
//! and leaves through the unwinder, not through this loop.

use std::fmt::Debug;
use std::future::Future;
use std::sync::Arc;

use acvus_extern::{Ctx, MOST_MEMBERS, RustCallee, Words};
use acvus_mir::graph::QualifiedRef;
use acvus_utils::Interner;
use futures::future::BoxFuture;

use crate::code::{
    BlockId, Body, Code, CodeBody, EntryKonst, Exit, Marked, Off, Op, Pending, Prepared, RETURN,
    SENTINEL, SUSPEND, SlicePair,
};
use crate::flight::FrameCells;
use crate::interpreter::{InterpreterContext, lookup_module};
use crate::regs::{FrameSlot, FrameState, Regs, RootFrame, Store};
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
/// crossing an extern call are one representation (RFC-0047 rule 6).
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
        // SAFETY: the obligation above: `Words` is read here only for a body
        // whose result the checker types a view, and that body exits with a
        // slice pair, which its producer wrote from `into_pair`.
        unsafe { Words::from_pair([exit[0].bits(), exit[1].bits()]) }
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
            // SAFETY: `take_window` handed this machine the one handle to
            // the window's cells, which `regs` keeps live beside the `Ctx`.
            ctx: unsafe { Ctx::new(rt, frame) },
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
    pub fn suspend_unit(&mut self, resume: BlockId, fut: BoxFuture<'static, ()>) {
        self.at = resume;
        self.pending = Some(Pending::Unit { fut });
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
        // SAFETY: `window`'s callers are this interpreter's call ops, which
        // lay and bind the frame in place; no handler is handed a `Machine`.
        unsafe { self.ctx.frame_mut() }
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
        // SAFETY: the frame is bound in place, never moved out or replaced.
        let window = unsafe { self.ctx.frame_mut() };
        if window.fits(callee) {
            let (regs, opened) = window.bind(callee);
            return run_frame(callee, named, regs, rt, opened, fill);
        }
        run_rooted(window, callee, named, arity, rt, fill)
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
    let laid = window.laid(arity);
    for (slot, arg) in FrameSlot::first(laid.len()).zip(laid) {
        regs.open(Off::bounded(slot), *arg);
    }
    run_frame(callee, named, regs, rt, true, fill)
}

/// What a frame holds for as long as it is bound to one body: the kind byte
/// of every word-typed register and the body's entry constants. A `set_word`
/// leaves the kind byte and an entry constant's register has no writer, so a
/// second call on the same frame reads what this wrote (RFC-0052 rules 5 and 7).
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
                    false => machine.regs.put(dst.at(), value),
                }
            }
            Pending::Pair { dst, fut } => {
                let [ptr, len] = fut.await.into_pair();
                machine.regs.set_word(dst.ptr, ptr);
                machine.regs.set_word(dst.len, len);
            }
            Pending::Unit { fut } => fut.await,
        }
    }
}

/// Run the entry body of the module `id` names.
pub async fn call_module<R>(rt: AcvusRuntime, id: QualifiedRef, args: Vec<Value>) -> R
where
    R: Returned,
{
    let prepared: Arc<Prepared> = Arc::clone(lookup_module(&rt.shared, &id));
    let body = prepared.main.as_ref();
    assert_eq!(
        args.len(),
        body.params.len(),
        "{id:?} is run with {} arguments, and it takes {}",
        args.len(),
        body.params.len()
    );
    let (mut cells, rt) = FrameCells::open(Store::new(), &rt);
    let (mut regs, _) = cells.store().bind(body);
    open_frame(body, &mut regs);
    for (slot, arg) in body.params.iter().zip(args) {
        regs.put(*slot, arg);
    }
    if let Some(order) = body.order_param {
        regs.put(order, Value::unit());
    }
    drive(Machine::new(body, regs, &rt)).await
}

/// Run the entry body of the module `id` names to its result on a frame of
/// its own, from an operation that cannot wait: an init a `Fetch` runs
/// under synchronous access, which `Host::compile` admits only where its
/// body cannot suspend.
pub(crate) fn call_module_rooted(rt: &AcvusRuntime, id: QualifiedRef) -> Value {
    let prepared: Arc<Prepared> = Arc::clone(lookup_module(&rt.shared, &id));
    let body = prepared.main.as_ref();
    let mut store = Store::new();
    let (regs, _) = store.bind(body);
    run_frame(body, &id, regs, rt, false, |callee| {
        if let Some(order) = body.order_param {
            callee.regs.put(order, Value::unit());
        }
    })
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

/// The entry of a framed body: bind the callee's frame in `window` and run a
/// `Machine` on it (RFC-0052 rule 7), whether the window is the one above a
/// calling frame or the one a handler was lent (RFC-0050 rule 6).
///
/// The `#[inline(never)]` is a measurement (RFC-0069 rule 2): a frame inlined
/// into the caller is part of every call that reaches this `Code`, framed or
/// not.
///
/// # Safety
/// As `Entry`.
#[inline(never)]
pub(crate) unsafe fn entry_body(
    code: &Code,
    f: Value,
    rt: &AcvusRuntime,
    window: &mut FrameState,
    arity: u16,
) -> Value {
    // SAFETY: `Code::body` is the one constructor that writes this entry, and
    // it writes it beside the `CodeBody::Body` this reads.
    let body = unsafe { code.body.body_unchecked() };
    if !window.fits(body) {
        return run_rooted(window, body, &body.span, arity, rt, |callee| {
            bind_captures(body, &f, &mut callee.regs)
        });
    }
    let (mut regs, bound) = window.bind(body);
    if !bound {
        open_frame(body, &mut regs);
    }
    bind_captures(body, &f, &mut regs);
    let mut machine = Machine::new(body, regs, rt);
    let stop = machine.run();
    debug_assert_eq!(
        stop, RETURN,
        "{:?} is typed pure, and its body left the machine at {stop}",
        body.span
    );
    let exit = machine.exit;
    machine.regs.sweep(body.mark_words);
    Value::of(exit)
}

/// The entry of a body that is one of its own arguments: no frame, no
/// `Machine`, no chain — the value is where the caller laid it.
///
/// # Safety
/// As `Entry`, and `code`'s body is an `Expr` whose body is an `Argument`.
pub(crate) unsafe fn entry_argument(
    code: &Code,
    _f: Value,
    _rt: &AcvusRuntime,
    window: &mut FrameState,
    arity: u16,
) -> Value {
    // SAFETY: as `entry_body`, for the `Expr` `Code::expr` took this entry
    // from.
    let expr = unsafe { code.body.expr_unchecked() };
    debug_assert_eq!(
        usize::from(arity),
        expr.arity as usize,
        "an expression body is called with the arguments it reads"
    );
    // SAFETY: as above.
    let at = unsafe { expr.argument_unchecked() };
    window.laid(arity)[usize::from(at)]
}

/// # Safety
/// As `Entry`, and `code` is `code::RUST`.
pub(crate) unsafe fn entry_rust(
    _code: &Code,
    f: Value,
    rt: &AcvusRuntime,
    window: &mut FrameState,
    arity: u16,
) -> Value {
    // SAFETY: `code::rust_fn` is the one writer of a closure over `RUST`, and
    // its one capture is the `RustCallee<AcvusRuntime>` it erased.
    let callee = unsafe { f.captures_of()[0].peek::<RustCallee<AcvusRuntime>>() };
    let mut moved = [Value::UNDEF; MOST_MEMBERS];
    let run = &mut moved[..usize::from(arity)];
    run.copy_from_slice(window.laid(arity));
    let mut lent = LentWindow::lend(window, rt);
    // SAFETY: the caller laid `arity` arguments, which the checker typed at
    // the callee's parameters, and moved them to it; they are copied out of
    // the window, so the `Ctx`'s frame names no cell `run` is read from.
    unsafe { callee.call(lent.ctx(), run) }
}

struct LentWindow<'w, 'r> {
    window: &'w mut FrameState,
    ctx: std::mem::ManuallyDrop<Ctx<'r, AcvusRuntime>>,
}

impl<'w, 'r> LentWindow<'w, 'r> {
    fn lend(window: &'w mut FrameState, rt: &'r AcvusRuntime) -> Self {
        let frame = std::mem::replace(window, FrameState::UNBOUND);
        LentWindow {
            window,
            // SAFETY: the window's state moved into this `Ctx`, and the
            // caller keeps its cells live for the call, which this guard does
            // not outlive.
            ctx: std::mem::ManuallyDrop::new(unsafe { Ctx::new(rt, frame) }),
        }
    }

    fn ctx(&mut self) -> &mut Ctx<'r, AcvusRuntime> {
        &mut self.ctx
    }
}

impl Drop for LentWindow<'_, '_> {
    fn drop(&mut self) {
        // SAFETY: `drop` runs once, and nothing else takes the `Ctx`.
        let ctx = unsafe { std::mem::ManuallyDrop::take(&mut self.ctx) };
        *self.window = ctx.into_frame();
    }
}

impl Code {
    /// The arguments read into the callee's frame before the future exists,
    /// so the caller may lend registers that die at the call.
    ///
    /// An `Expr` and a Rust body have no frame to read them into and no
    /// suspension to wait for, so they run here, through the same entry a
    /// synchronous call reaches — on a window of their own, because the
    /// arguments arrive as values rather than as a run of the caller's
    /// registers.
    pub fn start<'c>(&'c self, f: Value, rt: &AcvusRuntime, args: &mut [Value]) -> Resume<'c> {
        match &self.body {
            CodeBody::Body(body) => body.start(&f, args),
            CodeBody::Expr(_) | CodeBody::Rust => Resume::Done(self.frameless_now(f, rt, args)),
        }
    }

    /// The entry of a frameless body, on a window of this call's own:
    /// `start`'s caller hands its arguments as values, and every entry reads
    /// them out of a window.
    fn frameless_now(&self, f: Value, rt: &AcvusRuntime, args: &[Value]) -> Value {
        let arity = u16::try_from(args.len()).expect("an argument run is at most one cell wide");
        let RootFrame { mut state, cells } = RootFrame::new();
        for (slot, arg) in FrameSlot::first(args.len()).zip(args) {
            state.lay(Off::bounded(slot), *arg);
        }
        // SAFETY: the contract `fn_value_call` carries — `f` is a closure of
        // this `Code` — and the arguments are laid where an entry reads them.
        let value = unsafe { self.call(f, rt, &mut state, arity) };
        drop(cells);
        value
    }
}

/// What a closure call has to do after its arguments are read: run a body on a
/// frame, or — for an `Expr` or a Rust body — nothing, because the value is
/// already produced.
pub enum Resume<'c> {
    Frame { body: &'c Body, store: Store },
    Done(Value),
}

impl Body {
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
    let resume = unsafe { f.code_of() }.code().start(*f, rt, args);
    async move {
        match resume {
            Resume::Frame { body, store } => {
                let (mut cells, rt) = FrameCells::open(store, rt);
                let (regs, _) = cells.store().bind(body);
                drive(Machine::new(body, regs, &rt)).await
            }
            Resume::Done(value) => value,
        }
    }
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
