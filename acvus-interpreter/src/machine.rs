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

use acvus_mir::graph::QualifiedRef;
use acvus_utils::Interner;
use futures::future::BoxFuture;

use crate::code::{
    Block, BlockId, Body, Code, EntryKonst, Expr, ExprBody, ExprChain, Off, Pending, Prepared,
    RETURN, SENTINEL, SUSPEND,
};
use crate::interpreter::{InterpreterContext, lookup_module};
use crate::journal::RuntimeContext;
use crate::regs::{Regs, Store};
use crate::runtime::AcvusRuntime;
use crate::value::{FnValue, Value};

/// The registers and the interner a path walk needs, borrowed apart.
pub struct Frame<'m, 'f> {
    pub regs: &'m mut Regs<'f>,
    pub interner: &'m Interner,
}

pub struct Machine<'c> {
    body: &'c Body,
    regs: Regs<'c>,
    pub rt: &'c AcvusRuntime,
    pub page: &'c Arc<dyn RuntimeContext>,
    exit: Value,
    /// The block `run` enters at. A suspension writes the block after
    /// itself here, so the driver's next `run` resumes rather than
    /// re-entering the body.
    at: BlockId,
    pending: Option<Pending>,
}

impl<'c> Machine<'c> {
    pub fn new(
        body: &'c Body,
        regs: Regs<'c>,
        rt: &'c AcvusRuntime,
        page: &'c Arc<dyn RuntimeContext>,
    ) -> Machine<'c> {
        Machine {
            body,
            regs,
            rt,
            page,
            exit: Value::unit(),
            at: body.entry,
            pending: None,
        }
    }

    /// One compare per block. The block index is unchecked because every
    /// terminator's target was resolved by `prepare` against this same
    /// `blocks` array, and the two sentinels are what the compare catches.
    pub fn run(&mut self) -> BlockId {
        let body = self.body;
        let mut at = self.at;
        loop {
            debug_assert!(
                (at as usize) < body.blocks.len(),
                "a terminator names block {at}, which this body does not have"
            );
            // SAFETY: `prepare::link` resolves every terminator target to an
            // index of this array or to a sentinel, and a sentinel leaves the
            // loop at the compare below before it is used as an index.
            let block: &Block = unsafe { body.blocks.get_unchecked(at as usize) };
            at = block.run(self);
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
        &self.rt.0
    }

    pub fn interner(&self) -> &Interner {
        &self.rt.0.interner
    }

    #[inline]
    pub fn frame(&mut self) -> Frame<'_, 'c> {
        Frame {
            regs: &mut self.regs,
            interner: &self.rt.0.interner,
        }
    }

    /// The body returns this value; the `Return` terminator leaves the loop.
    #[inline]
    pub fn finish(&mut self, value: Value) {
        self.exit = value;
    }

    /// The future the driver awaits, the slot its result lands in, and the
    /// block the body goes on at once it has (RFC-0046).
    #[inline]
    pub fn suspend<const LARGE: bool>(
        &mut self,
        dst: Off,
        resume: BlockId,
        fut: BoxFuture<'static, Value>,
    ) {
        self.at = resume;
        self.pending = Some(Pending {
            dst,
            owns_large: LARGE,
            fut,
        });
    }

    /// Run `callee` to its result in the window above this frame (RFC-0052
    /// rule 7): no allocation, one mask store to enter and one sweep to leave.
    ///
    /// # Panics
    /// `callee` is typed pure and its prepared body can suspend, which is a
    /// disagreement between the effect the checker read and the body
    /// `prepare` produced.
    pub fn call_sync<F>(&mut self, callee: &Body, named: &dyn Debug, fill: F) -> Value
    where
        F: FnOnce(&mut Machine<'_>),
    {
        if self.regs.fits_above(callee.frame_len) {
            let window = self.regs.window();
            return run_frame(callee, named, window, self.rt, self.page, fill);
        }
        run_body(callee, named, self.rt, self.page, fill)
    }

    pub fn call_fn_sync(&mut self, f: &FnValue, args: &mut [Value]) -> Value {
        match f.code.as_ref() {
            Code::Expr(expr) => expr_value(expr, args),
            Code::Body(body) => self.call_sync(body, &f.code.site(), |callee| {
                enter(body, f, args, &mut callee.regs)
            }),
        }
    }
}

/// A body's registers at entry: the captures, the arguments, the order token
/// and the entry constants, and the one mask of which of them own a `Large`.
fn open_frame(body: &Body, regs: &mut Regs<'_>) {
    for kind in &body.slot_kinds {
        regs.open(kind.slot, Value::inline(kind.kind, 0));
    }
    for EntryKonst { slot, value } in &body.entry_konsts {
        regs.open(*slot, *value);
    }
    regs.own_mask(body.param_marks);
}

fn run_body<F>(
    body: &Body,
    named: &dyn Debug,
    rt: &AcvusRuntime,
    page: &Arc<dyn RuntimeContext>,
    fill: F,
) -> Value
where
    F: FnOnce(&mut Machine<'_>),
{
    let mut store = Store::new(body.frame_len);
    let regs = store.borrow();
    run_frame(body, named, regs, rt, page, fill)
}

/// # Panics
/// `body` is typed pure and its prepared body can suspend, which is a
/// disagreement between the effect the checker read and the body `prepare`
/// produced.
fn run_frame<F>(
    body: &Body,
    named: &dyn Debug,
    mut regs: Regs<'_>,
    rt: &AcvusRuntime,
    page: &Arc<dyn RuntimeContext>,
    fill: F,
) -> Value
where
    F: FnOnce(&mut Machine<'_>),
{
    assert!(
        !body.may_suspend,
        "{named:?} is typed pure, and its prepared body can suspend"
    );
    open_frame(body, &mut regs);
    let mut machine = Machine::new(body, regs, rt, page);
    fill(&mut machine);
    let stop = machine.run();
    assert_eq!(
        stop, RETURN,
        "{named:?} is typed pure, and its body left the machine at {stop}"
    );
    let value = machine.exit;
    machine.regs.sweep();
    value
}

async fn drive(mut machine: Machine<'_>) -> Value {
    loop {
        let stop = machine.run();
        if stop == RETURN {
            let value = machine.exit;
            machine.regs.sweep();
            return value;
        }
        debug_assert_eq!(stop, SUSPEND, "a body left the machine at {stop}");
        let Pending {
            dst,
            owns_large,
            fut,
        } = machine
            .pending
            .take()
            .expect("a Suspend terminator left no future for the driver");
        let value = fut.await;
        match owns_large {
            true => machine.regs.define::<true>(dst, value),
            false => machine.regs.define::<false>(dst, value),
        }
    }
}

/// Run the entry body of the module `id` names.
pub async fn call_module(
    shared: Arc<InterpreterContext>,
    page: Arc<dyn RuntimeContext>,
    id: QualifiedRef,
    args: Vec<Value>,
) -> Value {
    let prepared: Arc<Prepared> = Arc::clone(lookup_module(&shared, &id));
    let rt = AcvusRuntime(shared);
    let Code::Body(body) = prepared.main.as_ref() else {
        panic!("a module's entry body is one chain, which no call into a module can be")
    };
    let mut store = Store::new(body.frame_len);
    let mut regs = store.borrow();
    open_frame(body, &mut regs);
    for (slot, arg) in body.params.iter().zip(args) {
        regs.define::<false>(*slot, arg);
    }
    if let Some(order) = body.order_param {
        regs.define::<false>(order, Value::unit());
    }
    drive(Machine::new(body, regs, &rt, &page)).await
}

pub fn call_module_sync(
    machine: &mut Machine<'_>,
    prepared: &Prepared,
    id: QualifiedRef,
    args: Vec<Value>,
) -> Value {
    let Code::Body(body) = prepared.main.as_ref() else {
        panic!("a module's entry body is one chain, which no call into a module can be")
    };
    machine.call_sync(body, &id, |callee| {
        for (slot, arg) in body.params.iter().zip(args) {
            callee.regs.define::<false>(*slot, arg);
        }
        if let Some(order) = body.order_param {
            callee.regs.define::<false>(order, Value::unit());
        }
    })
}

/// The arguments are read into the callee's frame before the future exists, so
/// the caller may lend registers that die at the call.
pub fn fn_value_call<'f>(
    f: &'f FnValue,
    args: &mut [Value],
) -> impl Future<Output = Value> + Send + use<'f> {
    let entry = Entry::of(&f.code, f, args);
    async move {
        match entry {
            Entry::Frame { body, mut store } => {
                let regs = store.borrow();
                let machine = Machine::new(body, regs, AcvusRuntime::of(&f.shared), &f.page);
                drive(machine).await
            }
            Entry::Done(value) => value,
        }
    }
}

/// What a closure call has to do after its arguments are read: run a body on a
/// frame, or — for a `Code::Expr` — nothing, because the chain has already
/// produced the value.
enum Entry<'c> {
    Frame { body: &'c Body, store: Store },
    Done(Value),
}

impl<'c> Entry<'c> {
    fn of(code: &'c Code, f: &FnValue, args: &mut [Value]) -> Entry<'c> {
        match code {
            Code::Expr(expr) => Entry::Done(expr_value(expr, args)),
            Code::Body(body) => {
                let mut store = Store::new(body.frame_len);
                {
                    let mut regs = store.borrow();
                    enter(body, f, args, &mut regs);
                }
                Entry::Frame { body, store }
            }
        }
    }
}

/// Run `f` to its result with no caller frame in hand. Obligation across
/// artifacts: rule 7's window comes from the calling `Machine`, and an extern
/// handler's ABI is `fn(&Rt, Value) -> Value` (RFC-0052 rule 6) — it is handed
/// the runtime, never the machine — so a closure an extern calls back roots a
/// store of its own. Widen that ABI and this function has no callers left.
pub fn fn_value_call_sync(f: &FnValue, args: &mut [Value]) -> Value {
    match f.code.as_ref() {
        Code::Expr(expr) => expr_value(expr, args),
        Code::Body(body) => run_body(
            body,
            &f.code.site(),
            AcvusRuntime::of(&f.shared),
            &f.page,
            |callee| enter(body, f, args, &mut callee.regs),
        ),
    }
}

/// The closure owns its captures; the body sees each through a reference
/// (RFC-0018).
fn enter(body: &Body, f: &FnValue, args: &mut [Value], regs: &mut Regs<'_>) {
    open_frame(body, regs);
    for (slot, capture) in body.captures.iter().zip(f.captures.iter()) {
        regs.define::<false>(*slot, Value::reference(capture));
    }
    for (slot, arg) in body.params.iter().zip(args) {
        regs.define::<false>(*slot, *arg);
    }
    if let Some(order) = body.order_param {
        regs.define::<false>(order, Value::unit());
    }
}

/// A body that is one chain runs with no registers, no `Machine` and no
/// dispatch loop (RFC-0044, stage 4).
///
/// # Panics
/// The call brought an arity the body was not prepared with.
#[inline]
fn expr_value(expr: &Expr, args: &mut [Value]) -> Value {
    assert_eq!(
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
