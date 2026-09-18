//! The machine that runs a prepared body (RFC-0044).
//!
//! `Machine::run` is synchronous: it leaves its loop only to return or to
//! hand a future up. `drive` is the one `async fn` around it — it awaits
//! the pending future, stores the result in the register the operation
//! named, and re-enters the loop at the next operation. A synchronous
//! operation costs one indirect call. A run-time failure is a panic and
//! leaves through the unwinder, not through this loop.

use std::fmt::Debug;
use std::future::Future;
use std::mem::MaybeUninit;
use std::slice;
use std::sync::Arc;

use acvus_mir::graph::QualifiedRef;
use acvus_utils::Interner;

use crate::code::{
    ArgWindow, Body, Code, Expr, ExprBody, ExprChain, Flow, Op, Payload, Pending, Prepared,
};
use crate::interpreter::{InterpreterContext, lookup_module};
use crate::journal::RuntimeContext;
use crate::runtime::AcvusRuntime;
use crate::value::{FnValue, Value};

/// The registers and the interner a path walk needs, borrowed apart.
pub struct Frame<'m> {
    pub regs: &'m mut [Value],
    pub interner: &'m Interner,
}

/// Where a run of the loop stopped.
pub enum Step {
    Done(Value),
    Await { pc: u32, pending: Pending },
}

/// A synchronous call's registers live on the Rust stack of the call, as
/// a Rust function's locals do. `storage::ref_var` puts a raw pointer to a
/// register in a `Value::Ref`, so a frame must never move while its body
/// runs; a stack array does not, and neither does the heap array a body
/// wider than `INLINE` registers takes instead.
enum Registers {
    Stack([Value; Registers::INLINE]),
    Heap(Vec<Value>),
}

impl Registers {
    const INLINE: usize = 16;

    fn new(len: u32) -> Self {
        let len = len as usize;
        if len > Self::INLINE {
            return Self::Heap((0..len).map(|_| Value::EMPTY).collect());
        }
        const EMPTY: Value = Value::EMPTY;
        Self::Stack([EMPTY; Registers::INLINE])
    }

    fn regs(&mut self, len: u32) -> &mut [Value] {
        match self {
            Self::Stack(slots) => &mut slots[..len as usize],
            Self::Heap(regs) => regs,
        }
    }
}

pub struct Machine<'c> {
    code: &'c Body,
    regs: &'c mut [Value],
    pub rt: &'c AcvusRuntime,
    pub page: &'c Arc<dyn RuntimeContext>,
    exit: Option<Value>,
}

impl<'c> Machine<'c> {
    pub fn new(
        code: &'c Body,
        regs: &'c mut [Value],
        rt: &'c AcvusRuntime,
        page: &'c Arc<dyn RuntimeContext>,
    ) -> Self {
        assert_eq!(
            regs.len(),
            code.frame_len as usize,
            "a body runs on a frame of its own length"
        );
        Self {
            code,
            regs,
            rt,
            page,
            exit: None,
        }
    }

    /// Run `code` to its result on a frame of its own. The preparation
    /// chose this path from the callee's effect; `callee` names the body
    /// if the effect and the prepared code disagree.
    pub fn call_sync<F>(&mut self, code: &Body, callee: &dyn Debug, fill: F) -> Value
    where
        F: FnOnce(&mut Machine<'_>),
    {
        run_sync(code, callee, self.rt, self.page, fill)
    }

    pub fn code(&self) -> &'c Body {
        self.code
    }

    pub fn payload(&self, op: &Op) -> &'c Payload {
        &self.code.payloads[op.p]
    }

    pub fn shared(&self) -> &Arc<InterpreterContext> {
        &self.rt.0
    }

    pub fn interner(&self) -> &Interner {
        &self.rt.0.interner
    }

    #[inline]
    pub fn reg(&self, slot: u32) -> &Value {
        let value = &self.regs[slot as usize];
        assert!(!value.is_empty(), "read of register {slot}: already moved");
        value
    }

    #[inline]
    pub fn set(&mut self, slot: u32, value: Value) {
        self.regs[slot as usize] = value;
    }

    #[inline(always)]
    pub fn store(&mut self, slot: u32, value: Value) {
        debug_assert!(
            (slot as usize) < self.regs.len(),
            "a chain writes register {slot}, which its body's frame does not have"
        );
        // SAFETY: `prepare::check_assignment` states that every slot an
        // operation names is below the body's `frame_len`, and
        // `Machine::new` asserts the frame it runs on has that length.
        unsafe {
            *self.regs.get_unchecked_mut(slot as usize) = value;
        }
    }

    #[inline]
    pub fn take(&mut self, slot: u32) -> Value {
        let value = self.regs[slot as usize].take();
        assert!(!value.is_empty(), "take of register {slot}: already moved");
        value
    }

    /// A word is copied; a `Large` value leaves its register (RFC-0018).
    #[inline]
    pub fn use_val(&mut self, slot: u32) -> Value {
        Value::use_from(&mut self.regs[slot as usize])
    }

    /// The registers an extern call's arguments are in, lent with the
    /// runtime the handler is called with.
    #[inline]
    pub fn lend_window(&mut self, window: &ArgWindow) -> (&AcvusRuntime, &mut [Value]) {
        let Machine { rt, regs, .. } = self;
        (rt, &mut regs[run_of(window)])
    }

    #[inline]
    pub fn window(&mut self, window: &ArgWindow) -> &mut [Value] {
        &mut self.regs[run_of(window)]
    }

    /// The window's values, owned: what a spawn hands to work that
    /// outlives this frame.
    #[inline]
    pub fn take_window(&mut self, window: &ArgWindow) -> Vec<Value> {
        self.window(window).iter_mut().map(Value::take).collect()
    }

    /// The frame itself, which a chain's `Push` micro-ops read.
    #[inline]
    pub fn regs(&self) -> &[Value] {
        self.regs
    }

    #[inline]
    pub fn slot_mut(&mut self, slot: u32) -> &mut Value {
        &mut self.regs[slot as usize]
    }

    #[inline]
    pub fn frame(&mut self) -> Frame<'_> {
        Frame {
            regs: &mut self.regs[..],
            interner: &self.rt.0.interner,
        }
    }

    /// The body returns `value`.
    #[inline]
    pub fn finish(&mut self, value: Value) -> Flow {
        self.exit = Some(value);
        Flow::Return
    }

    fn leave(&mut self, pc: u32) -> Value {
        self.exit
            .take()
            .unwrap_or_else(|| panic!("operation at {pc} returned without leaving a value"))
    }

    pub fn run(&mut self, mut pc: u32) -> Step {
        let code = self.code;
        while (pc as usize) < code.ops.len() {
            let op = &code.ops[pc as usize];
            match (op.f)(self, op) {
                Flow::Next => pc += 1,
                Flow::Jump(target) => pc = target,
                Flow::Return => return Step::Done(self.leave(pc)),
                Flow::Await(pending) => return Step::Await { pc, pending },
            }
        }
        Step::Done(Value::unit())
    }
}

fn run_of(window: &ArgWindow) -> std::ops::Range<usize> {
    window.at as usize..(window.at + window.arity) as usize
}

fn run_sync<F>(
    code: &Body,
    callee: &dyn Debug,
    rt: &AcvusRuntime,
    page: &Arc<dyn RuntimeContext>,
    fill: F,
) -> Value
where
    F: FnOnce(&mut Machine<'_>),
{
    assert!(
        !code.may_suspend,
        "{callee:?} is typed pure, and its prepared body can suspend"
    );
    let mut registers = Registers::new(code.frame_len);
    let mut machine = Machine::new(code, registers.regs(code.frame_len), rt, page);
    fill(&mut machine);
    match machine.run(0) {
        Step::Done(value) => value,
        Step::Await { pc, .. } => {
            unreachable!("{callee:?} is typed pure, and its operation {pc} handed up a future")
        }
    }
}

async fn drive(mut machine: Machine<'_>) -> Value {
    let mut pc = 0;
    loop {
        match machine.run(pc) {
            Step::Done(value) => return value,
            Step::Await {
                pc: at,
                pending: Pending { dst, fut },
            } => {
                let value = fut.await;

                machine.set(dst, value);
                pc = at + 1;
            }
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
    let Code::Body(code) = prepared.main.as_ref() else {
        panic!("a module's entry body is one chain, which no call into a module can be")
    };
    let mut regs: Vec<Value> = (0..code.frame_len).map(|_| Value::EMPTY).collect();
    fill_entry_konsts(code, &mut regs);
    let mut machine = Machine::new(code, &mut regs, &rt, &page);
    for (slot, arg) in code.params.iter().zip(args) {
        machine.set(*slot, arg);
    }
    if let Some(order) = code.order_param {
        machine.set(order, Value::unit());
    }
    drive(machine).await
}

pub fn call_module_sync(
    machine: &mut Machine<'_>,
    prepared: &Prepared,
    id: QualifiedRef,
    args: Vec<Value>,
) -> Value {
    let Code::Body(code) = prepared.main.as_ref() else {
        panic!("a module's entry body is one chain, which no call into a module can be")
    };
    run_sync(code, &id, machine.rt, machine.page, |callee| {
        fill_entry_konsts(code, callee.regs);
        for (slot, arg) in code.params.iter().zip(args) {
            callee.set(*slot, arg);
        }
        if let Some(order) = code.order_param {
            callee.set(order, Value::unit());
        }
    })
}

/// The arguments are read into the callee's frame before the future
/// exists, so the caller may lend registers that die at the call.
pub fn fn_value_call<'f>(
    f: &'f FnValue,
    args: &mut [Value],
) -> impl Future<Output = Value> + Send + use<'f> {
    let entry = Entry::of(&f.code, f, args);
    async move {
        match entry {
            Entry::Frame { code, mut regs } => {
                let machine = Machine::new(code, &mut regs, AcvusRuntime::of(&f.shared), &f.page);
                drive(machine).await
            }
            Entry::Done(value) => value,
        }
    }
}

/// What a closure call has to do after its arguments are read: run a body
/// on a frame, or — for a `Code::Expr` — nothing, because the chain has
/// already produced the value.
enum Entry<'c> {
    Frame { code: &'c Body, regs: Vec<Value> },
    Done(Value),
}

impl<'c> Entry<'c> {
    fn of(code: &'c Code, f: &FnValue, args: &mut [Value]) -> Self {
        match code {
            Code::Body(body) => {
                let mut regs: Vec<Value> = (0..body.frame_len).map(|_| Value::EMPTY).collect();
                enter(body, f, args, &mut regs);
                Entry::Frame { code: body, regs }
            }
            Code::Expr(expr) => Entry::Done(expr_value(expr, args)),
        }
    }
}

/// Run `f` to its result on a frame of its own. The preparation chose
/// this path from the closure's effect; an extern's callback takes the
/// same path with no machine in hand.
pub fn fn_value_call_sync(f: &FnValue, args: &mut [Value]) -> Value {
    match f.code.as_ref() {
        Code::Expr(expr) => expr_value(expr, args),
        Code::Body(body) => run_sync(
            body,
            &f.code.site(),
            AcvusRuntime::of(&f.shared),
            &f.page,
            |callee| enter(body, f, args, callee.regs),
        ),
    }
}

/// A body that is one chain runs with no registers, no `Machine` and no
/// dispatch loop (RFC-0044, stage 4).
#[inline]
fn expr_value(expr: &Expr, args: &mut [Value]) -> Value {
    assert_eq!(
        args.len(),
        expr.arity as usize,
        "an expression body is called with the arguments it reads"
    );
    match &expr.body {
        ExprBody::Argument(at) => args[*at as usize].take(),
        ExprBody::Chain(body) => chain_value(body, args),
    }
}

/// The `#[inline(never)]` is a measurement, not a taste. The operand space
/// is `MAX_OPERANDS` `Value`s wide; while this was inlined, its frame was
/// part of every frameless call, and `map(|x| -> x) | sum` — which builds
/// no operand space at all — measured 4.8 ns per iteration instead of 4.1,
/// and `range | sum` 2.0 instead of 1.8. Splitting it restored both.
#[inline(never)]
fn chain_value(body: &ExprChain, args: &[Value]) -> Value {
    let space = OperandSpace::of(args, &body.konsts);
    (body.eval)(&body.chain, space.as_slice())
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
            slot.write(word_or_unread(value));
        }
        OperandSpace { values, len }
    }

    fn as_slice(&self) -> &[Value] {
        // SAFETY: `prepare::expression_body` refuses a body whose operands
        // outnumber `ExprChain::MAX_OPERANDS`, and `expr_value` asserts the
        // call brought the arity that body was prepared with, so `of` wrote
        // exactly `len` values into an array that holds them. Every one is
        // a word that owns nothing, so the prefix needs no drop.
        unsafe { slice::from_raw_parts(self.values.as_ptr().cast::<Value>(), self.len) }
    }
}

fn word_or_unread(value: &Value) -> Value {
    if value.kind().is_inline() {
        return value.copy_word();
    }
    Value::EMPTY
}

/// The closure owns its captures; the body sees each through a reference
/// (RFC-0018).
fn enter(code: &Body, f: &FnValue, args: &mut [Value], regs: &mut [Value]) {
    for (slot, capture) in code.captures.iter().zip(f.captures.iter()) {
        regs[*slot as usize] = Value::reference(capture);
    }
    for (slot, arg) in code.params.iter().zip(args) {
        regs[*slot as usize] = arg.take();
    }
    if let Some(order) = code.order_param {
        regs[order as usize] = Value::unit();
    }
    fill_entry_konsts(code, regs);
}

fn fill_entry_konsts(code: &Body, regs: &mut [Value]) {
    for konst in &code.entry_konsts {
        regs[konst.slot as usize] = konst.value.copy_word();
    }
}
