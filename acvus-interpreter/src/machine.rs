//! The machine that runs a prepared body (RFC-0044).
//!
//! `Machine::run` is synchronous: it leaves its loop only to return or to
//! hand a future up. `drive` is the one `async fn` around it — it awaits
//! the pending future, stores the result in the register the operation
//! named, and re-enters the loop at the next operation. A synchronous
//! operation costs one indirect call. A run-time failure is a panic and
//! leaves through the unwinder, not through this loop.

use std::cell::RefCell;
use std::fmt::Debug;
use std::future::Future;
use std::sync::Arc;

use acvus_ast::Span;
use acvus_mir::graph::QualifiedRef;
use acvus_utils::Interner;

use crate::code::{ArgWindow, Code, Flow, Op, Payload, Pending, Prepared};
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

/// The register files a run lends to the bodies it calls synchronously,
/// kept for reuse so a call after the first allocates nothing.
///
/// A run does not keep its frames in one growing buffer.
/// `storage::ref_var` puts a raw pointer to a register in a `Value::Ref`,
/// and a buffer that reallocated to make room for a callee would move
/// every frame below it and leave those references dangling. Each frame is
/// its own allocation, so pushing one moves nothing.
#[derive(Default)]
pub struct Frames(Vec<Vec<Value>>);

impl Frames {
    fn take(&mut self, len: u32) -> Vec<Value> {
        let mut regs = self.0.pop().unwrap_or_default();
        regs.resize_with(len as usize, || Value::Empty);
        regs
    }

    fn give(&mut self, mut regs: Vec<Value>) {
        regs.clear();
        self.0.push(regs);
    }
}

thread_local! {
    /// An extern's callback enters a body with no caller frame to grow
    /// from, so it borrows this thread's frames instead. It is taken out
    /// for the duration of the call, which is what lets a body reached
    /// this way reach another one the same way.
    static CALLBACK_FRAMES: RefCell<Frames> = RefCell::new(Frames::default());
}

pub struct Machine<'c> {
    code: &'c Code,
    regs: &'c mut [Value],
    frames: &'c mut Frames,
    pub rt: &'c AcvusRuntime,
    pub page: &'c Arc<dyn RuntimeContext>,
    exit: Option<Value>,
}

impl<'c> Machine<'c> {
    pub fn new(
        code: &'c Code,
        regs: &'c mut [Value],
        frames: &'c mut Frames,
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
            frames,
            rt,
            page,
            exit: None,
        }
    }

    /// Run `code` to its result on a frame of this run's own. The
    /// preparation chose this path from the callee's effect; `callee`
    /// names the body if the effect and the prepared code disagree.
    pub fn call_sync<F>(&mut self, code: &Code, callee: &dyn Debug, fill: F) -> Value
    where
        F: FnOnce(&mut Machine<'_>),
    {
        let Machine {
            frames, rt, page, ..
        } = self;
        run_sync(code, callee, frames, rt, page, fill)
    }

    pub fn code(&self) -> &'c Code {
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
    code: &Code,
    callee: &dyn Debug,
    frames: &mut Frames,
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
    let mut regs = frames.take(code.frame_len);
    let value = {
        let mut machine = Machine::new(code, &mut regs, frames, rt, page);
        fill(&mut machine);
        match machine.run(0) {
            Step::Done(value) => value,
            Step::Await { pc, .. } => {
                unreachable!("{callee:?} is typed pure, and its operation {pc} handed up a future")
            }
        }
    };

    frames.give(regs);
    value
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
    let code = &prepared.main;
    let mut frames = Frames::default();
    let mut regs = frames.take(code.frame_len);
    let mut machine = Machine::new(code, &mut regs, &mut frames, &rt, &page);
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
    let code = &prepared.main;
    let Machine {
        frames, rt, page, ..
    } = machine;
    run_sync(code, &id, frames, rt, page, |callee| {
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
    let code = &f.code;
    let mut entered: Vec<Value> = (0..code.frame_len).map(|_| Value::Empty).collect();
    enter(code, f, args, &mut entered);

    async move {
        let mut frames = Frames::default();
        let machine = Machine::new(
            code,
            &mut entered,
            &mut frames,
            AcvusRuntime::of(&f.shared),
            &f.page,
        );
        drive(machine).await
    }
}

/// Run `f` to its result on the caller's frames. The preparation chose
/// this path from the closure's effect.
pub fn fn_value_call_sync(machine: &mut Machine<'_>, f: &FnValue, args: &mut [Value]) -> Value {
    let code = &f.code;
    run_sync(
        code,
        &closure_site(code),
        machine.frames,
        AcvusRuntime::of(&f.shared),
        &f.page,
        |callee| enter(code, f, args, callee.regs),
    )
}

pub fn fn_value_call_now(f: &FnValue, args: &mut [Value]) -> Value {
    let code = &f.code;
    let mut frames = CALLBACK_FRAMES.with(|held| held.take());
    let value = run_sync(
        code,
        &closure_site(code),
        &mut frames,
        AcvusRuntime::of(&f.shared),
        &f.page,
        |callee| enter(code, f, args, callee.regs),
    );
    CALLBACK_FRAMES.with(|held| held.replace(frames));
    value
}

/// The span of a closure body's first operation, which is what an ICE
/// about that body has to name it by: a `Code` carries no name.
fn closure_site(code: &Code) -> Span {
    code.spans.first().copied().unwrap_or(Span::ZERO)
}

/// The closure owns its captures; the body sees each through a reference
/// (RFC-0018).
fn enter(code: &Code, f: &FnValue, args: &mut [Value], regs: &mut [Value]) {
    for (slot, capture) in code.captures.iter().zip(f.captures.iter()) {
        regs[*slot as usize] = Value::reference(capture);
    }
    for (slot, arg) in code.params.iter().zip(args) {
        regs[*slot as usize] = arg.take();
    }
    if let Some(order) = code.order_param {
        regs[order as usize] = Value::unit();
    }
}
