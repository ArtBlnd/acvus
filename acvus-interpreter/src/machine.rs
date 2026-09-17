//! The machine that runs a prepared body (RFC-0044).
//!
//! `Machine::run` is synchronous: it leaves its loop only to return, to
//! raise, or to hand a future up. `drive` is the one `async fn` around it —
//! it awaits the pending future, stores the result in the register the
//! operation named, and re-enters the loop at the next operation. A
//! synchronous operation costs one indirect call.

use std::sync::Arc;

use acvus_ast::Span;
use acvus_mir::graph::QualifiedRef;
use acvus_utils::Interner;

use crate::code::{Code, Flow, Op, Payload, Pending, Prepared};
use crate::error::RuntimeError;
use crate::interpreter::{InterpreterContext, lookup_module};
use crate::journal::RuntimeContext;
use crate::runtime::AcvusRuntime;
use crate::value::{FnValue, Value};

/// What a body left its loop with.
enum Exit {
    Value(Value),
    Error(RuntimeError),
}

/// The registers and the interner a path walk needs, borrowed apart.
pub struct Frame<'m> {
    pub regs: &'m mut [Value],
    pub interner: &'m Interner,
}

/// Where a run of the loop stopped.
pub enum Step {
    Done(Result<Value, RuntimeError>),
    Await { pc: u32, pending: Pending },
}

pub struct Machine<'c> {
    code: &'c Code,
    regs: Vec<Value>,
    pub rt: AcvusRuntime,
    pub page: Arc<dyn RuntimeContext>,
    exit: Option<Exit>,
}

impl<'c> Machine<'c> {
    pub fn new(code: &'c Code, rt: AcvusRuntime, page: Arc<dyn RuntimeContext>) -> Self {
        Self {
            code,
            regs: (0..code.frame_len).map(|_| Value::Empty).collect(),
            rt,
            page,
            exit: None,
        }
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

    #[inline]
    pub fn slot_mut(&mut self, slot: u32) -> &mut Value {
        &mut self.regs[slot as usize]
    }

    #[inline]
    pub fn frame(&mut self) -> Frame<'_> {
        Frame {
            regs: &mut self.regs,
            interner: &self.rt.0.interner,
        }
    }

    /// The body returns `value`.
    #[inline]
    pub fn finish(&mut self, value: Value) -> Flow {
        self.exit = Some(Exit::Value(value));
        Flow::Return
    }

    /// The body raises `error`; the span of the operation that raised it is
    /// attached when the loop is left.
    #[inline]
    pub fn fail(&mut self, error: RuntimeError) -> Flow {
        self.exit = Some(Exit::Error(error));
        Flow::Return
    }

    #[inline]
    pub fn attach_span(&mut self, span: Span) {
        if let Some(Exit::Error(error)) = &mut self.exit {
            error.span = error.span.or(Some(span));
        }
    }

    fn raised_at(&self, pc: u32, mut error: RuntimeError) -> RuntimeError {
        error.span = error.span.or(Some(self.code.spans[pc as usize]));
        error
    }

    fn leave(&mut self, pc: u32) -> Result<Value, RuntimeError> {
        match self.exit.take() {
            Some(Exit::Value(value)) => Ok(value),
            Some(Exit::Error(error)) => Err(self.raised_at(pc, error)),
            None => panic!("operation at {pc} left the loop with neither a value nor an error"),
        }
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
        Step::Done(Ok(Value::unit()))
    }
}

async fn drive(mut machine: Machine<'_>) -> Result<Value, RuntimeError> {
    let mut pc = 0;
    loop {
        match machine.run(pc) {
            Step::Done(outcome) => return outcome,
            Step::Await {
                pc: at,
                pending: Pending { dst, fut },
            } => {
                let value = fut.await.map_err(|e| machine.raised_at(at, e))?;

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
) -> Result<Value, RuntimeError> {
    let prepared: Arc<Prepared> = Arc::clone(lookup_module(&shared, &id));
    let rt = AcvusRuntime(shared);
    let code = &prepared.main;
    let mut machine = Machine::new(code, rt, page);
    for (slot, arg) in code.params.iter().zip(args) {
        machine.set(*slot, arg);
    }
    if let Some(order) = code.order_param {
        machine.set(order, Value::unit());
    }
    drive(machine).await
}

pub async fn fn_value_call(f: &FnValue, args: Vec<Value>) -> Result<Value, RuntimeError> {
    let code = &f.code;
    let mut machine = Machine::new(
        code,
        AcvusRuntime(Arc::clone(&f.shared)),
        Arc::clone(&f.page),
    );

    // The closure owns its captures; the body sees each through a
    // reference (RFC-0018).
    for (slot, capture) in code.captures.iter().zip(f.captures.iter()) {
        machine.set(*slot, Value::reference(capture));
    }
    for (slot, arg) in code.params.iter().zip(args) {
        machine.set(*slot, arg);
    }
    if let Some(order) = code.order_param {
        machine.set(order, Value::unit());
    }

    drive(machine).await
}
