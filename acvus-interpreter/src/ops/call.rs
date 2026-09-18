//! Calls: the extern instances, the named functions, the closures, the
//! spawns, and the closure constructor.
//!
//! A synchronous extern runs inside the loop. Everything that can suspend —
//! an asynchronous extern, a call into another body, an `Eval` — hands a
//! `'static` future up to the driver.

use std::sync::Arc;

use futures::future::BoxFuture;

use crate::code::{
    ArgWindow, ExternArgs, ExternCall, Flow, FusedCall, NO_SLOT, Op, OpFn, PREVIOUS, Payload,
    Pending,
};
use crate::interpreter::lookup_module;
use crate::machine::{Machine, call_module, call_module_sync, fn_value_call, fn_value_call_sync};
use crate::ops::control::move_all;
use crate::ops::payload;
use crate::runtime::{ExternHandler, SyncHandler};
use crate::value::{FnValue, HandleValue, Value};

/// The call's arguments, moved out of their registers. A body is entered
/// with the arguments it owns, so a call into one stages them; an extern
/// handler is lent its window instead.
fn arg_values(machine: &mut Machine<'_>, slots: &[u32]) -> Vec<Value> {
    slots.iter().map(|slot| machine.use_val(*slot)).collect()
}

/// The `Order` an effectful call yields is the unit value (RFC-0007); a
/// Pure call names no order register.
#[inline]
fn yield_order(machine: &mut Machine<'_>, slot: u32) {
    if slot != NO_SLOT {
        machine.define(slot, Value::unit());
    }
}

#[inline]
fn extern_call<'c>(machine: &Machine<'c>, op: &Op) -> &'c ExternCall {
    let Payload::Extern(call) = machine.payload(op) else {
        panic!(
            "a call to an extern instance wants an Extern payload, found {}",
            crate::code::payload_name(machine.payload(op))
        )
    };
    call
}

#[inline]
fn window_of<'c>(call: &'c ExternCall) -> &'c ArgWindow {
    let ExternArgs::Window(window) = &call.args else {
        panic!("this extern call site passes its arguments by value")
    };
    window
}

/// RFC-0044, stage 2c.
pub fn call_extern_0(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let call = extern_call(machine, op);
    let ExternHandler::Sync(SyncHandler::Arity0(f)) = &call.handler else {
        panic!("prepared as an arity-0 extern call, but the handler is not one")
    };
    yield_order(machine, call.order);
    let value = f(&machine.rt);
    machine.define(op.a, value);
    Flow::Next
}

pub fn call_extern_1(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let call = extern_call(machine, op);
    let ExternHandler::Sync(SyncHandler::Arity1(f)) = &call.handler else {
        panic!("prepared as an arity-1 extern call, but the handler is not one")
    };
    yield_order(machine, call.order);
    let a0 = machine.use_val(op.b);
    let value = f(&machine.rt, a0);
    machine.define(op.a, value);
    Flow::Next
}

pub fn call_extern_2(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let call = extern_call(machine, op);
    let ExternHandler::Sync(SyncHandler::Arity2(f)) = &call.handler else {
        panic!("prepared as an arity-2 extern call, but the handler is not one")
    };
    yield_order(machine, call.order);
    let a0 = machine.use_val(op.b);
    let a1 = machine.use_val(op.c);
    let value = f(&machine.rt, a0, a1);
    machine.define(op.a, value);
    Flow::Next
}

pub fn call_extern_3(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let call = extern_call(machine, op);
    let ExternHandler::Sync(SyncHandler::Arity3(f)) = &call.handler else {
        panic!("prepared as an arity-3 extern call, but the handler is not one")
    };
    yield_order(machine, call.order);
    let a0 = machine.use_val(op.b);
    let a1 = machine.use_val(op.c);
    let a2 = machine.use_val(op.d);
    let value = f(&machine.rt, a0, a1, a2);
    machine.define(op.a, value);
    Flow::Next
}

pub fn call_extern_n(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let call = extern_call(machine, op);
    let ExternHandler::Sync(SyncHandler::ArityN(f)) = &call.handler else {
        panic!("prepared as a window extern call, but the handler is not one")
    };
    let window = window_of(call);
    yield_order(machine, call.order);
    move_all(machine, &window.moves);
    let (rt, args) = machine.lend_window(window);
    let value = f(rt, args);
    machine.define(op.a, value);
    Flow::Next
}

#[inline]
fn fused_arg(machine: &mut Machine<'_>, held: &mut Value, slot: u32) -> Value {
    if slot == PREVIOUS {
        return held.take();
    }
    machine.use_val(slot)
}

fn fused_call(machine: &mut Machine<'_>, call: &FusedCall, held: &mut Value) -> Value {
    let rt = machine.rt;
    match &call.handler {
        SyncHandler::Arity0(f) => f(rt),
        SyncHandler::Arity1(f) => {
            let a0 = fused_arg(machine, held, call.args[0]);
            f(rt, a0)
        }
        SyncHandler::Arity2(f) => {
            let a0 = fused_arg(machine, held, call.args[0]);
            let a1 = fused_arg(machine, held, call.args[1]);
            f(rt, a0, a1)
        }
        SyncHandler::Arity3(_) | SyncHandler::ArityN(_) => {
            panic!("a fused run holds a call the fusion rule does not admit")
        }
    }
}

/// RFC-0044, stage 6.
///
/// `CALLS` and `TAIL` are the run's shape, which the preparation resolved
/// like every other static fact: a body of this instance holds no loop
/// bound and no test the payload would have to answer.
pub fn fused<const CALLS: usize, const TAIL: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let Payload::Fused(run) = machine.payload(op) else {
        panic!(
            "a fused run wants a Fused payload, found {}",
            crate::code::payload_name(machine.payload(op))
        )
    };
    debug_assert_eq!(run.calls.len(), CALLS, "a fused instance of another length");
    let mut held = Value::EMPTY;
    for call in &run.calls[..CALLS] {
        held = fused_call(machine, call, &mut held);
    }
    let value = if TAIL {
        let Some(read) = run.tail else {
            panic!("a fused instance with a tail holds none")
        };
        read(&held)
    } else {
        held
    };
    machine.define(op.a, value);
    Flow::Next
}

/// The instance a run of `calls` calls, with or without a deref, runs as.
/// `prepare::FusedRegion` builds no other shape: a lone call with no deref
/// is not a run, and the recognizer stops at `MAX_CALLS`.
pub fn fused_instance(calls: usize, tail: bool) -> OpFn {
    match (calls, tail) {
        (1, true) => fused::<1, true>,
        (2, true) => fused::<2, true>,
        (3, true) => fused::<3, true>,
        (2, false) => fused::<2, false>,
        (3, false) => fused::<3, false>,
        (1, false) => panic!("a lone call with no deref is not a fused run"),
        (calls, _) => panic!("a fused run of {calls} calls has no instance"),
    }
}

/// The longest run `fused_instance` holds an instance for.
pub const MAX_CALLS: usize = 3;

pub fn call_extern_async(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let call = extern_call(machine, op);
    let ExternHandler::Async(f) = &call.handler else {
        panic!("prepared as an asynchronous extern call, but the handler is synchronous")
    };
    let window = window_of(call);
    yield_order(machine, call.order);
    move_all(machine, &window.moves);
    let rt = machine.rt.clone();
    let fut = f(rt, machine.window(window));
    Flow::Await(Pending { dst: op.a, fut })
}

/// A call into another body runs to its result here when that body's
/// prepared `Code` cannot suspend, and hands a future up when it can.
pub fn call_direct(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let Payload::Direct { callee, args } = machine.payload(op) else {
        panic!("a direct call wants a Direct payload")
    };
    let callee = *callee;
    yield_order(machine, op.d);
    let args = arg_values(machine, args);
    let prepared = Arc::clone(lookup_module(machine.shared(), &callee));
    if prepared.main.may_suspend() {
        let shared = Arc::clone(machine.shared());
        let page = Arc::clone(machine.page);
        return Flow::Await(Pending {
            dst: op.a,
            fut: Box::pin(call_module(shared, page, callee, args)),
        });
    }

    let value = call_module_sync(machine, &prepared, callee, args);
    machine.define(op.a, value);
    Flow::Next
}

/// `THROUGH` is what the preparation read from the callee register's type:
/// a reference names a closure the caller keeps, a value is one this call
/// consumes. As `call_direct`, the closure's own `Code` decides whether
/// this runs to a result here or hands a future up.
pub fn call_indirect<const THROUGH: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let slots = payload!(machine, op, Slots);
    yield_order(machine, op.d);
    let mut args = arg_values(machine, slots);
    if THROUGH {
        // SAFETY: the type checker admits only a live reference to a closure
        // here; the closure's register is not written during the call, and
        // the machine holding it outlives the future the driver awaits.
        let closure: &'static FnValue =
            unsafe { &*(machine.reg(op.b).target().as_fn() as *const FnValue) };
        if closure.code.may_suspend() {
            let fut: BoxFuture<'static, _> = Box::pin(fn_value_call(closure, &mut args));
            return Flow::Await(Pending { dst: op.a, fut });
        }

        let value = fn_value_call_sync(closure, &mut args);
        machine.define(op.a, value);
        return Flow::Next;
    }

    // SAFETY: the type checker admits only a closure value here.
    let closure = unsafe { machine.take(op.b).materialize::<FnValue>() };
    if closure.code.may_suspend() {
        let fut: BoxFuture<'static, _> =
            Box::pin(async move { fn_value_call(&closure, &mut args).await });
        return Flow::Await(Pending { dst: op.a, fut });
    }

    let value = fn_value_call_sync(&closure, &mut args);
    machine.define(op.a, value);
    Flow::Next
}

/// A spawn's work outlives this frame, so it owns its arguments rather
/// than borrowing registers; it keeps the window at every arity.
pub fn spawn_extern_sync(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let call = extern_call(machine, op);
    let ExternHandler::Sync(f) = &call.handler else {
        panic!("prepared as a synchronous extern spawn, but the handler is asynchronous")
    };
    let f = f.clone();
    let window = window_of(call);
    move_all(machine, &window.moves);
    let mut args = machine.take_window(window);
    let rt = machine.rt.clone();
    let handle = machine
        .shared()
        .executor
        .spawn_blocking(Box::new(move || f.call_taking(&rt, &mut args)));
    machine.define(op.a, Value::handle(handle));
    Flow::Next
}

pub fn spawn_extern_async(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let call = extern_call(machine, op);
    let ExternHandler::Async(f) = &call.handler else {
        panic!("prepared as an asynchronous extern spawn, but the handler is synchronous")
    };
    let window = window_of(call);
    move_all(machine, &window.moves);
    let mut args = machine.take_window(window);
    let rt = machine.rt.clone();
    let f = Arc::clone(f);
    let handle = machine.shared().executor.spawn_async(f(rt, &mut args));
    machine.define(op.a, Value::handle(handle));
    Flow::Next
}

pub fn spawn_module(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let Payload::Direct { callee, args } = machine.payload(op) else {
        panic!("a module spawn wants a Direct payload")
    };
    let args = arg_values(machine, args);
    let child = crate::interpreter::Interpreter::spawned(
        Arc::clone(machine.shared()),
        *callee,
        Arc::clone(machine.page),
        args,
    );
    let handle = machine.shared().executor.spawn_interpreter(child);
    machine.define(op.a, Value::handle(handle));
    Flow::Next
}

pub fn eval(machine: &mut Machine<'_>, op: &Op) -> Flow {
    yield_order(machine, op.d);
    // SAFETY: the type checker admits only a handle value here.
    let handle = unsafe { machine.take(op.b).materialize::<HandleValue>() };
    let executor = Arc::clone(&machine.shared().executor);
    Flow::Await(Pending {
        dst: op.a,
        fut: Box::pin(async move { executor.eval(handle).await }),
    })
}

pub fn make_closure(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let Payload::Closure { code, captures } = machine.payload(op) else {
        panic!("a closure constructor wants a Closure payload")
    };
    let captured: Vec<Value> = captures.iter().map(|slot| machine.use_val(*slot)).collect();
    let closure = FnValue {
        shared: Arc::clone(machine.shared()),
        page: Arc::clone(machine.page),
        code: Arc::clone(code),
        captures: captured.into(),
    };
    machine.define(op.a, Value::closure(closure));
    Flow::Next
}
