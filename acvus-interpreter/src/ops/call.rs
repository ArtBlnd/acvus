//! Calls: the extern instances, the named functions, the closures, the
//! spawns, and the closure constructor.
//!
//! A synchronous extern runs inside the loop. Everything that can suspend —
//! an asynchronous extern, a call into another body, an `Eval` — hands a
//! `'static` future up to the driver.

use std::sync::Arc;

use futures::future::BoxFuture;

use crate::code::{ArgWindow, Flow, NO_SLOT, Op, Payload, Pending};
use crate::machine::{Machine, call_module, fn_value_call};
use crate::ops::control::move_all;
use crate::ops::payload;
use crate::runtime::ExternHandler;
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
        machine.set(slot, Value::unit());
    }
}

fn extern_payload<'c>(machine: &Machine<'c>, op: &Op) -> (&'c ExternHandler, &'c ArgWindow) {
    match machine.payload(op) {
        Payload::Extern { handler, window } => (handler, window),
        other => panic!(
            "a call to an extern instance wants an Extern payload, found {}",
            crate::code::payload_name(other)
        ),
    }
}

pub fn call_extern_sync(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let (handler, window) = extern_payload(machine, op);
    let ExternHandler::Sync(f) = handler else {
        panic!("prepared as a synchronous extern call, but the handler is asynchronous")
    };
    yield_order(machine, op.d);
    move_all(machine, &window.moves);
    let (rt, args) = machine.lend_window(window);
    match f(rt, args) {
        Ok(value) => {
            machine.set(op.a, value);
            Flow::Next
        }
        Err(error) => machine.fail(error),
    }
}

pub fn call_extern_async(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let (handler, window) = extern_payload(machine, op);
    let ExternHandler::Async(f) = handler else {
        panic!("prepared as an asynchronous extern call, but the handler is synchronous")
    };
    yield_order(machine, op.d);
    move_all(machine, &window.moves);
    let rt = machine.rt.clone();
    let fut = f(rt, machine.window(window));
    Flow::Await(Pending { dst: op.a, fut })
}

pub fn call_direct(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let Payload::Direct { callee, args } = machine.payload(op) else {
        panic!("a direct call wants a Direct payload")
    };
    yield_order(machine, op.d);
    let args = arg_values(machine, args);
    let shared = Arc::clone(machine.shared());
    let page = Arc::clone(&machine.page);
    Flow::Await(Pending {
        dst: op.a,
        fut: Box::pin(call_module(shared, page, *callee, args)),
    })
}

/// `THROUGH` is what the preparation read from the callee register's type:
/// a reference names a closure the caller keeps, a value is one this call
/// consumes.
pub fn call_indirect<const THROUGH: bool>(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let slots = payload!(machine, op, Slots);
    yield_order(machine, op.d);
    let mut args = arg_values(machine, slots);
    let fut: BoxFuture<'static, _> = if THROUGH {
        // SAFETY: the type checker admits only a live reference to a closure
        // here; the closure's register is not written during the call, and
        // the machine holding it outlives the future the driver awaits.
        let closure: &'static FnValue =
            unsafe { &*(machine.reg(op.b).target().as_fn() as *const FnValue) };
        Box::pin(fn_value_call(closure, &mut args))
    } else {
        // SAFETY: the type checker admits only a closure value here.
        let closure = unsafe { machine.take(op.b).materialize::<FnValue>() };
        Box::pin(async move { fn_value_call(&closure, &mut args).await })
    };
    Flow::Await(Pending { dst: op.a, fut })
}

pub fn spawn_extern_sync(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let (handler, window) = extern_payload(machine, op);
    let ExternHandler::Sync(f) = handler else {
        panic!("prepared as a synchronous extern spawn, but the handler is asynchronous")
    };
    move_all(machine, &window.moves);
    let mut args = machine.take_window(window);
    let rt = machine.rt.clone();
    let f = Arc::clone(f);
    let handle = machine
        .shared()
        .executor
        .spawn_blocking(Box::new(move || f(&rt, &mut args)));
    machine.set(op.a, Value::handle(handle));
    Flow::Next
}

pub fn spawn_extern_async(machine: &mut Machine<'_>, op: &Op) -> Flow {
    let (handler, window) = extern_payload(machine, op);
    let ExternHandler::Async(f) = handler else {
        panic!("prepared as an asynchronous extern spawn, but the handler is synchronous")
    };
    move_all(machine, &window.moves);
    let mut args = machine.take_window(window);
    let rt = machine.rt.clone();
    let f = Arc::clone(f);
    let handle = machine
        .shared()
        .executor
        .spawn_async(Box::pin(async move { f(rt, &mut args).await }));
    machine.set(op.a, Value::handle(handle));
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
        Arc::clone(&machine.page),
        args,
    );
    let handle = machine.shared().executor.spawn_interpreter(child);
    machine.set(op.a, Value::handle(handle));
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
        page: Arc::clone(&machine.page),
        code: Arc::clone(code),
        captures: captured.into(),
    };
    machine.set(op.a, Value::closure(closure));
    Flow::Next
}
