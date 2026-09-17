//! Calls: the extern instances, the named functions, the closures, the
//! spawns, and the closure constructor.
//!
//! A synchronous extern runs inside the loop. Everything that can suspend —
//! an asynchronous extern, a call into another body, an `Eval` — hands a
//! `'static` future up to the driver.

use std::sync::Arc;

use futures::future::BoxFuture;

use crate::code::{ArgWindow, ExternArgs, ExternCall, Flow, NO_SLOT, Op, Payload, Pending};
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
        machine.set(slot, Value::unit());
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
    machine.set(op.a, value);
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
    machine.set(op.a, value);
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
    machine.set(op.a, value);
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
    machine.set(op.a, value);
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
    machine.set(op.a, value);
    Flow::Next
}

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
    if prepared.main.may_suspend {
        let shared = Arc::clone(machine.shared());
        let page = Arc::clone(machine.page);
        return Flow::Await(Pending {
            dst: op.a,
            fut: Box::pin(call_module(shared, page, callee, args)),
        });
    }

    let value = call_module_sync(machine, &prepared, callee, args);
    machine.set(op.a, value);
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
        if closure.code.may_suspend {
            let fut: BoxFuture<'static, _> = Box::pin(fn_value_call(closure, &mut args));
            return Flow::Await(Pending { dst: op.a, fut });
        }

        let value = fn_value_call_sync(closure, &mut args);
        machine.set(op.a, value);
        return Flow::Next;
    }

    // SAFETY: the type checker admits only a closure value here.
    let closure = unsafe { machine.take(op.b).materialize::<FnValue>() };
    if closure.code.may_suspend {
        let fut: BoxFuture<'static, _> =
            Box::pin(async move { fn_value_call(&closure, &mut args).await });
        return Flow::Await(Pending { dst: op.a, fut });
    }

    let value = fn_value_call_sync(&closure, &mut args);
    machine.set(op.a, value);
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
    machine.set(op.a, Value::handle(handle));
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
        Arc::clone(machine.page),
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
        page: Arc::clone(machine.page),
        code: Arc::clone(code),
        captures: captured.into(),
    };
    machine.set(op.a, Value::closure(closure));
    Flow::Next
}
