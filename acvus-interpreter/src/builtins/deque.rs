//! Deque operation builtins (sync only).
//!
//! `extend` is async (needs iterator collection) and is handled in the interpreter.

use acvus_mir::builtins::BuiltinId;

use super::handler::BuiltinExecute;
use super::types::{FromTyped, IntoTyped};
use crate::value::{TypedValue, Value};
use acvus_utils::TrackedDeque;

// ── Implementations ────────────────────────────────────────────────

fn append_impl(mut deque: TrackedDeque<Value>, item: Value) -> Value {
    deque.push(item);
    Value::deque(deque)
}

fn consume_impl(mut deque: TrackedDeque<Value>, n: i64) -> Value {
    deque.consume(n as usize);
    Value::deque(deque)
}

// ── Registration ───────────────────────────────────────────────────

pub fn entries() -> Vec<(BuiltinId, BuiltinExecute)> {
    vec![
        (BuiltinId::Append, Box::new(|args: Vec<TypedValue>| {
            let mut it = args.into_iter();
            let deque = TrackedDeque::from_typed(it.next().expect("missing arg 0"))?;
            let item = Value::from_typed(it.next().expect("missing arg 1"))?;
            Ok(append_impl(deque, item).into_typed())
        })),
        (BuiltinId::Consume, Box::new(|args: Vec<TypedValue>| {
            let mut it = args.into_iter();
            let deque = TrackedDeque::from_typed(it.next().expect("missing arg 0"))?;
            let n = i64::from_typed(it.next().expect("missing arg 1"))?;
            Ok(consume_impl(deque, n).into_typed())
        })),
    ]
}
