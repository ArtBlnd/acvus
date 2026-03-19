//! Option builtins (unwrap, unwrap_or).
//!
//! Both are polymorphic — they operate on `Option<T>` variants at the Value level.

use acvus_mir::builtins::BuiltinId;

use super::handler::BuiltinExecute;
use super::types::{FromTyped, IntoTyped};
use crate::value::{LazyValue, TypedValue, Value};

// ── Implementations ────────────────────────────────────────────────

fn unwrap_impl(v: Value) -> Value {
    let interner =
        crate::interner_ctx::get_interner().expect("unwrap: requires interner context");
    let some_tag = interner.intern("Some");
    let none_tag = interner.intern("None");
    match v {
        Value::Lazy(LazyValue::Variant {
            tag,
            payload: Some(inner),
        }) if tag == some_tag => *inner,
        Value::Lazy(LazyValue::Variant { tag, .. }) if tag == none_tag => {
            panic!("unwrap: called on None")
        }
        _ => panic!("unwrap: expected Option variant, got {v:?}"),
    }
}

fn unwrap_or_impl(v: Value, default: Value) -> Value {
    let interner =
        crate::interner_ctx::get_interner().expect("unwrap_or: requires interner context");
    let some_tag = interner.intern("Some");
    let none_tag = interner.intern("None");
    match v {
        Value::Lazy(LazyValue::Variant {
            tag,
            payload: Some(inner),
        }) if tag == some_tag => *inner,
        Value::Lazy(LazyValue::Variant { tag, .. }) if tag == none_tag => default,
        _ => panic!("unwrap_or: expected Option variant, got {v:?}"),
    }
}

// ── Registration ───────────────────────────────────────────────────

pub fn entries() -> Vec<(BuiltinId, BuiltinExecute)> {
    vec![
        (BuiltinId::Unwrap, Box::new(|args: Vec<TypedValue>| {
            let mut it = args.into_iter();
            let v = Value::from_typed(it.next().expect("missing arg 0"))?;
            Ok(unwrap_impl(v).into_typed())
        })),
        (BuiltinId::UnwrapOr, Box::new(|args: Vec<TypedValue>| {
            let mut it = args.into_iter();
            let v = Value::from_typed(it.next().expect("missing arg 0"))?;
            let default = Value::from_typed(it.next().expect("missing arg 1"))?;
            Ok(unwrap_or_impl(v, default).into_typed())
        })),
    ]
}
