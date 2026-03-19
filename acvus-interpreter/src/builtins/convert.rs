//! Type conversion builtins.
//!
//! `to_string` and `to_int` are polymorphic (accept `Value` directly),
//! so they use manual `BuiltinExecute` construction instead of `sync()`.

use acvus_mir::builtins::BuiltinId;

use super::handler::BuiltinExecute;
use super::types::{FromTyped, IntoTyped};
use crate::value::{PureValue, TypedValue, Value};

// ── Implementations ────────────────────────────────────────────────

fn to_string_impl(v: Value) -> Value {
    Value::string(match v {
        Value::Pure(PureValue::Int(n)) => n.to_string(),
        Value::Pure(PureValue::Float(f)) => f.to_string(),
        Value::Pure(PureValue::String(s)) => s,
        Value::Pure(PureValue::Bool(b)) => b.to_string(),
        Value::Pure(PureValue::Byte(b)) => b.to_string(),
        Value::Pure(PureValue::Unit) => "()".to_string(),
        _ => unreachable!("to_string: expected scalar or Unit, got {v:?}"),
    })
}

fn to_int_impl(v: Value) -> Value {
    match v {
        Value::Pure(PureValue::Float(f)) => Value::int(f as i64),
        Value::Pure(PureValue::Byte(b)) => Value::int(b as i64),
        _ => unreachable!("to_int: expected Float or Byte, got {v:?}"),
    }
}

fn to_float(n: i64) -> f64 {
    n as f64
}

fn char_to_int(s: String) -> i64 {
    s.chars().next().expect("char_to_int: empty string") as i64
}

fn int_to_char(n: i64) -> String {
    char::from_u32(n as u32)
        .expect("int_to_char: invalid codepoint")
        .to_string()
}

// ── Manual BuiltinExecute for polymorphic functions ────────────────

fn poly1(f: fn(Value) -> Value) -> BuiltinExecute {
    Box::new(move |args: Vec<TypedValue>| {
        let mut it = args.into_iter();
        let v = Value::from_typed(it.next().expect("missing arg 0"))?;
        Ok(f(v).into_typed())
    })
}

// ── Registration ───────────────────────────────────────────────────

pub fn entries() -> Vec<(BuiltinId, BuiltinExecute)> {
    vec![
        (BuiltinId::ToString,  poly1(to_string_impl)),
        (BuiltinId::ToInt,     poly1(to_int_impl)),
        (BuiltinId::ToFloat,   super::handler::sync(to_float)),
        (BuiltinId::CharToInt, super::handler::sync(char_to_int)),
        (BuiltinId::IntToChar, super::handler::sync(int_to_char)),
    ]
}
