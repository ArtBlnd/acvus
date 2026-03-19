//! Byte/UTF-8 conversion builtins.

use std::marker::PhantomData;

use acvus_mir::builtins::BuiltinId;

use super::handler::{sync, BuiltinExecute};
use super::types::{FromTyped, IntoTyped, Opt};
use crate::value::{TypedValue, Value};

// ── Implementations ────────────────────────────────────────────────

fn to_bytes_impl(s: String) -> Value {
    Value::list(s.into_bytes().into_iter().map(Value::byte).collect())
}

fn to_utf8(bytes: Vec<u8>) -> Opt<String> {
    Opt(
        String::from_utf8(bytes).ok().map(Value::string),
        PhantomData,
    )
}

fn to_utf8_lossy(bytes: Vec<u8>) -> String {
    String::from_utf8_lossy(&bytes).into_owned()
}

// ── Registration ───────────────────────────────────────────────────

pub fn entries() -> Vec<(BuiltinId, BuiltinExecute)> {
    vec![
        (BuiltinId::ToBytes, Box::new(|args: Vec<TypedValue>| {
            let mut it = args.into_iter();
            let s = String::from_typed(it.next().expect("missing arg 0"))?;
            Ok(to_bytes_impl(s).into_typed())
        })),
        (BuiltinId::ToUtf8,      sync(to_utf8)),
        (BuiltinId::ToUtf8Lossy, sync(to_utf8_lossy)),
    ]
}
