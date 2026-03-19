//! List operation builtins (pure, synchronous).
//!
//! Only `len` and `reverse` remain as List-specific ops.
//! All other operations (`contains`, `first`, `last`, `join`, `flatten`)
//! are Iterator-only — List coerces to Iterator via Cast.

use acvus_mir::builtins::BuiltinId;

use super::handler::BuiltinExecute;
use super::types::{FromTyped, IntoTyped};
use crate::value::{TypedValue, Value};

// ── Implementations ────────────────────────────────────────────────

fn len_impl(items: Vec<Value>) -> i64 {
    items.len() as i64
}

fn reverse_impl(items: Vec<Value>) -> Vec<Value> {
    let mut v = items;
    v.reverse();
    v
}

// ── Manual BuiltinExecute helpers ──────────────────────────────────

fn extract_list(args: &mut std::vec::IntoIter<TypedValue>) -> Result<Vec<Value>, crate::error::RuntimeError> {
    Vec::<Value>::from_typed(args.next().expect("missing arg"))
}

// ── Registration ───────────────────────────────────────────────────

pub fn entries() -> Vec<(BuiltinId, BuiltinExecute)> {
    vec![
        (BuiltinId::Len, Box::new(|args: Vec<TypedValue>| {
            let mut it = args.into_iter();
            let items = extract_list(&mut it)?;
            Ok(len_impl(items).into_typed())
        })),
        (BuiltinId::Reverse, Box::new(|args: Vec<TypedValue>| {
            let mut it = args.into_iter();
            let items = extract_list(&mut it)?;
            Ok(reverse_impl(items).into_typed())
        })),
    ]
}
