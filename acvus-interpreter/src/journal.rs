//! Context - runtime context storage for the interpreter.
//!
//! `Context` is a single snapshot of context state. Read/write via `&self`
//! (interior mutability via RwLock). Projection-aware: `set_field` writes to
//! a nested path directly, enabling precise diff computation without
//! object-level dirty tracking.
//!
//! Values are copied in and out through the vtable table's explicit
//! `clone`: the bridge until context dump/restore replaces sharing.
//!
//! `ContextWrite` describes a single context mutation (diff output).

use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

use acvus_utils::Interner;

use crate::value::Value;

// -- ContextWrite -----------------------------------------------------

/// A single context mutation recorded during execution.
#[derive(Debug)]
pub enum ContextWrite {
    /// Whole-value replacement (scalar, list, etc.)
    Set { key: String, value: Value },
    /// Nested object field patch.
    FieldPatch {
        key: String,
        path: Vec<String>,
        value: Value,
    },
}

// -- Context trait ---------------------------------------------------

/// Single snapshot of context state. Read/write via `&self`.
///
/// - `get` / `get_field`: read whole value or projected field.
/// - `set` / `set_field`: write whole value or projected field.
/// - `fork`: create an independent copy (for Spawn).
/// - `take_writes`: drain the accumulated context mutations.
pub trait RuntimeContext: Send + Sync + Sized {
    fn get(&self, key: &str) -> Option<Value>;
    fn get_field(&self, key: &str, path: &[&str]) -> Option<Value>;
    fn set(&self, key: &str, value: Value);
    fn set_field(&self, key: &str, path: &[&str], value: Value);
    fn fork(&self) -> Self;
    fn take_writes(&self) -> Vec<ContextWrite>;
}

// -- InMemoryContext -------------------------------------------------

/// In-memory Context backed by RwLock<HashMap>. No persistence.
/// Suitable for tests and the sequential executor.
pub struct InMemoryContext {
    data: RwLock<HashMap<String, Value>>,
    writes: Mutex<Vec<ContextWrite>>,
    interner: Interner,
}

impl InMemoryContext {
    pub fn new(initial: HashMap<String, Value>, interner: Interner) -> Self {
        Self {
            data: RwLock::new(initial),
            writes: Mutex::new(Vec::new()),
            interner,
        }
    }

    pub fn empty(interner: Interner) -> Self {
        Self::new(HashMap::new(), interner)
    }
}

impl RuntimeContext for InMemoryContext {
    fn get(&self, key: &str) -> Option<Value> {
        self.data
            .read()
            .unwrap()
            .get(key)
            .map(|v| v.deep_clone())
    }

    fn get_field(&self, key: &str, path: &[&str]) -> Option<Value> {
        let data = self.data.read().unwrap();
        let root = data.get(key)?;
        navigate_field(&self.interner, root, path).map(|v| v.deep_clone())
    }

    fn set(&self, key: &str, value: Value) {
        self.writes.lock().unwrap().push(ContextWrite::Set {
            key: key.to_string(),
            value: value.deep_clone(),
        });
        if let Some(old) = self.data.write().unwrap().insert(key.to_string(), value) {
            drop(old);
        }
    }

    fn set_field(&self, key: &str, path: &[&str], value: Value) {
        if path.is_empty() {
            self.set(key, value);
            return;
        }
        self.writes.lock().unwrap().push(ContextWrite::FieldPatch {
            key: key.to_string(),
            path: path.iter().map(|s| s.to_string()).collect(),
            value: value.deep_clone(),
        });
        let mut data = self.data.write().unwrap();
        let root = data.entry(key.to_string()).or_insert_with(Value::unit);
        deep_set_field(&self.interner, root, path, value);
    }

    fn fork(&self) -> Self {
        let data = self
            .data
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.deep_clone()))
            .collect();
        Self {
            data: RwLock::new(data),
            writes: Mutex::new(Vec::new()),
            interner: self.interner.clone(),
        }
    }

    fn take_writes(&self) -> Vec<ContextWrite> {
        std::mem::take(&mut *self.writes.lock().unwrap())
    }
}

// -- Helpers ---------------------------------------------------------

fn navigate_field<'a>(interner: &Interner, root: &'a Value, path: &[&str]) -> Option<&'a Value> {
    if path.is_empty() {
        return Some(root);
    }
    if !root.is_object() {
        return None;
    }
    // SAFETY: is_object checked the vtable.
    let map = unsafe { root.as_object() };
    let child = map.get(&interner.intern(path[0]))?;
    navigate_field(interner, child, &path[1..])
}

/// Deep-set a nested field in place; a non-object on the path is replaced
/// by `value`.
fn deep_set_field(interner: &Interner, root: &mut Value, path: &[&str], value: Value) {
    debug_assert!(!path.is_empty());
    if !root.is_object() {
        *root = value;
        return;
    }
    // SAFETY: is_object checked the vtable.
    let map = unsafe { root.as_object_mut() };
    let field_key = interner.intern(path[0]);
    if path.len() == 1 {
        map.insert(field_key, value);
        return;
    }
    let child = map.entry(field_key).or_insert_with(Value::unit);
    deep_set_field(interner, child, &path[1..], value);
}

// -- Tests -----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::Interner;
    use rustc_hash::FxHashMap;

    fn make_ctx(pairs: Vec<(&str, Value)>) -> InMemoryContext {
        let i = Interner::new();
        let data: HashMap<String, Value> =
            pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        InMemoryContext::new(data, i)
    }

    /// Same bits for a `Small`, same text for a string: what the tests store.
    fn eq(_: &InMemoryContext, a: Option<Value>, b: Value) -> bool {
        let Some(a) = a else {
            return false;
        };
        match (&a, &b) {
            (Value::Small(x), Value::Small(y)) => x == y,
            // SAFETY: both are strings by the vtable check.
            (Value::Large(..), Value::Large(..)) if a.is_string() && b.is_string() => unsafe {
                a.as_str() == b.as_str()
            },
            _ => false,
        }
    }

    // -- get / set ----------------------------------------------

    #[test]
    fn get_returns_stored_value() {
        let ctx = make_ctx(vec![("x", Value::int(42))]);
        assert!(eq(&ctx, ctx.get("x"), Value::int(42)));
    }

    #[test]
    fn get_missing_returns_none() {
        let ctx = make_ctx(vec![]);
        assert!(ctx.get("x").is_none());
    }

    #[test]
    fn set_overwrites_value() {
        let ctx = make_ctx(vec![("x", Value::int(1))]);
        ctx.set("x", Value::int(2));
        assert!(eq(&ctx, ctx.get("x"), Value::int(2)));
    }

    #[test]
    fn set_records_write() {
        let ctx = make_ctx(vec![]);
        ctx.set("x", Value::int(42));
        let writes = ctx.take_writes();
        assert_eq!(writes.len(), 1);
        assert!(matches!(&writes[0], ContextWrite::Set { key, .. } if key == "x"));
    }

    // -- get_field / set_field ----------------------------------

    fn user(i: &Interner) -> Value {
        Value::object(FxHashMap::from_iter([
            (i.intern("name"), Value::string("alice")),
            (i.intern("age"), Value::int(30)),
        ]))
    }

    fn nested_user(i: &Interner) -> Value {
        let inner = Value::object(FxHashMap::from_iter([(
            i.intern("city"),
            Value::string("seoul"),
        )]));
        Value::object(FxHashMap::from_iter([(i.intern("address"), inner)]))
    }

    fn ctx_with(i: Interner, key: &str, value: Value) -> InMemoryContext {
        InMemoryContext::new(HashMap::from([(key.to_string(), value)]), i)
    }

    #[test]
    fn get_field_navigates_object() {
        let i = Interner::new();
        let ctx = ctx_with(i.clone(), "user", user(&i));
        assert!(eq(&ctx, ctx.get_field("user", &["name"]), Value::string("alice")));
        assert!(eq(&ctx, ctx.get_field("user", &["age"]), Value::int(30)));
    }

    #[test]
    fn get_field_nested() {
        let i = Interner::new();
        let ctx = ctx_with(i.clone(), "user", nested_user(&i));
        assert!(eq(
            &ctx,
            ctx.get_field("user", &["address", "city"]),
            Value::string("seoul")
        ));
    }

    #[test]
    fn get_field_missing_returns_none() {
        let ctx = make_ctx(vec![("x", Value::int(42))]);
        assert!(ctx.get_field("x", &["name"]).is_none());
    }

    #[test]
    fn set_field_updates_nested() {
        let i = Interner::new();
        let ctx = ctx_with(i.clone(), "user", user(&i));
        ctx.set_field("user", &["name"], Value::string("bob"));
        assert!(eq(&ctx, ctx.get_field("user", &["name"]), Value::string("bob")));
        assert!(eq(&ctx, ctx.get_field("user", &["age"]), Value::int(30)));
    }

    #[test]
    fn set_field_records_field_patch() {
        let i = Interner::new();
        let ctx = ctx_with(i.clone(), "user", user(&i));
        ctx.set_field("user", &["name"], Value::string("bob"));
        let writes = ctx.take_writes();
        assert_eq!(writes.len(), 1);
        assert!(matches!(
            &writes[0],
            ContextWrite::FieldPatch { key, path, .. }
            if key == "user" && path == &["name"]
        ));
    }

    #[test]
    fn set_field_deep_nested() {
        let i = Interner::new();
        let ctx = ctx_with(i.clone(), "user", nested_user(&i));
        ctx.set_field("user", &["address", "city"], Value::string("busan"));
        assert!(eq(
            &ctx,
            ctx.get_field("user", &["address", "city"]),
            Value::string("busan")
        ));
    }

    #[test]
    fn set_field_empty_path_is_set() {
        let ctx = make_ctx(vec![("x", Value::int(1))]);
        ctx.set_field("x", &[], Value::int(2));
        assert!(eq(&ctx, ctx.get("x"), Value::int(2)));
        let writes = ctx.take_writes();
        assert!(matches!(&writes[0], ContextWrite::Set { .. }));
    }

    // -- fork ---------------------------------------------------

    #[test]
    fn fork_creates_independent_copy() {
        let ctx = make_ctx(vec![("x", Value::int(1))]);
        let forked = ctx.fork();
        assert!(eq(&forked, forked.get("x"), Value::int(1)));
        forked.set("x", Value::int(2));
        assert!(eq(&forked, forked.get("x"), Value::int(2)));
        assert!(eq(&ctx, ctx.get("x"), Value::int(1)));
    }

    #[test]
    fn fork_has_empty_writes() {
        let ctx = make_ctx(vec![("x", Value::int(1))]);
        ctx.set("x", Value::int(2));
        let forked = ctx.fork();
        assert!(forked.take_writes().is_empty());
    }

    // -- concurrent read/write ----------------------------------

    #[test]
    fn concurrent_read_write() {
        use std::thread;
        let ctx = make_ctx(vec![("counter", Value::int(0))]);
        let ctx_ref = &ctx;

        thread::scope(|s| {
            s.spawn(|| {
                for i in 1..=100 {
                    ctx_ref.set("counter", Value::int(i));
                }
            });
            s.spawn(|| {
                for _ in 0..100 {
                    let _ = ctx_ref.get("counter");
                }
            });
        });

        assert!(eq(&ctx, ctx.get("counter"), Value::int(100)));
    }
}
