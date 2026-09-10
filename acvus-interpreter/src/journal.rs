//! Context - runtime context storage for the interpreter.
//!
//! `Context` is a single snapshot of context state. Read/write via `&self`
//! (interior mutability via RwLock). Projection-aware: `set_field` writes to
//! a nested path directly, enabling precise diff computation without
//! object-level dirty tracking.
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
        self.data.read().unwrap().get(key).cloned()
    }

    fn get_field(&self, key: &str, path: &[&str]) -> Option<Value> {
        let data = self.data.read().unwrap();
        let root = data.get(key)?;
        navigate_field(&self.interner, root, path).cloned()
    }

    fn set(&self, key: &str, value: Value) {
        self.writes.lock().unwrap().push(ContextWrite::Set {
            key: key.to_string(),
            value: value.clone(),
        });
        self.data.write().unwrap().insert(key.to_string(), value);
    }

    fn set_field(&self, key: &str, path: &[&str], value: Value) {
        if path.is_empty() {
            self.set(key, value);
            return;
        }
        self.writes.lock().unwrap().push(ContextWrite::FieldPatch {
            key: key.to_string(),
            path: path.iter().map(|s| s.to_string()).collect(),
            value: value.clone(),
        });
        let mut data = self.data.write().unwrap();
        let root = data.get(key).cloned().unwrap_or(Value::Unit);
        let updated = deep_set_field(&self.interner, root, path, value);
        data.insert(key.to_string(), updated);
    }

    fn fork(&self) -> Self {
        Self {
            data: RwLock::new(self.data.read().unwrap().clone()),
            writes: Mutex::new(Vec::new()),
            interner: self.interner.clone(),
        }
    }

    fn take_writes(&self) -> Vec<ContextWrite> {
        std::mem::take(&mut *self.writes.lock().unwrap())
    }
}

// -- Helpers ---------------------------------------------------------

/// Navigate into a nested field of a Value.
fn navigate_field<'a>(interner: &Interner, root: &'a Value, path: &[&str]) -> Option<&'a Value> {
    if path.is_empty() {
        return Some(root);
    }
    match root {
        Value::Object(arc_map) => {
            let field_key = interner.intern(path[0]);
            let child = arc_map.get(&field_key)?;
            navigate_field(interner, child, &path[1..])
        }
        _ => None,
    }
}

/// Deep-set a nested field on a Value. Clones the object at each level.
/// path must be non-empty.
fn deep_set_field(interner: &Interner, root: Value, path: &[&str], value: Value) -> Value {
    debug_assert!(!path.is_empty());

    if path.len() == 1 {
        if let Value::Object(ref arc_map) = root {
            let mut map = (**arc_map).clone();
            let field_key = interner.intern(path[0]);
            map.insert(field_key, value);
            return Value::object(map);
        }
        return value;
    }

    if let Value::Object(ref arc_map) = root {
        let mut map = (**arc_map).clone();
        let field_key = interner.intern(path[0]);
        let child = map.get(&field_key).cloned().unwrap_or(Value::Unit);
        let updated_child = deep_set_field(interner, child, &path[1..], value);
        map.insert(field_key, updated_child);
        return Value::object(map);
    }

    value
}

// -- Tests -----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::Interner;
    use rustc_hash::FxHashMap;

    fn make_ctx(pairs: &[(&str, Value)]) -> InMemoryContext {
        let i = Interner::new();
        let data: HashMap<String, Value> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        InMemoryContext::new(data, i)
    }

    // -- get / set ----------------------------------------------

    #[test]
    fn get_returns_stored_value() {
        let ctx = make_ctx(&[("x", Value::int(42))]);
        assert_eq!(ctx.get("x"), Some(Value::int(42)));
    }

    #[test]
    fn get_missing_returns_none() {
        let ctx = make_ctx(&[]);
        assert_eq!(ctx.get("x"), None);
    }

    #[test]
    fn set_overwrites_value() {
        let ctx = make_ctx(&[("x", Value::int(1))]);
        ctx.set("x", Value::int(2));
        assert_eq!(ctx.get("x"), Some(Value::int(2)));
    }

    #[test]
    fn set_records_write() {
        let ctx = make_ctx(&[]);
        ctx.set("x", Value::int(42));
        let writes = ctx.take_writes();
        assert_eq!(writes.len(), 1);
        assert!(matches!(&writes[0], ContextWrite::Set { key, .. } if key == "x"));
    }

    // -- get_field / set_field ----------------------------------

    #[test]
    fn get_field_navigates_object() {
        let i = Interner::new();
        let obj = Value::object(FxHashMap::from_iter([
            (i.intern("name"), Value::string("alice")),
            (i.intern("age"), Value::int(30)),
        ]));
        let ctx = InMemoryContext::new(HashMap::from([("user".to_string(), obj)]), i);
        assert_eq!(
            ctx.get_field("user", &["name"]),
            Some(Value::string("alice"))
        );
        assert_eq!(ctx.get_field("user", &["age"]), Some(Value::int(30)));
    }

    #[test]
    fn get_field_nested() {
        let i = Interner::new();
        let inner = Value::object(FxHashMap::from_iter([(
            i.intern("city"),
            Value::string("seoul"),
        )]));
        let outer = Value::object(FxHashMap::from_iter([(i.intern("address"), inner)]));
        let ctx = InMemoryContext::new(HashMap::from([("user".to_string(), outer)]), i);
        assert_eq!(
            ctx.get_field("user", &["address", "city"]),
            Some(Value::string("seoul"))
        );
    }

    #[test]
    fn get_field_missing_returns_none() {
        let ctx = make_ctx(&[("x", Value::int(42))]);
        assert_eq!(ctx.get_field("x", &["name"]), None);
    }

    #[test]
    fn set_field_updates_nested() {
        let i = Interner::new();
        let obj = Value::object(FxHashMap::from_iter([
            (i.intern("name"), Value::string("alice")),
            (i.intern("age"), Value::int(30)),
        ]));
        let ctx = InMemoryContext::new(HashMap::from([("user".to_string(), obj)]), i);
        ctx.set_field("user", &["name"], Value::string("bob"));
        assert_eq!(ctx.get_field("user", &["name"]), Some(Value::string("bob")));
        // age unchanged
        assert_eq!(ctx.get_field("user", &["age"]), Some(Value::int(30)));
    }

    #[test]
    fn set_field_records_field_patch() {
        let i = Interner::new();
        let obj = Value::object(FxHashMap::from_iter([(
            i.intern("name"),
            Value::string("alice"),
        )]));
        let ctx = InMemoryContext::new(HashMap::from([("user".to_string(), obj)]), i);
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
        let inner = Value::object(FxHashMap::from_iter([(
            i.intern("city"),
            Value::string("seoul"),
        )]));
        let outer = Value::object(FxHashMap::from_iter([(i.intern("address"), inner)]));
        let ctx = InMemoryContext::new(HashMap::from([("user".to_string(), outer)]), i);
        ctx.set_field("user", &["address", "city"], Value::string("busan"));
        assert_eq!(
            ctx.get_field("user", &["address", "city"]),
            Some(Value::string("busan"))
        );
    }

    #[test]
    fn set_field_empty_path_is_set() {
        let ctx = make_ctx(&[("x", Value::int(1))]);
        ctx.set_field("x", &[], Value::int(2));
        assert_eq!(ctx.get("x"), Some(Value::int(2)));
        let writes = ctx.take_writes();
        // Empty path -> ContextWrite::Set, not FieldPatch.
        assert!(matches!(&writes[0], ContextWrite::Set { .. }));
    }

    // -- fork ---------------------------------------------------

    #[test]
    fn fork_creates_independent_copy() {
        let ctx = make_ctx(&[("x", Value::int(1))]);
        let forked = ctx.fork();
        // Forked sees same value.
        assert_eq!(forked.get("x"), Some(Value::int(1)));
        // Mutate forked - original unchanged.
        forked.set("x", Value::int(2));
        assert_eq!(forked.get("x"), Some(Value::int(2)));
        assert_eq!(ctx.get("x"), Some(Value::int(1)));
    }

    #[test]
    fn fork_has_empty_writes() {
        let ctx = make_ctx(&[("x", Value::int(1))]);
        ctx.set("x", Value::int(2)); // parent has a write
        let forked = ctx.fork();
        let writes = forked.take_writes();
        assert!(writes.is_empty(), "forked context should have empty writes");
    }

    // -- concurrent read/write ----------------------------------

    #[test]
    fn concurrent_read_write() {
        use std::thread;
        let ctx = make_ctx(&[("counter", Value::int(0))]);
        let ctx_ref = &ctx;

        thread::scope(|s| {
            // Writer thread
            s.spawn(|| {
                for i in 1..=100 {
                    ctx_ref.set("counter", Value::int(i));
                }
            });
            // Reader thread
            s.spawn(|| {
                for _ in 0..100 {
                    let _ = ctx_ref.get("counter");
                }
            });
        });

        // Final value should be 100.
        assert_eq!(ctx.get("counter"), Some(Value::int(100)));
    }
}
