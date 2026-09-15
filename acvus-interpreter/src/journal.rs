//! Context - runtime context storage for the interpreter.
//!
//! `Context` is a single snapshot of context state. Read/write via `&self`
//! (interior mutability via RwLock). A context place is a storage like a
//! local (RFC-0018): `take` moves its value out and `set` moves one in;
//! nothing is copied. The page remembers which keys a run assigned, and
//! `take_writes` hands their final values out.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Mutex, RwLock};

use acvus_utils::Interner;

use crate::value::Value;

// -- ContextWrite -----------------------------------------------------

/// The final value of a context a run assigned.
#[derive(Debug)]
pub struct ContextWrite {
    pub key: String,
    pub value: Value,
}

// -- Context trait ---------------------------------------------------

/// Single snapshot of context state. Read/write via `&self`.
pub trait RuntimeContext: Send + Sync + Sized {
    /// Move the whole value out; the key is unset until `set`.
    fn take(&self, key: &str) -> Option<Value>;
    /// Move a field out of the stored object, leaving that field empty; a
    /// primitive field is copied.
    fn take_field(&self, key: &str, path: &[&str]) -> Option<Value>;
    fn set(&self, key: &str, value: Value);
    fn set_field(&self, key: &str, path: &[&str], value: Value);
    /// The final value of every key `set` since the last drain, moved out.
    fn take_writes(&self) -> Vec<ContextWrite>;
}

// -- InMemoryContext -------------------------------------------------

/// In-memory Context backed by RwLock<HashMap>. No persistence.
/// Suitable for tests and the sequential executor.
pub struct InMemoryContext {
    data: RwLock<HashMap<String, Value>>,
    assigned: Mutex<BTreeSet<String>>,
    interner: Interner,
}

impl InMemoryContext {
    pub fn new(initial: HashMap<String, Value>, interner: Interner) -> Self {
        Self {
            data: RwLock::new(initial),
            assigned: Mutex::new(BTreeSet::new()),
            interner,
        }
    }

    pub fn empty(interner: Interner) -> Self {
        Self::new(HashMap::new(), interner)
    }
}

impl RuntimeContext for InMemoryContext {
    fn take(&self, key: &str) -> Option<Value> {
        self.data.write().unwrap().remove(key)
    }

    fn take_field(&self, key: &str, path: &[&str]) -> Option<Value> {
        let mut data = self.data.write().unwrap();
        let root = data.get_mut(key)?;
        navigate_field_mut(&self.interner, root, path).map(Value::use_from)
    }

    fn set(&self, key: &str, value: Value) {
        self.assigned.lock().unwrap().insert(key.to_string());
        if let Some(old) = self.data.write().unwrap().insert(key.to_string(), value) {
            drop(old);
        }
    }

    fn set_field(&self, key: &str, path: &[&str], value: Value) {
        if path.is_empty() {
            self.set(key, value);
            return;
        }
        self.assigned.lock().unwrap().insert(key.to_string());
        let mut data = self.data.write().unwrap();
        let root = data.entry(key.to_string()).or_insert_with(Value::unit);
        deep_set_field(&self.interner, root, path, value);
    }

    fn take_writes(&self) -> Vec<ContextWrite> {
        let assigned = std::mem::take(&mut *self.assigned.lock().unwrap());
        let mut data = self.data.write().unwrap();
        assigned
            .into_iter()
            .map(|key| {
                let value = data.remove(&key).unwrap_or_else(|| {
                    panic!("context '{key}' was assigned but holds no value at the end of the run")
                });
                ContextWrite { key, value }
            })
            .collect()
    }
}

// -- Helpers ---------------------------------------------------------

fn navigate_field_mut<'a>(
    interner: &Interner,
    root: &'a mut Value,
    path: &[&str],
) -> Option<&'a mut Value> {
    if path.is_empty() {
        return Some(root);
    }
    if !root.is_object() {
        return None;
    }
    // SAFETY: is_object checked the vtable.
    let map = unsafe { root.as_object_mut() };
    let child = map.get_mut(&interner.intern(path[0]))?;
    navigate_field_mut(interner, child, &path[1..])
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

    fn is_int(v: Option<Value>, n: i64) -> bool {
        matches!(v, Some(Value::Small(bits)) if bits == n as u64)
    }

    fn is_str(v: Option<Value>, s: &str) -> bool {
        let Some(v) = v else {
            return false;
        };
        // SAFETY: is_string checked the vtable.
        v.is_string() && unsafe { v.as_str() } == s
    }

    #[test]
    fn take_moves_the_value_out() {
        let ctx = make_ctx(vec![("x", Value::string("hi"))]);
        assert!(is_str(ctx.take("x"), "hi"));
        assert!(ctx.take("x").is_none());
    }

    #[test]
    fn take_missing_returns_none() {
        let ctx = make_ctx(vec![]);
        assert!(ctx.take("x").is_none());
    }

    #[test]
    fn set_after_take_restores_the_key() {
        let ctx = make_ctx(vec![("x", Value::int(1))]);
        assert!(is_int(ctx.take("x"), 1));
        ctx.set("x", Value::int(2));
        assert!(is_int(ctx.take("x"), 2));
    }

    #[test]
    fn take_writes_hands_out_final_values() {
        let ctx = make_ctx(vec![("x", Value::int(1)), ("y", Value::int(9))]);
        ctx.set("x", Value::int(2));
        ctx.set("x", Value::int(3));
        let writes = ctx.take_writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].key, "x");
        assert!(is_int(Some(Value::use_from(&mut { writes.into_iter().next().unwrap().value })), 3));
        assert!(ctx.take("x").is_none(), "the drained value left the page");
        assert!(is_int(ctx.take("y"), 9), "an unassigned key stays");
    }

    fn user(i: &Interner) -> Value {
        Value::object(FxHashMap::from_iter([
            (i.intern("name"), Value::string("Alice")),
            (i.intern("age"), Value::int(30)),
        ]))
    }

    fn nested_user(i: &Interner) -> Value {
        Value::object(FxHashMap::from_iter([(
            i.intern("profile"),
            Value::object(FxHashMap::from_iter([(i.intern("city"), Value::string("Seoul"))])),
        )]))
    }

    fn ctx_with(i: Interner, key: &str, value: Value) -> InMemoryContext {
        let mut data = HashMap::new();
        data.insert(key.to_string(), value);
        InMemoryContext::new(data, i)
    }

    #[test]
    fn take_field_moves_a_field_out() {
        let i = Interner::new();
        let ctx = ctx_with(i.clone(), "user", user(&i));
        assert!(is_str(ctx.take_field("user", &["name"]), "Alice"));
        assert!(is_int(ctx.take_field("user", &["age"]), 30));
        assert!(is_int(ctx.take_field("user", &["age"]), 30), "a primitive field is copied");
    }

    #[test]
    fn take_field_nested() {
        let i = Interner::new();
        let ctx = ctx_with(i.clone(), "user", nested_user(&i));
        assert!(is_str(ctx.take_field("user", &["profile", "city"]), "Seoul"));
    }

    #[test]
    fn take_field_missing_returns_none() {
        let i = Interner::new();
        let ctx = ctx_with(i.clone(), "user", user(&i));
        assert!(ctx.take_field("user", &["missing"]).is_none());
    }

    #[test]
    fn set_field_updates_nested() {
        let i = Interner::new();
        let ctx = ctx_with(i.clone(), "user", user(&i));
        ctx.set_field("user", &["age"], Value::int(31));
        assert!(is_int(ctx.take_field("user", &["age"]), 31));
    }

    #[test]
    fn set_field_records_the_key() {
        let i = Interner::new();
        let ctx = ctx_with(i.clone(), "user", user(&i));
        ctx.set_field("user", &["age"], Value::int(31));
        let writes = ctx.take_writes();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].key, "user");
    }

    #[test]
    fn set_field_deep_nested() {
        let i = Interner::new();
        let ctx = ctx_with(i.clone(), "user", nested_user(&i));
        ctx.set_field("user", &["profile", "city"], Value::string("Busan"));
        assert!(is_str(ctx.take_field("user", &["profile", "city"]), "Busan"));
    }

    #[test]
    fn set_field_empty_path_is_set() {
        let ctx = make_ctx(vec![("x", Value::int(1))]);
        ctx.set_field("x", &[], Value::int(2));
        assert!(is_int(ctx.take("x"), 2));
    }

    #[test]
    fn concurrent_set_and_take() {
        use std::sync::Arc;
        let ctx = Arc::new(make_ctx(vec![("counter", Value::int(0))]));
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let ctx_ref = Arc::clone(&ctx);
                std::thread::spawn(move || {
                    ctx_ref.set("counter", Value::int(i));
                    let _ = ctx_ref.take("counter");
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
