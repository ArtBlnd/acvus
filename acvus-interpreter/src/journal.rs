//! Context - runtime context storage for the interpreter.
//!
//! `Context` is a single snapshot of context state. Read/write via `&self`
//! (interior mutability via RwLock). A context place is a storage like a
//! local (RFC-0018): `take` moves its value out and `set` moves one in;
//! nothing is copied. The page remembers which keys a run assigned, and
//! `take_writes` hands their final values out.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Mutex, RwLock};

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
pub trait RuntimeContext: Send + Sync {
    /// Move the whole value out; the key is unset until `set`.
    fn take(&self, key: &str) -> Option<Value>;
    fn set(&self, key: &str, value: Value);
    /// The final value of every key `set` since the last drain, moved out.
    fn take_writes(&self) -> Vec<ContextWrite>;
}

// -- InMemoryContext -------------------------------------------------

/// In-memory Context backed by RwLock<HashMap>. No persistence.
/// Suitable for tests and the sequential executor.
pub struct InMemoryContext {
    data: RwLock<HashMap<String, Value>>,
    assigned: Mutex<BTreeSet<String>>,
}

impl InMemoryContext {
    pub fn new(initial: HashMap<String, Value>) -> Self {
        Self {
            data: RwLock::new(initial),
            assigned: Mutex::new(BTreeSet::new()),
        }
    }

    pub fn empty() -> Self {
        Self::new(HashMap::new())
    }
}

impl RuntimeContext for InMemoryContext {
    fn take(&self, key: &str) -> Option<Value> {
        self.data.write().unwrap().remove(key)
    }

    fn set(&self, key: &str, value: Value) {
        self.assigned.lock().unwrap().insert(key.to_string());
        if let Some(old) = self.data.write().unwrap().insert(key.to_string(), value) {
            drop(old);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx(pairs: Vec<(&str, Value)>) -> InMemoryContext {
        let data: HashMap<String, Value> =
            pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        InMemoryContext::new(data)
    }

    fn is_int(v: Option<Value>, n: i64) -> bool {
        matches!(v, Some(Value::Small(_, bits)) if bits == n as u64)
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
        assert!(is_int(
            Some(Value::use_from(&mut {
                writes.into_iter().next().unwrap().value
            })),
            3
        ));
        assert!(ctx.take("x").is_none(), "the drained value left the page");
        assert!(is_int(ctx.take("y"), 9), "an unassigned key stays");
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
