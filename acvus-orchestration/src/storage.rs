use std::collections::HashMap;
use std::sync::Arc;

use acvus_interpreter::Value;

/// Storage backend trait for passing data between orchestration nodes.
///
/// Stores `Value` wrapped in `Arc` for cheap cloning. `get_mut` uses
/// `Arc::make_mut` for copy-on-write — safe as long as all coroutines
/// that may hold clones have been dropped before mutation.
pub trait Storage {
    fn get(&self, key: &str) -> Option<Arc<Value>>;
    fn get_mut(&mut self, key: &str) -> Option<&mut Value>;
    fn set(&mut self, key: String, value: Value);
    fn remove(&mut self, key: &str);
}

/// Simple in-memory storage backed by a `HashMap`.
#[derive(Debug)]
pub struct HashMapStorage {
    entries: HashMap<String, Arc<Value>>,
}

impl Default for HashMapStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl HashMapStorage {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}

impl Storage for HashMapStorage {
    fn get(&self, key: &str) -> Option<Arc<Value>> {
        self.entries.get(key).cloned()
    }

    fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.entries.get_mut(key).map(Arc::make_mut)
    }

    fn set(&mut self, key: String, value: Value) {
        self.entries.insert(key, Arc::new(value));
    }

    fn remove(&mut self, key: &str) {
        self.entries.remove(key);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn set_get() {
        let mut s = HashMapStorage::new();
        s.set("x".into(), Value::String("hello".into()));
        assert!(matches!(&*s.get("x").unwrap(), Value::String(s) if s == "hello"));
        assert!(s.get("y").is_none());
    }

    #[test]
    fn overwrite() {
        let mut s = HashMapStorage::new();
        s.set("x".into(), Value::String("first".into()));
        s.set(
            "x".into(),
            Value::Object(BTreeMap::from([("v".into(), Value::Int(2))])),
        );
        assert!(matches!(&*s.get("x").unwrap(), Value::Object(_)));
    }
}
