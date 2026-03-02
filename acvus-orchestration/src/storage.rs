use std::collections::HashMap;

use crate::message::Output;

/// Storage backend trait for passing data between orchestration nodes.
///
/// Stores `Output` values (Text, Json, Image) keyed by node name.
/// Implement this trait to plug in your own backend (in-memory, Redis, DB, etc.).
pub trait Storage {
    fn get(&self, key: &str) -> Option<Output>;
    fn set(&mut self, key: String, value: Output);
}

/// Simple in-memory storage backed by a `HashMap`.
pub struct HashMapStorage {
    entries: HashMap<String, Output>,
}

impl HashMapStorage {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }
}

impl Storage for HashMapStorage {
    fn get(&self, key: &str) -> Option<Output> {
        self.entries.get(key).cloned()
    }

    fn set(&mut self, key: String, value: Output) {
        self.entries.insert(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get() {
        let mut s = HashMapStorage::new();
        s.set("x".into(), Output::Text("hello".into()));
        assert!(matches!(s.get("x"), Some(Output::Text(ref s)) if s == "hello"));
        assert!(s.get("y").is_none());
    }

    #[test]
    fn overwrite() {
        let mut s = HashMapStorage::new();
        s.set("x".into(), Output::Text("first".into()));
        s.set("x".into(), Output::Json(serde_json::json!({"v": 2})));
        assert!(matches!(s.get("x"), Some(Output::Json(_))));
    }
}
