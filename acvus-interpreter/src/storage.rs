use std::collections::HashMap;
use std::fmt;

use crate::value::{PureValue, StorageKey};

#[trait_variant::make(Send)]
pub trait Storage {
    type Error;

    async fn get(&self, key: &StorageKey) -> Result<PureValue, Self::Error>;
    async fn set(&mut self, key: &StorageKey, value: PureValue) -> Result<(), Self::Error>;
    async fn flush(&mut self) -> Result<(), Self::Error>;
    async fn sync(&mut self) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub struct InMemoryStorage {
    data: HashMap<StorageKey, PureValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KeyNotFound(pub StorageKey);

impl fmt::Display for KeyNotFound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "key not found: {}", self.0)
    }
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl Storage for InMemoryStorage {
    type Error = KeyNotFound;

    async fn get(&self, key: &StorageKey) -> Result<PureValue, Self::Error> {
        self.data.get(key).cloned().ok_or_else(|| KeyNotFound(key.clone()))
    }

    async fn set(&mut self, key: &StorageKey, value: PureValue) -> Result<(), Self::Error> {
        self.data.insert(key.clone(), value);
        Ok(())
    }

    async fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn sync(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn set_and_get() {
        let mut storage = InMemoryStorage::new();
        let key = StorageKey::root("user");
        storage.set(&key, PureValue::Int(42)).await.unwrap();
        assert_eq!(storage.get(&key).await.unwrap(), PureValue::Int(42));
    }

    #[tokio::test]
    async fn get_missing_key() {
        let storage = InMemoryStorage::new();
        let key = StorageKey::root("missing");
        assert!(storage.get(&key).await.is_err());
    }

    #[tokio::test]
    async fn overwrite() {
        let mut storage = InMemoryStorage::new();
        let key = StorageKey::root("x");
        storage.set(&key, PureValue::Int(1)).await.unwrap();
        storage.set(&key, PureValue::Int(2)).await.unwrap();
        assert_eq!(storage.get(&key).await.unwrap(), PureValue::Int(2));
    }

    #[tokio::test]
    async fn hierarchical_keys_independent() {
        let mut storage = InMemoryStorage::new();
        let parent = StorageKey::root("user");
        let child = parent.field("name");
        storage.set(&parent, PureValue::Int(1)).await.unwrap();
        storage.set(&child, PureValue::String("alice".into())).await.unwrap();
        assert_eq!(storage.get(&parent).await.unwrap(), PureValue::Int(1));
        assert_eq!(
            storage.get(&child).await.unwrap(),
            PureValue::String("alice".into())
        );
    }
}
