use std::ops::Deref;
use std::sync::Arc;

/// Immutable, shared wrapper. Indicates the value is finalized and read-only.
///
/// Built via `Freeze::new(value)`. Cheaply cloneable (Arc internally).
/// Deref to &T for transparent read access.
#[derive(Debug)]
pub struct Freeze<T>(Arc<T>);

impl<T> Freeze<T> {
    pub fn new(value: T) -> Self {
        Freeze(Arc::new(value))
    }
}

impl<T> Deref for Freeze<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> Clone for Freeze<T> {
    fn clone(&self) -> Self {
        Freeze(Arc::clone(&self.0))
    }
}

impl<T: Default> Default for Freeze<T> {
    fn default() -> Self {
        Freeze::new(T::default())
    }
}
