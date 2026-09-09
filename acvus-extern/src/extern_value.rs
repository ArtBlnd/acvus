//! A value of an extension type: a static name and an erased payload.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use acvus_utils::{Interner, QualifiedRef};

/// The static name of an extension type. Interned to a `QualifiedRef` when
/// the compiler needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExternTypeName {
    pub ns: Option<&'static str>,
    pub name: &'static str,
}

impl ExternTypeName {
    pub fn qref(self, interner: &Interner) -> QualifiedRef {
        match self.ns {
            Some(ns) => QualifiedRef::qualified(interner.intern(ns), interner.intern(self.name)),
            None => QualifiedRef::root(interner.intern(self.name)),
        }
    }
}

impl fmt::Display for ExternTypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ns {
            Some(ns) => write!(f, "{ns}::{}", self.name),
            None => write!(f, "{}", self.name),
        }
    }
}

pub struct ExternValue {
    pub type_name: ExternTypeName,
    inner: Arc<dyn Any + Send + Sync>,
}

/// Why a payload could not be taken out of an `ExternValue`.
pub enum PayloadMismatch {
    /// The payload is another Rust type.
    OtherType,
    /// The payload is this type but still shared; the `Arc` is returned.
    Shared(Arc<dyn Any + Send + Sync>),
}

impl Clone for ExternValue {
    fn clone(&self) -> Self {
        Self {
            type_name: self.type_name,
            inner: Arc::clone(&self.inner),
        }
    }
}

impl ExternValue {
    pub fn new<T: Any + Send + Sync>(type_name: ExternTypeName, value: T) -> Self {
        Self {
            type_name,
            inner: Arc::new(value),
        }
    }

    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.inner.downcast_ref()
    }

    pub fn into_owned<T: Any + Send + Sync>(self) -> Result<T, PayloadMismatch> {
        match self.inner.downcast::<T>() {
            Ok(arc) => Arc::try_unwrap(arc)
                .map_err(|arc| PayloadMismatch::Shared(arc as Arc<dyn Any + Send + Sync>)),
            Err(_) => Err(PayloadMismatch::OtherType),
        }
    }

    /// Take the payload out, cloning it when it is still shared.
    pub fn into_cloned<T: Any + Send + Sync + Clone>(self) -> Result<T, PayloadMismatch> {
        match self.inner.downcast::<T>() {
            Ok(arc) => Ok(Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone())),
            Err(_) => Err(PayloadMismatch::OtherType),
        }
    }
}

impl fmt::Debug for ExternValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.type_name)
    }
}

impl PartialEq for ExternValue {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}
