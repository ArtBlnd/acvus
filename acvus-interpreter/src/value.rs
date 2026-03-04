use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use acvus_mir::ir::Label;
use serde::{Deserialize, Serialize};

/// Data-only value — no functions, no closures.
/// Serializable, cloneable, used at context boundaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PureValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Unit,
    Range {
        start: i64,
        end: i64,
        inclusive: bool,
    },
    List(Vec<PureValue>),
    Object(BTreeMap<String, PureValue>),
    Tuple(Vec<PureValue>),
    Byte(u8),
}

/// Runtime value — flat enum for fast dispatch.
/// Includes everything PureValue has, plus Fn for closures.
///
/// `PartialEq` compares structural equality for data variants.
/// `Fn` and `Opaque` are never equal (not comparable).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Unit,
    Range {
        start: i64,
        end: i64,
        inclusive: bool,
    },
    List(Vec<Value>),
    Object(BTreeMap<String, Value>),
    Tuple(Vec<Value>),
    Byte(u8),
    Fn(FnValue),
    Opaque(OpaqueValue),
}

/// An opaque value: carries a type name and an arbitrary payload.
/// Templates cannot inspect or destructure this — only pass it between extern functions.
#[derive(Clone)]
pub struct OpaqueValue {
    pub type_name: String,
    inner: Arc<dyn Any + Send + Sync>,
}

impl OpaqueValue {
    pub fn new<T>(type_name: impl Into<String>, value: T) -> Self
    where
        T: Any + Send + Sync,
    {
        Self {
            type_name: type_name.into(),
            inner: Arc::new(value),
        }
    }

    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.inner.downcast_ref()
    }
}

impl PartialEq for OpaqueValue {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl fmt::Debug for OpaqueValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Opaque<{}>", self.type_name)
    }
}

/// A closure value: label pointing to its body + captured values.
#[derive(Debug, Clone)]
pub struct FnValue {
    pub body: Label,
    pub captures: Vec<Arc<Value>>,
}

impl PartialEq for FnValue {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl Value {
    /// Convert a PureValue into a Value. Infallible.
    pub fn from_pure(pure: PureValue) -> Self {
        match pure {
            PureValue::Int(v) => Value::Int(v),
            PureValue::Float(v) => Value::Float(v),
            PureValue::String(v) => Value::String(v),
            PureValue::Bool(v) => Value::Bool(v),
            PureValue::Unit => Value::Unit,
            PureValue::Range {
                start,
                end,
                inclusive,
            } => Value::Range {
                start,
                end,
                inclusive,
            },
            PureValue::List(items) => {
                Value::List(items.into_iter().map(Value::from_pure).collect())
            }
            PureValue::Object(fields) => Value::Object(
                fields
                    .into_iter()
                    .map(|(k, v)| (k, Value::from_pure(v)))
                    .collect(),
            ),
            PureValue::Tuple(elems) => {
                Value::Tuple(elems.into_iter().map(Value::from_pure).collect())
            }
            PureValue::Byte(b) => Value::Byte(b),
        }
    }

    /// Convert a Value into a PureValue.
    /// Panics if the value contains Fn — the type checker guarantees this won't happen
    /// at context boundaries.
    pub fn into_pure(self) -> PureValue {
        match self {
            Value::Int(v) => PureValue::Int(v),
            Value::Float(v) => PureValue::Float(v),
            Value::String(v) => PureValue::String(v),
            Value::Bool(v) => PureValue::Bool(v),
            Value::Unit => PureValue::Unit,
            Value::Range {
                start,
                end,
                inclusive,
            } => PureValue::Range {
                start,
                end,
                inclusive,
            },
            Value::List(items) => {
                PureValue::List(items.into_iter().map(Value::into_pure).collect())
            }
            Value::Object(fields) => PureValue::Object(
                fields
                    .into_iter()
                    .map(|(k, v)| (k, v.into_pure()))
                    .collect(),
            ),
            Value::Tuple(elems) => {
                PureValue::Tuple(elems.into_iter().map(Value::into_pure).collect())
            }
            Value::Byte(b) => PureValue::Byte(b),
            Value::Fn(_) => panic!("cannot convert Fn to PureValue"),
            Value::Opaque(o) => panic!("cannot convert Opaque<{}> to PureValue", o.type_name),
        }
    }
}
