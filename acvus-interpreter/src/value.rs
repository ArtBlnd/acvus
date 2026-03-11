use std::any::Any;

use std::fmt;
use std::sync::Arc;

use acvus_mir::ir::Label;
use acvus_utils::Astr;
use rustc_hash::FxHashMap;

/// Data-only value — no functions, no closures.
/// Cloneable, used at context boundaries.
///
/// For serialization, convert to [`ConcreteValue`] via [`PureValue::to_concrete`].
/// `Astr` fields require an interner for resolution.
#[derive(Debug, Clone, PartialEq)]
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
    Object(FxHashMap<Astr, PureValue>),
    Tuple(Vec<PureValue>),
    Byte(u8),
    Variant {
        tag: Astr,
        payload: Option<Box<PureValue>>,
    },
}

/// Serialization-safe mirror of [`PureValue`].
///
/// All `Astr` fields are resolved to `String`. Derives `Serialize`/`Deserialize`
/// with `#[serde(tag = "t")]` so the JSON format is self-describing and round-trips
/// correctly (no ambiguity between e.g. String vs Variant).
///
/// When adding a new variant to [`PureValue`], add the corresponding variant here
/// and update `to_concrete`/`from_concrete` — the compiler enforces exhaustive matching.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "t")]
pub enum ConcreteValue {
    Int { v: i64 },
    Float { v: f64 },
    String { v: String },
    Bool { v: bool },
    Unit,
    Range { start: i64, end: i64, inclusive: bool },
    List { items: Vec<ConcreteValue> },
    Object { fields: Vec<(String, ConcreteValue)> },
    Tuple { items: Vec<ConcreteValue> },
    Byte { v: u8 },
    Variant { tag: String, payload: Option<Box<ConcreteValue>> },
}

impl PureValue {
    /// Convert to a serialization-safe [`ConcreteValue`].
    /// Every variant is matched explicitly — no `_` catch-all.
    pub fn to_concrete(&self, interner: &acvus_utils::Interner) -> ConcreteValue {
        match self {
            PureValue::Int(v) => ConcreteValue::Int { v: *v },
            PureValue::Float(v) => ConcreteValue::Float { v: *v },
            PureValue::String(v) => ConcreteValue::String { v: v.clone() },
            PureValue::Bool(v) => ConcreteValue::Bool { v: *v },
            PureValue::Unit => ConcreteValue::Unit,
            PureValue::Range { start, end, inclusive } => ConcreteValue::Range {
                start: *start,
                end: *end,
                inclusive: *inclusive,
            },
            PureValue::List(items) => ConcreteValue::List {
                items: items.iter().map(|i| i.to_concrete(interner)).collect(),
            },
            PureValue::Object(fields) => ConcreteValue::Object {
                fields: fields
                    .iter()
                    .map(|(k, v)| (interner.resolve(*k).to_string(), v.to_concrete(interner)))
                    .collect(),
            },
            PureValue::Tuple(items) => ConcreteValue::Tuple {
                items: items.iter().map(|i| i.to_concrete(interner)).collect(),
            },
            PureValue::Byte(v) => ConcreteValue::Byte { v: *v },
            PureValue::Variant { tag, payload } => ConcreteValue::Variant {
                tag: interner.resolve(*tag).to_string(),
                payload: payload.as_ref().map(|p| Box::new(p.to_concrete(interner))),
            },
        }
    }

    /// Restore from a [`ConcreteValue`].
    /// Every variant is matched explicitly — no `_` catch-all.
    pub fn from_concrete(cv: &ConcreteValue, interner: &acvus_utils::Interner) -> Self {
        match cv {
            ConcreteValue::Int { v } => PureValue::Int(*v),
            ConcreteValue::Float { v } => PureValue::Float(*v),
            ConcreteValue::String { v } => PureValue::String(v.clone()),
            ConcreteValue::Bool { v } => PureValue::Bool(*v),
            ConcreteValue::Unit => PureValue::Unit,
            ConcreteValue::Range { start, end, inclusive } => PureValue::Range {
                start: *start,
                end: *end,
                inclusive: *inclusive,
            },
            ConcreteValue::List { items } => PureValue::List(
                items.iter().map(|i| PureValue::from_concrete(i, interner)).collect(),
            ),
            ConcreteValue::Object { fields } => PureValue::Object(
                fields
                    .iter()
                    .map(|(k, v)| (interner.intern(k), PureValue::from_concrete(v, interner)))
                    .collect(),
            ),
            ConcreteValue::Tuple { items } => PureValue::Tuple(
                items.iter().map(|i| PureValue::from_concrete(i, interner)).collect(),
            ),
            ConcreteValue::Byte { v } => PureValue::Byte(*v),
            ConcreteValue::Variant { tag, payload } => PureValue::Variant {
                tag: interner.intern(tag),
                payload: payload.as_ref().map(|p| Box::new(PureValue::from_concrete(p, interner))),
            },
        }
    }
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
    Object(FxHashMap<Astr, Value>),
    Tuple(Vec<Value>),
    Byte(u8),
    Variant {
        tag: Astr,
        payload: Option<Box<Value>>,
    },
    Fn(FnValue),
    /// Opaque handle to an extern function, identified by name.
    /// Produced by ContextLoad for extern/function-node entries.
    ExternFn(Astr),
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
            PureValue::Variant { tag, payload } => Value::Variant {
                tag,
                payload: payload.map(|p| Box::new(Value::from_pure(*p))),
            },
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
            Value::Variant { tag, payload } => PureValue::Variant {
                tag,
                payload: payload.map(|p| Box::new((*p).into_pure())),
            },
            Value::Fn(_) => panic!("cannot convert Fn to PureValue"),
            Value::ExternFn(_) => panic!("cannot convert ExternFn to PureValue"),
            Value::Opaque(o) => panic!("cannot convert Opaque<{}> to PureValue", o.type_name),
        }
    }
}
