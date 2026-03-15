use std::any::Any;

use std::fmt;
use std::sync::Arc;

use acvus_mir::ir::ClosureBody;
use acvus_utils::{Astr, TrackedDeque};
use rustc_hash::FxHashMap;

use crate::iter::SharedIter;

/// Scalar-only data value — no containers, no functions, no closures.
/// Cloneable, used at context boundaries.
///
/// For serialization, convert to [`ConcreteValue`] via [`PureValue::to_concrete`].
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
    Byte(u8),
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
    pub fn to_concrete(&self, _interner: &acvus_utils::Interner) -> ConcreteValue {
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
            PureValue::Byte(v) => ConcreteValue::Byte { v: *v },
        }
    }

    /// Restore from a [`ConcreteValue`].
    /// Only handles scalar variants — container variants are handled by [`Value::from_concrete`].
    pub fn from_concrete(cv: &ConcreteValue, _interner: &acvus_utils::Interner) -> Self {
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
            ConcreteValue::Byte { v } => PureValue::Byte(*v),
            _ => panic!("PureValue::from_concrete: container variants moved to Value::from_concrete"),
        }
    }
}

/// Container and callable values — may contain any Value tier inside.
#[derive(Debug, Clone, PartialEq)]
pub enum LazyValue {
    List(Vec<Value>),
    Deque(TrackedDeque<Value>),
    Object(FxHashMap<Astr, Value>),
    Tuple(Vec<Value>),
    Variant { tag: Astr, payload: Option<Box<Value>> },
    Fn(FnValue),
    ExternFn(Astr),
    Iterator(SharedIter),
    Sequence(SharedIter),
}

/// Values that cannot cross context boundaries.
#[derive(Clone)]
pub enum UnpureValue {
    Opaque(OpaqueValue),
}

impl fmt::Debug for UnpureValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnpureValue::Opaque(o) => write!(f, "{o:?}"),
        }
    }
}

impl PartialEq for UnpureValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (UnpureValue::Opaque(a), UnpureValue::Opaque(b)) => a == b,
        }
    }
}

/// Runtime value — 3-tier enum for purity-aware dispatch.
///
/// `PartialEq` compares structural equality for data variants.
/// `Fn`, `ExternFn`, and `Opaque` are never equal (not comparable).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Pure(PureValue),
    Lazy(LazyValue),
    Unpure(UnpureValue),
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

/// A closure value: self-contained body + captured values.
#[derive(Debug, Clone)]
pub struct FnValue {
    pub body: Arc<ClosureBody>,
    pub captures: Vec<Arc<Value>>,
}

impl PartialEq for FnValue {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl Value {
    // --- Pure constructors ---
    pub fn int(n: i64) -> Self { Value::Pure(PureValue::Int(n)) }
    pub fn float(f: f64) -> Self { Value::Pure(PureValue::Float(f)) }
    pub fn string(s: String) -> Self { Value::Pure(PureValue::String(s)) }
    pub fn bool_(b: bool) -> Self { Value::Pure(PureValue::Bool(b)) }
    pub fn unit() -> Self { Value::Pure(PureValue::Unit) }
    pub fn byte(b: u8) -> Self { Value::Pure(PureValue::Byte(b)) }
    pub fn range(start: i64, end: i64, inclusive: bool) -> Self {
        Value::Pure(PureValue::Range { start, end, inclusive })
    }

    // --- Lazy constructors ---
    pub fn list(items: Vec<Value>) -> Self { Value::Lazy(LazyValue::List(items)) }
    pub fn deque(d: TrackedDeque<Value>) -> Self { Value::Lazy(LazyValue::Deque(d)) }
    pub fn object(fields: FxHashMap<Astr, Value>) -> Self { Value::Lazy(LazyValue::Object(fields)) }
    pub fn tuple(elems: Vec<Value>) -> Self { Value::Lazy(LazyValue::Tuple(elems)) }
    pub fn variant(tag: Astr, payload: Option<Box<Value>>) -> Self {
        Value::Lazy(LazyValue::Variant { tag, payload })
    }
    pub fn closure(fv: FnValue) -> Self { Value::Lazy(LazyValue::Fn(fv)) }
    pub fn extern_fn(name: Astr) -> Self { Value::Lazy(LazyValue::ExternFn(name)) }
    pub fn iterator(si: SharedIter) -> Self { Value::Lazy(LazyValue::Iterator(si)) }
    pub fn sequence(si: SharedIter) -> Self { Value::Lazy(LazyValue::Sequence(si)) }

    // --- Unpure constructors ---
    pub fn opaque(ov: OpaqueValue) -> Self { Value::Unpure(UnpureValue::Opaque(ov)) }

    /// Coerce into a `SharedIter`.
    ///
    /// Mirrors the type-level `Deque → Iterator` coercion:
    /// - `LazyValue::Iterator` is returned as-is.
    /// - `LazyValue::List` (runtime repr of Deque) is converted via `SharedIter::from_list`.
    ///
    /// Panics on any other variant.
    pub fn into_shared_iter(self) -> SharedIter {
        match self {
            Value::Lazy(LazyValue::Iterator(s)) => s,
            Value::Lazy(LazyValue::Sequence(s)) => s,
            Value::Lazy(LazyValue::List(items)) => SharedIter::from_list(items),
            Value::Lazy(LazyValue::Deque(deque)) => SharedIter::from_list(deque.into_vec()),
            other => panic!("into_shared_iter: expected Iterator, List, or Deque, got {other:?}"),
        }
    }

    /// Convert a PureValue into a Value. Infallible.
    pub fn from_pure(pure: PureValue) -> Self {
        Value::Pure(pure)
    }

    /// Convert a storeable Value to ConcreteValue for serialization.
    /// Panics on Unpure values (type checker guarantees this won't happen at boundaries).
    pub fn to_concrete(&self, interner: &acvus_utils::Interner) -> ConcreteValue {
        match self {
            Value::Pure(pv) => match pv {
                PureValue::Int(v) => ConcreteValue::Int { v: *v },
                PureValue::Float(v) => ConcreteValue::Float { v: *v },
                PureValue::String(v) => ConcreteValue::String { v: v.clone() },
                PureValue::Bool(v) => ConcreteValue::Bool { v: *v },
                PureValue::Unit => ConcreteValue::Unit,
                PureValue::Range { start, end, inclusive } => ConcreteValue::Range {
                    start: *start, end: *end, inclusive: *inclusive,
                },
                PureValue::Byte(v) => ConcreteValue::Byte { v: *v },
            },
            Value::Lazy(lv) => match lv {
                LazyValue::List(items) => ConcreteValue::List {
                    items: items.iter().map(|i| i.to_concrete(interner)).collect(),
                },
                LazyValue::Deque(deque) => ConcreteValue::List {
                    items: deque.as_slice().iter().map(|i| i.to_concrete(interner)).collect(),
                },
                LazyValue::Object(fields) => ConcreteValue::Object {
                    fields: fields.iter()
                        .map(|(k, v)| (interner.resolve(*k).to_string(), v.to_concrete(interner)))
                        .collect(),
                },
                LazyValue::Tuple(items) => ConcreteValue::Tuple {
                    items: items.iter().map(|i| i.to_concrete(interner)).collect(),
                },
                LazyValue::Variant { tag, payload } => ConcreteValue::Variant {
                    tag: interner.resolve(*tag).to_string(),
                    payload: payload.as_ref().map(|p| Box::new(p.to_concrete(interner))),
                },
                LazyValue::Fn(_) => panic!("cannot convert Fn to ConcreteValue"),
                LazyValue::ExternFn(_) => panic!("cannot convert ExternFn to ConcreteValue"),
                LazyValue::Iterator(_) => panic!("cannot convert Iterator to ConcreteValue"),
                LazyValue::Sequence(_) => panic!("cannot convert Sequence to ConcreteValue"),
            },
            Value::Unpure(uv) => match uv {
                UnpureValue::Opaque(o) => panic!("cannot convert Opaque<{}> to ConcreteValue", o.type_name),
            },
        }
    }

    /// Restore a Value from a ConcreteValue.
    pub fn from_concrete(cv: &ConcreteValue, interner: &acvus_utils::Interner) -> Value {
        match cv {
            ConcreteValue::Int { v } => Value::int(*v),
            ConcreteValue::Float { v } => Value::float(*v),
            ConcreteValue::String { v } => Value::string(v.clone()),
            ConcreteValue::Bool { v } => Value::bool_(*v),
            ConcreteValue::Unit => Value::unit(),
            ConcreteValue::Range { start, end, inclusive } => Value::range(*start, *end, *inclusive),
            ConcreteValue::List { items } => Value::list(
                items.iter().map(|i| Value::from_concrete(i, interner)).collect(),
            ),
            ConcreteValue::Object { fields } => Value::object(
                fields.iter()
                    .map(|(k, v)| (interner.intern(k), Value::from_concrete(v, interner)))
                    .collect(),
            ),
            ConcreteValue::Tuple { items } => Value::tuple(
                items.iter().map(|i| Value::from_concrete(i, interner)).collect(),
            ),
            ConcreteValue::Byte { v } => Value::byte(*v),
            ConcreteValue::Variant { tag, payload } => Value::variant(
                interner.intern(tag),
                payload.as_ref().map(|p| Box::new(Value::from_concrete(p, interner))),
            ),
        }
    }
}
