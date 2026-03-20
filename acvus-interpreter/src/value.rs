use std::any::Any;

use std::fmt;
use std::sync::Arc;

use acvus_mir::ir::ClosureBody;
use acvus_mir::ty::{Effect, FnKind, Ty};
use acvus_utils::{Astr, TrackedDeque};
use rustc_hash::FxHashMap;

use crate::iter::{IterHandle, SequenceChain};

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
    Sequence { items: Vec<ConcreteValue> },
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

/// Tuple value — heterogeneous, positional elements.
///
/// Distinct from `List`/`Deque` (homogeneous, variable-length).
/// Each position may have a different type at the type level.
#[derive(Debug, Clone, PartialEq)]
pub struct Tuple(pub Vec<Value>);

/// Container and callable values — may contain any Value tier inside.
#[derive(Debug, Clone, PartialEq)]
pub enum LazyValue {
    List(Vec<Value>),
    Deque(TrackedDeque<Value>),
    Object(FxHashMap<Astr, Value>),
    Tuple(Tuple),
    Variant { tag: Astr, payload: Option<Box<Value>> },
    Fn(FnValue),
    ExternFn(Astr),
    Iterator(IterHandle),
    Sequence(SequenceChain),
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

impl Value {
    /// Lightweight discriminant for error reporting.
    pub fn kind(&self) -> crate::error::ValueKind {
        use crate::error::ValueKind;
        match self {
            Value::Pure(PureValue::Int(_)) => ValueKind::Int,
            Value::Pure(PureValue::Float(_)) => ValueKind::Float,
            Value::Pure(PureValue::String(_)) => ValueKind::String,
            Value::Pure(PureValue::Bool(_)) => ValueKind::Bool,
            Value::Pure(PureValue::Unit) => ValueKind::Unit,
            Value::Pure(PureValue::Range { .. }) => ValueKind::Range,
            Value::Pure(PureValue::Byte(_)) => ValueKind::Byte,
            Value::Lazy(LazyValue::List(_)) => ValueKind::List,
            Value::Lazy(LazyValue::Deque(_)) => ValueKind::Deque,
            Value::Lazy(LazyValue::Object(_)) => ValueKind::Object,
            Value::Lazy(LazyValue::Tuple(_)) => ValueKind::Tuple,
            Value::Lazy(LazyValue::Variant { .. }) => ValueKind::Variant,
            Value::Lazy(LazyValue::Fn(_)) => ValueKind::Fn,
            Value::Lazy(LazyValue::ExternFn(_)) => ValueKind::ExternFn,
            Value::Lazy(LazyValue::Iterator(_)) => ValueKind::Iterator,
            Value::Lazy(LazyValue::Sequence(_)) => ValueKind::Sequence,
            Value::Unpure(UnpureValue::Opaque(_)) => ValueKind::Opaque,
        }
    }
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
    pub fn string(s: impl Into<String>) -> Self { Value::Pure(PureValue::String(s.into())) }
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
    pub fn tuple(elems: Vec<Value>) -> Self { Value::Lazy(LazyValue::Tuple(Tuple(elems))) }
    pub fn variant(tag: Astr, payload: Option<Box<Value>>) -> Self {
        Value::Lazy(LazyValue::Variant { tag, payload })
    }
    pub fn closure(fv: FnValue) -> Self { Value::Lazy(LazyValue::Fn(fv)) }
    pub fn extern_fn(name: Astr) -> Self { Value::Lazy(LazyValue::ExternFn(name)) }
    pub fn iterator(ih: IterHandle) -> Self { Value::Lazy(LazyValue::Iterator(ih)) }
    pub fn sequence(sc: SequenceChain) -> Self { Value::Lazy(LazyValue::Sequence(sc)) }

    // --- Unpure constructors ---
    pub fn opaque(ov: OpaqueValue) -> Self { Value::Unpure(UnpureValue::Opaque(ov)) }

    // --- Extraction ---

    /// Consume and extract an inner type. Panics if the variant doesn't match.
    ///
    /// Use when the type checker guarantees the variant — a panic indicates
    /// a programmer bug, not a user error.
    pub fn expect<T: FromValue>(self, ctx: &'static str) -> T {
        T::from_value(self)
            .unwrap_or_else(|| panic!("{ctx}: expected {}", T::KIND_NAME))
    }

    /// Consume and try to extract an inner type. Returns `None` if the variant
    /// doesn't match.
    pub fn try_expect<T: FromValue>(self) -> Option<T> {
        T::from_value(self)
    }

    /// Borrow and extract a reference to an inner type. Panics if the variant
    /// doesn't match.
    pub fn expect_ref<T: FromValueRef + ?Sized>(&self, ctx: &'static str) -> &T {
        T::from_value_ref(self)
            .unwrap_or_else(|| panic!("{ctx}: expected {}", T::KIND_NAME))
    }

    /// Borrow and try to extract a reference to an inner type. Returns `None`
    /// if the variant doesn't match.
    pub fn try_expect_ref<T: FromValueRef + ?Sized>(&self) -> Option<&T> {
        T::from_value_ref(self)
    }

    /// Structural equality for language-level `==` and pattern matching.
    ///
    /// Compares values by structure. Each variant only compares with itself —
    /// Cast ensures type consistency before comparison.
    /// Functions, iterators, sequences, and opaque values are never equal.
    pub fn structural_eq(&self, other: &Value) -> bool {
        match (self, other) {
            // Pure scalars — delegate to derived PartialEq (handles Range too)
            (Value::Pure(a), Value::Pure(b)) => a == b,

            // List/Deque — treat as equivalent sequences
            (Value::Lazy(a), Value::Lazy(b)) => lazy_structural_eq(a, b),

            // Pure vs Lazy, or anything involving Unpure → not equal
            _ => false,
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
                LazyValue::Tuple(tuple) => ConcreteValue::Tuple {
                    items: tuple.0.iter().map(|i| i.to_concrete(interner)).collect(),
                },
                LazyValue::Variant { tag, payload } => ConcreteValue::Variant {
                    tag: interner.resolve(*tag).to_string(),
                    payload: payload.as_ref().map(|p| Box::new(p.to_concrete(interner))),
                },
                LazyValue::Fn(_) => panic!("cannot convert Fn to ConcreteValue"),
                LazyValue::ExternFn(_) => panic!("cannot convert ExternFn to ConcreteValue"),
                LazyValue::Iterator(_) => panic!("cannot convert Iterator to ConcreteValue"),
                LazyValue::Sequence(sc) => ConcreteValue::Sequence {
                    items: sc.origin().as_slice().iter().map(|i| i.to_concrete(interner)).collect(),
                },
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
            ConcreteValue::Sequence { items } => {
                let values: Vec<Value> = items.iter().map(|i| Value::from_concrete(i, interner)).collect();
                let deque = acvus_utils::TrackedDeque::from_vec(values);
                Value::sequence(crate::iter::SequenceChain::from_stored(deque))
            }
        }
    }
}

// =========================================================================
// TypedValue — Value + Ty pair for use at storage/orchestration boundaries
// =========================================================================

/// A runtime value paired with its compile-time type.
///
/// Used at **boundaries** between the interpreter and external systems
/// (storage, orchestration, node-to-node passing). The interpreter's internal
/// execution uses bare [`Value`] for performance; `TypedValue` is constructed
/// when values leave the interpreter.
///
/// # Invariant
///
/// The `value` must be consistent with `ty`. This is enforced by a
/// `debug_assert` in [`TypedValue::new`] — violations are programming bugs,
/// not user errors.
///
/// # Storability
///
/// [`TypedValue::is_storable`] checks whether the value can be persisted:
/// - Pure scalars: always OK.
/// - Lazy containers with Pure effect: OK (Iterator/Sequence are collected).
/// - Effectful or Unpure: rejected.
pub struct TypedValue {
    value: Arc<Value>,
    ty: Ty,
}

impl TypedValue {
    /// Create a TypedValue from an owned Value. Wraps in Arc internally.
    pub fn new(value: Value, ty: Ty) -> Self {
        debug_assert!(
            value_matches_ty(&value, &ty),
            "TypedValue invariant violated: value={value:?}, ty={ty:?}"
        );
        Self { value: Arc::new(value), ty }
    }

    /// Create a TypedValue from a shared Arc<Value>.
    pub fn new_shared(value: Arc<Value>, ty: Ty) -> Self {
        debug_assert!(
            value_matches_ty(&value, &ty),
            "TypedValue invariant violated: value={value:?}, ty={ty:?}"
        );
        Self { value, ty }
    }

    /// Borrow the inner value.
    pub fn value(&self) -> &Value { &self.value }

    /// Borrow the type.
    pub fn ty(&self) -> &Ty { &self.ty }

    /// Consume and return the inner `Value`, discarding the type.
    ///
    /// Unwraps the internal `Arc<Value>` — if this is the only reference,
    /// avoids a clone; otherwise clones the value.
    pub fn into_inner(self) -> Value { Arc::unwrap_or_clone(self.value) }

    /// Consume and return both parts as owned values.
    pub fn into_inner_with_ty(self) -> (Value, Ty) { (Arc::unwrap_or_clone(self.value), self.ty) }

    /// Whether this value can be persisted to storage.
    /// Delegates to [`Ty::is_storable`].
    pub fn is_storable(&self) -> bool {
        self.ty.is_storable()
    }

    // --- Pure scalar constructors ---
    pub fn int(n: i64) -> Self { Self { value: Arc::new(Value::int(n)), ty: Ty::Int } }
    pub fn float(f: f64) -> Self { Self { value: Arc::new(Value::float(f)), ty: Ty::Float } }
    pub fn string(s: impl Into<String>) -> Self { Self { value: Arc::new(Value::string(s.into())), ty: Ty::String } }
    pub fn bool_(b: bool) -> Self { Self { value: Arc::new(Value::bool_(b)), ty: Ty::Bool } }
    pub fn unit() -> Self { Self { value: Arc::new(Value::unit()), ty: Ty::Unit } }
    pub fn byte(b: u8) -> Self { Self { value: Arc::new(Value::byte(b)), ty: Ty::Byte } }

    /// Convert to a serialization-safe [`ConcreteValue`].
    /// Delegates to [`Value::to_concrete`].
    pub fn to_concrete(&self, interner: &acvus_utils::Interner) -> ConcreteValue {
        self.value.to_concrete(interner)
    }

    /// Restore a TypedValue from a [`ConcreteValue`].
    /// The type must be provided externally since ConcreteValue is untyped.
    ///
    /// `ConcreteValue::Sequence` → SequenceChain directly via `Value::from_concrete`.
    pub fn from_concrete(cv: &ConcreteValue, interner: &acvus_utils::Interner, ty: Ty) -> Self {
        Self::new(Value::from_concrete(cv, interner), ty)
    }
}

impl PartialEq for TypedValue {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl fmt::Debug for TypedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TypedValue({:?})", self.value)
    }
}

impl Clone for TypedValue {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
            ty: self.ty.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Structural equality helpers
// ---------------------------------------------------------------------------

/// Compare two slices of Values structurally.
fn slice_structural_eq(a: &[Value], b: &[Value]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.structural_eq(y))
}

/// Structural equality for LazyValue.
///
/// Each variant only compares with itself — Cast ensures type consistency.
/// Fn, ExternFn, Iterator, Sequence are never equal.
fn lazy_structural_eq(a: &LazyValue, b: &LazyValue) -> bool {
    match (a, b) {
        // List ↔ List
        (LazyValue::List(a), LazyValue::List(b)) => slice_structural_eq(a, b),
        // Deque ↔ Deque
        (LazyValue::Deque(a), LazyValue::Deque(b)) => {
            slice_structural_eq(a.as_slice(), b.as_slice())
        }
        // Tuple
        (LazyValue::Tuple(a), LazyValue::Tuple(b)) => slice_structural_eq(&a.0, &b.0),
        // Object — unordered field comparison
        (LazyValue::Object(a), LazyValue::Object(b)) => {
            a.len() == b.len()
                && a.iter().all(|(k, v)| b.get(k).is_some_and(|bv| v.structural_eq(bv)))
        }
        // Variant — tag + payload
        (
            LazyValue::Variant {
                tag: ta,
                payload: pa,
            },
            LazyValue::Variant {
                tag: tb,
                payload: pb,
            },
        ) => {
            ta == tb
                && match (pa, pb) {
                    (Some(a), Some(b)) => a.structural_eq(b),
                    (None, None) => true,
                    _ => false,
                }
        }
        // Fn, ExternFn, Iterator, Sequence — not comparable
        _ => false,
    }
}

/// Shallow check that a Value variant matches its Ty.
///
/// This is a **debug-only sanity check**, not a full recursive validation.
/// It verifies that the top-level Value constructor corresponds to the top-level
/// Ty constructor (e.g. `PureValue::Int` ↔ `Ty::Int`). Container contents are
/// not recursively checked — that would be too expensive for a debug assert.
fn value_matches_ty(value: &Value, ty: &Ty) -> bool {
    // Error/Infer/Var types accept any value (unresolved or poison types).
    if matches!(ty, Ty::Error(_) | Ty::Infer(_) | Ty::Var(_)) {
        return true;
    }
    match (value, ty) {
        (Value::Pure(PureValue::Int(_)), Ty::Int) => true,
        (Value::Pure(PureValue::Float(_)), Ty::Float) => true,
        (Value::Pure(PureValue::String(_)), Ty::String) => true,
        (Value::Pure(PureValue::Bool(_)), Ty::Bool) => true,
        (Value::Pure(PureValue::Unit), Ty::Unit) => true,
        (Value::Pure(PureValue::Range { .. }), Ty::Range) => true,
        (Value::Pure(PureValue::Byte(_)), Ty::Byte) => true,
        (Value::Lazy(LazyValue::List(_)), Ty::List(_)) => true,
        (Value::Lazy(LazyValue::Deque(_)), Ty::Deque(..)) => true,
        (Value::Lazy(LazyValue::Object(_)), Ty::Object(_)) => true,
        (Value::Lazy(LazyValue::Tuple(_)), Ty::Tuple(_)) => true,
        (Value::Lazy(LazyValue::Variant { .. }), Ty::Option(_)) => true,
        (Value::Lazy(LazyValue::Variant { .. }), Ty::Enum { .. }) => true,
        (Value::Lazy(LazyValue::Fn(_)), Ty::Fn { kind: FnKind::Lambda, .. }) => true,
        (Value::Lazy(LazyValue::ExternFn(_)), Ty::Fn { kind: FnKind::Extern, .. }) => true,
        (Value::Lazy(LazyValue::Iterator(_)), Ty::Iterator(..)) => true,
        (Value::Lazy(LazyValue::Sequence(_)), Ty::Sequence(..)) => true,
        (Value::Unpure(UnpureValue::Opaque(_)), Ty::Opaque(_)) => true,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// FromValue — owned extraction (consumes the Value)
// ---------------------------------------------------------------------------

/// Extract an owned inner type from a `Value`.
///
/// Used with `Value::expect` for cases where the type checker guarantees the
/// variant. A `None` return indicates a programmer bug.
pub trait FromValue: Sized {
    const KIND_NAME: &'static str;
    fn from_value(value: Value) -> Option<Self>;
}

// --- Scalars ---

impl FromValue for i64 {
    const KIND_NAME: &'static str = "Int";
    fn from_value(value: Value) -> Option<Self> {
        match value { Value::Pure(PureValue::Int(n)) => Some(n), _ => None }
    }
}

impl FromValue for f64 {
    const KIND_NAME: &'static str = "Float";
    fn from_value(value: Value) -> Option<Self> {
        match value { Value::Pure(PureValue::Float(f)) => Some(f), _ => None }
    }
}

impl FromValue for String {
    const KIND_NAME: &'static str = "String";
    fn from_value(value: Value) -> Option<Self> {
        match value { Value::Pure(PureValue::String(s)) => Some(s), _ => None }
    }
}

impl FromValue for bool {
    const KIND_NAME: &'static str = "Bool";
    fn from_value(value: Value) -> Option<Self> {
        match value { Value::Pure(PureValue::Bool(b)) => Some(b), _ => None }
    }
}

impl FromValue for u8 {
    const KIND_NAME: &'static str = "Byte";
    fn from_value(value: Value) -> Option<Self> {
        match value { Value::Pure(PureValue::Byte(b)) => Some(b), _ => None }
    }
}

// --- Containers ---

impl FromValue for TrackedDeque<Value> {
    const KIND_NAME: &'static str = "Deque";
    fn from_value(value: Value) -> Option<Self> {
        match value { Value::Lazy(LazyValue::Deque(d)) => Some(d), _ => None }
    }
}

impl FromValue for Vec<Value> {
    const KIND_NAME: &'static str = "List";
    fn from_value(value: Value) -> Option<Self> {
        match value { Value::Lazy(LazyValue::List(v)) => Some(v), _ => None }
    }
}

impl FromValue for Tuple {
    const KIND_NAME: &'static str = "Tuple";
    fn from_value(value: Value) -> Option<Self> {
        match value { Value::Lazy(LazyValue::Tuple(t)) => Some(t), _ => None }
    }
}

impl FromValue for FxHashMap<Astr, Value> {
    const KIND_NAME: &'static str = "Object";
    fn from_value(value: Value) -> Option<Self> {
        match value { Value::Lazy(LazyValue::Object(o)) => Some(o), _ => None }
    }
}

impl FromValue for FnValue {
    const KIND_NAME: &'static str = "Fn";
    fn from_value(value: Value) -> Option<Self> {
        match value { Value::Lazy(LazyValue::Fn(f)) => Some(f), _ => None }
    }
}

impl FromValue for IterHandle {
    const KIND_NAME: &'static str = "Iterator";
    fn from_value(value: Value) -> Option<Self> {
        match value { Value::Lazy(LazyValue::Iterator(ih)) => Some(ih), _ => None }
    }
}

impl FromValue for SequenceChain {
    const KIND_NAME: &'static str = "Sequence";
    fn from_value(value: Value) -> Option<Self> {
        match value { Value::Lazy(LazyValue::Sequence(sc)) => Some(sc), _ => None }
    }
}

// ---------------------------------------------------------------------------
// FromValueRef — borrowed extraction (borrows the Value)
// ---------------------------------------------------------------------------

/// Extract a borrowed reference to an inner type from a `&Value`.
///
/// Used with `Value::expect_ref` for read-only access where the type checker
/// guarantees the variant.
pub trait FromValueRef {
    const KIND_NAME: &'static str;
    fn from_value_ref(value: &Value) -> Option<&Self>;
}

impl FromValueRef for i64 {
    const KIND_NAME: &'static str = "Int";
    fn from_value_ref(value: &Value) -> Option<&Self> {
        match value { Value::Pure(PureValue::Int(n)) => Some(n), _ => None }
    }
}

impl FromValueRef for f64 {
    const KIND_NAME: &'static str = "Float";
    fn from_value_ref(value: &Value) -> Option<&Self> {
        match value { Value::Pure(PureValue::Float(f)) => Some(f), _ => None }
    }
}

impl FromValueRef for str {
    const KIND_NAME: &'static str = "String";
    fn from_value_ref(value: &Value) -> Option<&Self> {
        match value { Value::Pure(PureValue::String(s)) => Some(s.as_str()), _ => None }
    }
}

impl FromValueRef for bool {
    const KIND_NAME: &'static str = "Bool";
    fn from_value_ref(value: &Value) -> Option<&Self> {
        match value { Value::Pure(PureValue::Bool(b)) => Some(b), _ => None }
    }
}

impl FromValueRef for TrackedDeque<Value> {
    const KIND_NAME: &'static str = "Deque";
    fn from_value_ref(value: &Value) -> Option<&Self> {
        match value { Value::Lazy(LazyValue::Deque(d)) => Some(d), _ => None }
    }
}

impl FromValueRef for [Value] {
    const KIND_NAME: &'static str = "List";
    fn from_value_ref(value: &Value) -> Option<&Self> {
        match value { Value::Lazy(LazyValue::List(v)) => Some(v.as_slice()), _ => None }
    }
}

impl FromValueRef for Tuple {
    const KIND_NAME: &'static str = "Tuple";
    fn from_value_ref(value: &Value) -> Option<&Self> {
        match value { Value::Lazy(LazyValue::Tuple(t)) => Some(t), _ => None }
    }
}

impl FromValueRef for FxHashMap<Astr, Value> {
    const KIND_NAME: &'static str = "Object";
    fn from_value_ref(value: &Value) -> Option<&Self> {
        match value { Value::Lazy(LazyValue::Object(o)) => Some(o), _ => None }
    }
}

impl FromValueRef for FnValue {
    const KIND_NAME: &'static str = "Fn";
    fn from_value_ref(value: &Value) -> Option<&Self> {
        match value { Value::Lazy(LazyValue::Fn(f)) => Some(f), _ => None }
    }
}

impl FromValueRef for IterHandle {
    const KIND_NAME: &'static str = "Iterator";
    fn from_value_ref(value: &Value) -> Option<&Self> {
        match value { Value::Lazy(LazyValue::Iterator(ih)) => Some(ih), _ => None }
    }
}

impl FromValueRef for SequenceChain {
    const KIND_NAME: &'static str = "Sequence";
    fn from_value_ref(value: &Value) -> Option<&Self> {
        match value { Value::Lazy(LazyValue::Sequence(sc)) => Some(sc), _ => None }
    }
}
