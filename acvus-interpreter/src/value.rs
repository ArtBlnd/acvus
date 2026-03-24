use std::any::Any;
use std::fmt;
use std::sync::Arc;

use acvus_mir::ir::MirBody;
use acvus_utils::{Astr, Interner, TrackedDeque};
use rustc_hash::FxHashMap;

pub use crate::iter::{IterHandle, SequenceChain};

// ── Value ────────────────────────────────────────────────────────────

/// Runtime value. Flat enum — no nested tiers.
///
/// # Layout (16 bytes on 64-bit)
///
/// - **Inline**: Int, Float, Bool, Unit, Byte — no heap allocation.
/// - **Shared**: String, List, Object, Tuple, Deque, Variant — `Arc` wrapped,
///   clone = refcount bump. CoW via `Arc::make_mut` when mutation needed.
/// - **Owned**: Fn, Iterator, Sequence, Handle — `Box` wrapped, move-only.
///   SSA guarantees single use; `take()` replaces with `Empty`.
/// - **Opaque**: extern boundary values.
///
/// `Empty` is the moved-out sentinel. Accessing an `Empty` register is a
/// programmer bug (SSA guarantees this cannot happen). Debug-asserted.
pub enum Value {
    // ── Inline (no allocation) ───────────────────────────────────
    Empty,
    Int(i64),
    Float(f64),
    Bool(bool),
    Unit,
    Byte(u8),

    // ── Shared (Arc, clone = refcount bump) ──────────────────────
    String(Arc<str>),
    List(Arc<Vec<Value>>),
    Object(Arc<FxHashMap<Astr, Value>>),
    Tuple(Arc<Vec<Value>>),
    Deque(Arc<TrackedDeque<Value>>),
    Variant {
        tag: Astr,
        payload: Option<Arc<Value>>,
    },

    // ── Boxed (rarely used, keep enum small) ─────────────────────
    Range(Box<RangeValue>),

    // ── Owned (move-only, Box) ───────────────────────────────────
    Fn(Box<FnValue>),
    Iterator(Box<IterHandle>),
    Sequence(Box<SequenceChain>),
    Handle(Box<HandleValue>),

    // ── Opaque (extern boundary) ─────────────────────────────────
    Opaque(Box<OpaqueValue>),
}

// ── Satellite types ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct RangeValue {
    pub start: i64,
    pub end: i64,
    pub inclusive: bool,
}

/// A closure: body + captured values.
/// Captures are shared (Arc) since closures can be cloned.
#[derive(Debug, Clone)]
pub struct FnValue {
    pub body: Arc<MirBody>,
    pub captures: Arc<[Value]>,
}

/// A deferred computation handle (spawn result).
/// A deferred computation handle (spawn result).
/// Consumed exactly once by eval. Move-only.
/// Inner type is executor-specific (type-erased via Box<dyn Any>).
pub struct HandleValue {
    inner: Box<dyn std::any::Any + Send + Sync>,
}

impl HandleValue {
    pub fn new<T: std::any::Any + Send + Sync + 'static>(value: T) -> Self {
        Self {
            inner: Box::new(value),
        }
    }
    pub fn downcast<T: std::any::Any + Send + Sync>(self) -> T {
        *self.inner.downcast().expect("HandleValue type mismatch")
    }
}

impl std::fmt::Debug for HandleValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Handle")
    }
}

/// An opaque value from extern boundary.
pub struct OpaqueValue {
    pub type_name: &'static str,
    inner: Arc<dyn Any + Send + Sync>,
}

// ── Constructors ─────────────────────────────────────────────────────

impl Value {
    // Inline
    pub fn int(n: i64) -> Self {
        Value::Int(n)
    }
    pub fn float(f: f64) -> Self {
        Value::Float(f)
    }
    pub fn bool_(b: bool) -> Self {
        Value::Bool(b)
    }
    pub fn unit() -> Self {
        Value::Unit
    }
    pub fn byte(b: u8) -> Self {
        Value::Byte(b)
    }

    // Shared
    pub fn string(s: impl Into<Arc<str>>) -> Self {
        Value::String(s.into())
    }
    pub fn list(items: Vec<Value>) -> Self {
        Value::List(Arc::new(items))
    }
    pub fn object(fields: FxHashMap<Astr, Value>) -> Self {
        Value::Object(Arc::new(fields))
    }
    pub fn tuple(elems: Vec<Value>) -> Self {
        Value::Tuple(Arc::new(elems))
    }
    pub fn deque(d: TrackedDeque<Value>) -> Self {
        Value::Deque(Arc::new(d))
    }
    pub fn variant(tag: Astr, payload: Option<Value>) -> Self {
        Value::Variant {
            tag,
            payload: payload.map(Arc::new),
        }
    }

    // Option (well-known variants)
    pub fn some(interner: &Interner, payload: Value) -> Self {
        Value::Variant {
            tag: interner.intern("Some"),
            payload: Some(Arc::new(payload)),
        }
    }
    pub fn none(interner: &Interner) -> Self {
        Value::Variant {
            tag: interner.intern("None"),
            payload: None,
        }
    }

    // Boxed
    pub fn range(start: i64, end: i64, inclusive: bool) -> Self {
        Value::Range(Box::new(RangeValue {
            start,
            end,
            inclusive,
        }))
    }

    // Owned
    pub fn closure(fv: FnValue) -> Self {
        Value::Fn(Box::new(fv))
    }
    pub fn iterator(ih: IterHandle) -> Self {
        Value::Iterator(Box::new(ih))
    }
    pub fn sequence(sc: SequenceChain) -> Self {
        Value::Sequence(Box::new(sc))
    }

    // Opaque
    pub fn opaque(ov: OpaqueValue) -> Self {
        Value::Opaque(Box::new(ov))
    }
}

// ── Move / Clone ─────────────────────────────────────────────────────

impl Value {
    /// Take the value out, leaving `Empty` behind. For move-only semantics.
    #[inline]
    pub fn take(&mut self) -> Value {
        std::mem::replace(self, Value::Empty)
    }

    /// Alias for `clone()`. Prefer `take()` for move-only values.
    #[inline]
    pub fn share(&self) -> Value {
        self.clone()
    }

    /// Whether this value has been moved out.
    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(self, Value::Empty)
    }

    /// Lightweight discriminant for error reporting.
    pub fn kind(&self) -> crate::error::ValueKind {
        use crate::error::ValueKind;
        match self {
            Value::Empty => panic!("kind: accessed moved-out value"),
            Value::Int(_) => ValueKind::Int,
            Value::Float(_) => ValueKind::Float,
            Value::Bool(_) => ValueKind::Bool,
            Value::Unit => ValueKind::Unit,
            Value::Byte(_) => ValueKind::Byte,
            Value::String(_) => ValueKind::String,
            Value::List(_) => ValueKind::List,
            Value::Object(_) => ValueKind::Object,
            Value::Tuple(_) => ValueKind::Tuple,
            Value::Deque(_) => ValueKind::Deque,
            Value::Variant { .. } => ValueKind::Variant,
            Value::Range(_) => ValueKind::Range,
            Value::Fn(_) => ValueKind::Fn,
            Value::Iterator(_) => ValueKind::Iterator,
            Value::Sequence(_) => ValueKind::Sequence,
            Value::Handle(_) => ValueKind::Handle,
            Value::Opaque(_) => ValueKind::Opaque,
        }
    }
}

// ── Extraction (borrow) ──────────────────────────────────────────────

impl Value {
    #[inline]
    pub fn as_int(&self) -> i64 {
        match self {
            Value::Int(n) => *n,
            other => panic!("expected Int, got {other:?}"),
        }
    }
    #[inline]
    pub fn as_float(&self) -> f64 {
        match self {
            Value::Float(f) => *f,
            other => panic!("expected Float, got {other:?}"),
        }
    }
    #[inline]
    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            other => panic!("expected Bool, got {other:?}"),
        }
    }
    #[inline]
    pub fn as_str(&self) -> &str {
        match self {
            Value::String(s) => s,
            other => panic!("expected String, got {other:?}"),
        }
    }
    #[inline]
    pub fn as_byte(&self) -> u8 {
        match self {
            Value::Byte(b) => *b,
            other => panic!("expected Byte, got {other:?}"),
        }
    }
    #[inline]
    pub fn as_list(&self) -> &[Value] {
        match self {
            Value::List(l) => l,
            other => panic!("expected List, got {other:?}"),
        }
    }
    #[inline]
    pub fn as_object(&self) -> &FxHashMap<Astr, Value> {
        match self {
            Value::Object(o) => o,
            other => panic!("expected Object, got {other:?}"),
        }
    }
    #[inline]
    pub fn as_tuple(&self) -> &[Value] {
        match self {
            Value::Tuple(t) => t,
            other => panic!("expected Tuple, got {other:?}"),
        }
    }
    #[inline]
    pub fn as_range(&self) -> &RangeValue {
        match self {
            Value::Range(r) => r,
            other => panic!("expected Range, got {other:?}"),
        }
    }
}

// ── Extraction (owned — consumes the value) ──────────────────────────

impl Value {
    #[inline]
    pub fn into_string(self) -> Arc<str> {
        match self {
            Value::String(s) => s,
            other => panic!("expected String, got {other:?}"),
        }
    }
    #[inline]
    pub fn into_list(self) -> Arc<Vec<Value>> {
        match self {
            Value::List(l) => l,
            other => panic!("expected List, got {other:?}"),
        }
    }
    #[inline]
    pub fn into_object(self) -> Arc<FxHashMap<Astr, Value>> {
        match self {
            Value::Object(o) => o,
            other => panic!("expected Object, got {other:?}"),
        }
    }
    #[inline]
    pub fn into_fn(self) -> Box<FnValue> {
        match self {
            Value::Fn(f) => f,
            other => panic!("expected Fn, got {other:?}"),
        }
    }
    #[inline]
    pub fn into_iterator(self) -> Box<IterHandle> {
        match self {
            Value::Iterator(i) => i,
            other => panic!("expected Iterator, got {other:?}"),
        }
    }
    #[inline]
    pub fn into_sequence(self) -> Box<SequenceChain> {
        match self {
            Value::Sequence(s) => s,
            other => panic!("expected Sequence, got {other:?}"),
        }
    }
}

// ── Structural equality ──────────────────────────────────────────────

impl Value {
    /// Language-level `==` and pattern matching comparison.
    /// Functions, iterators, handles, opaques are never equal.
    pub fn structural_eq(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Unit, Value::Unit) => true,
            (Value::Byte(a), Value::Byte(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Range(a), Value::Range(b)) => a == b,

            (Value::List(a), Value::List(b)) => slice_eq(a, b),
            (Value::Tuple(a), Value::Tuple(b)) => slice_eq(a, b),
            (Value::Object(a), Value::Object(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .all(|(k, v)| b.get(k).is_some_and(|bv| v.structural_eq(bv)))
            }
            (Value::Deque(a), Value::Deque(b)) => slice_eq(a.as_slice(), b.as_slice()),

            (
                Value::Variant {
                    tag: ta,
                    payload: pa,
                },
                Value::Variant {
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

            _ => false,
        }
    }
}

fn slice_eq(a: &[Value], b: &[Value]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.structural_eq(y))
}

// ── Clone ────────────────────────────────────────────────────────────

impl Clone for Value {
    fn clone(&self) -> Self {
        match self {
            Value::Empty => panic!("clone: accessed moved-out value"),
            Value::Int(n) => Value::Int(*n),
            Value::Float(f) => Value::Float(*f),
            Value::Bool(b) => Value::Bool(*b),
            Value::Unit => Value::Unit,
            Value::Byte(b) => Value::Byte(*b),
            Value::String(s) => Value::String(Arc::clone(s)),
            Value::List(l) => Value::List(Arc::clone(l)),
            Value::Object(o) => Value::Object(Arc::clone(o)),
            Value::Tuple(t) => Value::Tuple(Arc::clone(t)),
            Value::Deque(d) => Value::Deque(Arc::clone(d)),
            Value::Variant { tag, payload } => Value::Variant {
                tag: *tag,
                payload: payload.as_ref().map(Arc::clone),
            },
            Value::Range(r) => Value::Range(r.clone()),
            Value::Fn(f) => Value::Fn(f.clone()),
            Value::Iterator(ih) => {
                if ih.effect().is_pure() {
                    Value::Iterator(Box::new(ih.as_ref().clone()))
                } else {
                    panic!("clone: effectful Iterator is move-only")
                }
            }
            Value::Sequence(sc) => {
                if sc.effect().is_pure() {
                    Value::Sequence(sc.clone())
                } else {
                    panic!("clone: effectful Sequence is move-only")
                }
            }
            Value::Handle(_) => panic!("clone: Handle is move-only"),
            Value::Opaque(o) => Value::Opaque(o.clone()),
        }
    }
}

// ── Debug / PartialEq ────────────────────────────────────────────────

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Empty => write!(f, "<empty>"),
            Value::Int(n) => write!(f, "{n}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Unit => write!(f, "()"),
            Value::Byte(b) => write!(f, "0x{b:02x}"),
            Value::String(s) => write!(f, "{s:?}"),
            Value::List(l) => f.debug_list().entries(l.iter()).finish(),
            Value::Object(o) => f.debug_map().entries(o.iter()).finish(),
            Value::Tuple(t) => {
                let mut d = f.debug_tuple("");
                for v in t.iter() {
                    d.field(v);
                }
                d.finish()
            }
            Value::Deque(d) => f.debug_list().entries(d.as_slice().iter()).finish(),
            Value::Variant { tag, payload } => match payload {
                Some(p) => write!(f, "{tag:?}({p:?})"),
                None => write!(f, "{tag:?}"),
            },
            Value::Range(r) => {
                if r.inclusive {
                    write!(f, "{}..={}", r.start, r.end)
                } else {
                    write!(f, "{}..{}", r.start, r.end)
                }
            }
            Value::Fn(fv) => write!(f, "Fn({} captures)", fv.captures.len()),
            Value::Iterator(_) => write!(f, "Iterator"),
            Value::Sequence(_) => write!(f, "Sequence"),
            Value::Handle(_) => write!(f, "Handle"),
            Value::Opaque(o) => write!(f, "Opaque<{}>", o.type_name),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.structural_eq(other)
    }
}

impl PartialEq for FnValue {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

// ── OpaqueValue ──────────────────────────────────────────────────────

impl Clone for OpaqueValue {
    fn clone(&self) -> Self {
        Self {
            type_name: self.type_name,
            inner: Arc::clone(&self.inner),
        }
    }
}

impl OpaqueValue {
    pub fn new<T: Any + Send + Sync>(type_name: &'static str, value: T) -> Self {
        Self {
            type_name,
            inner: Arc::new(value),
        }
    }

    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.inner.downcast_ref()
    }
}

impl fmt::Debug for OpaqueValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Opaque<{}>", self.type_name)
    }
}

impl PartialEq for OpaqueValue {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

// ── Size assertion ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_size() {
        let size = std::mem::size_of::<Value>();
        assert!(size <= 24, "Value enum is {size} bytes, expected <= 24");
        eprintln!("Value size: {size} bytes");
    }

    #[test]
    fn inline_no_alloc() {
        // These should not touch the heap.
        let _ = Value::int(42);
        let _ = Value::float(3.14);
        let _ = Value::bool_(true);
        let _ = Value::unit();
        let _ = Value::byte(0xff);
    }

    #[test]
    fn take_leaves_empty() {
        let mut v = Value::int(42);
        let taken = v.take();
        assert!(v.is_empty());
        assert_eq!(taken, Value::int(42));
    }

    #[test]
    fn share_inline() {
        let v = Value::int(42);
        let v2 = v.share();
        assert_eq!(v, v2);
    }

    #[test]
    fn share_arc_string() {
        let v = Value::string("hello");
        let v2 = v.share();
        assert_eq!(v, v2);
    }

    #[test]
    fn share_pure_iterator_ok() {
        use acvus_mir::ty::Effect;
        let v = Value::iterator(IterHandle::done(Effect::pure()));
        let v2 = v.share();
        // Pure iterator can be shared.
        drop(v2);
    }

    #[test]
    #[should_panic(expected = "move-only")]
    fn share_effectful_iterator_panics() {
        use acvus_mir::ty::Effect;
        let v = Value::iterator(IterHandle::done(Effect::io()));
        let _ = v.share();
    }

    #[test]
    fn structural_eq_basic() {
        assert!(Value::int(1).structural_eq(&Value::int(1)));
        assert!(!Value::int(1).structural_eq(&Value::int(2)));
        assert!(!Value::int(1).structural_eq(&Value::float(1.0)));
    }
}
