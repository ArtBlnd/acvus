use std::fmt;
use std::sync::Arc;

use acvus_extern::ExternValue;
use acvus_mir::ir::MirBody;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::error::RuntimeError;
use crate::interpreter::InterpreterContext;
use crate::journal::{InMemoryContext, RuntimeContext};

// -- Value ------------------------------------------------------------

/// Runtime value. Flat enum - no nested tiers.
///
/// # Layout (16 bytes on 64-bit)
///
/// - **Inline**: Int, Float, Bool, Unit, Byte - no heap allocation.
/// - **Shared**: String, Array, Object, Tuple, Variant - `Arc` wrapped,
///   clone = refcount bump. CoW via `Arc::make_mut` when mutation needed.
/// - **Owned**: Fn, Handle - `Box` wrapped, move-only.
///   SSA guarantees single use; `take()` replaces with `Empty`.
/// - **Extern**: extern boundary values.
///
/// `Empty` is the moved-out sentinel. Accessing an `Empty` register is a
/// programmer bug (SSA guarantees this cannot happen). Debug-asserted.
pub enum Value {
    // -- Inline (no allocation) -----------------------------------
    Empty,
    /// Undefined value - clone/move OK, read as concrete value = UB.
    /// Used as SSA initial value for variables defined inside loops.
    Undef,
    Int(i64),
    Float(f64),
    Bool(bool),
    Unit,
    Byte(u8),

    // -- Shared (Arc, clone = refcount bump) ----------------------
    String(Arc<String>),
    Array(Arc<Vec<Value>>),
    Object(Arc<FxHashMap<Astr, Value>>),
    Tuple(Arc<Vec<Value>>),
    Variant(Box<VariantValue>),

    // -- Owned (move-only, Box) -----------------------------------
    Fn(Box<FnValue>),
    Handle(Box<HandleValue>),

    // -- Extern (extern boundary) ---------------------------------
    Extern(Box<ExternValue>),
}

// -- Satellite types --------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct VariantValue {
    pub tag: Astr,
    pub payload: Option<Arc<Value>>,
}

/// A self-contained callable: execution context + body + captured values.
///
/// Created at `MakeClosure` time. It shares the run's live page: a
/// context read in its body sees the store that precedes the call, as
/// any call does (RFC-0014). Only captures are taken by value.
pub struct FnValue {
    pub shared: InterpreterContext,
    pub page: Arc<InMemoryContext>,
    pub body: Arc<MirBody>,
    pub captures: Arc<[Value]>,
}

impl Clone for FnValue {
    fn clone(&self) -> Self {
        Self {
            shared: self.shared.clone(),
            page: Arc::clone(&self.page),
            body: Arc::clone(&self.body),
            captures: Arc::clone(&self.captures),
        }
    }
}

impl fmt::Debug for FnValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn {:?}>", Arc::as_ptr(&self.body))
    }
}

impl FnValue {
    /// Call this closure with a single argument. Self-contained - no Interpreter needed.
    pub async fn call(&self, arg: Value) -> Result<Value, RuntimeError> {
        crate::interpreter::fn_value_call(self, vec![arg]).await
    }

    /// Call this closure with two arguments.
    pub async fn call2(&self, arg1: Value, arg2: Value) -> Result<Value, RuntimeError> {
        crate::interpreter::fn_value_call(self, vec![arg1, arg2]).await
    }
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

    /// Try to downcast, returning the original HandleValue on failure.
    pub fn try_downcast<T: std::any::Any + Send + Sync>(self) -> Result<T, Self> {
        match self.inner.downcast::<T>() {
            Ok(val) => Ok(*val),
            Err(inner) => Err(Self { inner }),
        }
    }

    /// Consume into the inner type-erased box (for executor dispatch).
    pub fn into_inner(self) -> Box<dyn std::any::Any + Send + Sync> {
        self.inner
    }
}

impl std::fmt::Debug for HandleValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Handle")
    }
}

// -- Constructors -----------------------------------------------------

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
    pub fn string(s: impl Into<String>) -> Self {
        Value::String(Arc::new(s.into()))
    }
    pub fn array(items: Vec<Value>) -> Self {
        Value::Array(Arc::new(items))
    }
    pub fn object(fields: FxHashMap<Astr, Value>) -> Self {
        Value::Object(Arc::new(fields))
    }
    pub fn tuple(elems: Vec<Value>) -> Self {
        Value::Tuple(Arc::new(elems))
    }
    pub fn variant(tag: Astr, payload: Option<Value>) -> Self {
        Value::Variant(Box::new(VariantValue {
            tag,
            payload: payload.map(Arc::new),
        }))
    }

    // Option (well-known variants)
    pub fn some(interner: &Interner, payload: Value) -> Self {
        Value::Variant(Box::new(VariantValue {
            tag: interner.intern("Some"),
            payload: Some(Arc::new(payload)),
        }))
    }
    pub fn none(interner: &Interner) -> Self {
        Value::Variant(Box::new(VariantValue {
            tag: interner.intern("None"),
            payload: None,
        }))
    }

    // Owned
    pub fn closure(fv: FnValue) -> Self {
        Value::Fn(Box::new(fv))
    }

    // Extern
    pub fn extern_value(ov: ExternValue) -> Self {
        Value::Extern(Box::new(ov))
    }
}

// -- Move / Clone -----------------------------------------------------

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
            Value::Undef => panic!("kind: accessed undef value"),
            Value::Int(_) => ValueKind::Int,
            Value::Float(_) => ValueKind::Float,
            Value::Bool(_) => ValueKind::Bool,
            Value::Unit => ValueKind::Unit,
            Value::Byte(_) => ValueKind::Byte,
            Value::String(_) => ValueKind::String,
            Value::Array(_) => ValueKind::Array,
            Value::Object(_) => ValueKind::Object,
            Value::Tuple(_) => ValueKind::Tuple,
            Value::Variant(_) => ValueKind::Variant,
            Value::Fn(_) => ValueKind::Fn,
            Value::Handle(_) => ValueKind::Handle,
            Value::Extern(_) => ValueKind::Extern,
        }
    }
}

// -- Extraction (borrow) ----------------------------------------------

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
    pub fn as_array(&self) -> &[Value] {
        match self {
            Value::Array(l) => l,
            other => panic!("expected Array, got {other:?}"),
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
}

// -- Extraction (owned - consumes the value) --------------------------

impl Value {
    #[inline]
    pub fn into_string(self) -> Arc<String> {
        match self {
            Value::String(s) => s,
            other => panic!("expected String, got {other:?}"),
        }
    }
    #[inline]
    pub fn into_array(self) -> Arc<Vec<Value>> {
        match self {
            Value::Array(l) => l,
            other => panic!("expected Array, got {other:?}"),
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
}

// -- Structural equality ----------------------------------------------

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

            (Value::Array(a), Value::Array(b)) => slice_eq(a, b),
            (Value::Tuple(a), Value::Tuple(b)) => slice_eq(a, b),
            (Value::Object(a), Value::Object(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .all(|(k, v)| b.get(k).is_some_and(|bv| v.structural_eq(bv)))
            }

            (Value::Variant(a), Value::Variant(b)) => {
                a.tag == b.tag
                    && match (&a.payload, &b.payload) {
                        (Some(pa), Some(pb)) => pa.structural_eq(pb),
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

// -- Clone ------------------------------------------------------------

impl Clone for Value {
    fn clone(&self) -> Self {
        match self {
            Value::Empty => panic!("clone: accessed moved-out value"),
            Value::Undef => Value::Undef,
            Value::Int(n) => Value::Int(*n),
            Value::Float(f) => Value::Float(*f),
            Value::Bool(b) => Value::Bool(*b),
            Value::Unit => Value::Unit,
            Value::Byte(b) => Value::Byte(*b),
            Value::String(s) => Value::String(Arc::clone(s)),
            Value::Array(l) => Value::Array(Arc::clone(l)),
            Value::Object(o) => Value::Object(Arc::clone(o)),
            Value::Tuple(t) => Value::Tuple(Arc::clone(t)),
            Value::Variant(v) => Value::Variant(Box::new(VariantValue {
                tag: v.tag,
                payload: v.payload.as_ref().map(Arc::clone),
            })),
            Value::Fn(f) => Value::Fn(f.clone()),
            Value::Handle(_) => panic!("clone: Handle is move-only"),
            Value::Extern(o) => Value::Extern(o.clone()),
        }
    }
}

// -- Debug / PartialEq ------------------------------------------------

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Empty => write!(f, "<empty>"),
            Value::Undef => write!(f, "<undef>"),
            Value::Int(n) => write!(f, "{n}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Unit => write!(f, "()"),
            Value::Byte(b) => write!(f, "0x{b:02x}"),
            Value::String(s) => write!(f, "{s:?}"),
            Value::Array(l) => f.debug_list().entries(l.iter()).finish(),
            Value::Object(o) => f.debug_map().entries(o.iter()).finish(),
            Value::Tuple(t) => {
                let mut d = f.debug_tuple("");
                for v in t.iter() {
                    d.field(v);
                }
                d.finish()
            }
            Value::Variant(v) => match &v.payload {
                Some(p) => write!(f, "{:?}({p:?})", v.tag),
                None => write!(f, "{:?}", v.tag),
            },
            Value::Fn(fv) => write!(f, "Fn({} captures)", fv.captures.len()),
            Value::Handle(_) => write!(f, "Handle"),
            Value::Extern(o) => write!(f, "{}", o.type_name),
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

// -- Size assertion ---------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_size() {
        let size = std::mem::size_of::<Value>();
        assert!(size <= 16, "Value enum is {size} bytes, expected <= 16");
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
    fn structural_eq_basic() {
        assert!(Value::int(1).structural_eq(&Value::int(1)));
        assert!(!Value::int(1).structural_eq(&Value::int(2)));
        assert!(!Value::int(1).structural_eq(&Value::float(1.0)));
    }
}
