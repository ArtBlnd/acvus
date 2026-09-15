//! The interpreter's value: an erased word or an erased pointer. What a
//! value *is* lives in the MIR type the interpreter carries per value and in
//! the vtable its allocation carries, never in the value itself.

use std::any::TypeId;
use std::fmt;
use std::mem::{self, MaybeUninit};
use std::ptr::{self, NonNull};
use std::sync::{Arc, LazyLock};

use acvus_mir::ir::MirBody;
use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::error::RuntimeError;
use crate::interpreter::InterpreterContext;
use crate::journal::InMemoryContext;
use crate::vtable::{Composite, Header, Slot, VtableRegistry, Vtable};

// -- Value ------------------------------------------------------------

/// `Empty` is the moved-out sentinel and `Undef` the SSA initial value of a
/// loop-defined variable; neither is a value the program can read.
pub enum Value {
    Empty,
    Undef,
    Small(u64),
    Large(NonNull<Header>),
}

// SAFETY: every `Large` payload entered through `erase<T: Send + Sync>` or a
// composite constructor, all `Send + Sync`, and its vtable is a shared
// static.
unsafe impl Send for Value {}
unsafe impl Sync for Value {}

impl Drop for Value {
    fn drop(&mut self) {
        if let Value::Large(p) = *self {
            // SAFETY: the payload is live and is not used after this.
            unsafe { (p.as_ref().vtable.drop)(p) }
        }
    }
}

/// Whether `T` rides inline in `Small`: it fits the word and owns nothing.
pub const fn is_small<T>() -> bool {
    mem::size_of::<T>() <= 8 && mem::align_of::<T>() <= 8 && !mem::needs_drop::<T>()
}

fn large<T>(vtable: &'static Vtable, value: T) -> Value {
    let slot = Box::new(Slot {
        header: Header { vtable },
        value,
    });
    Value::Large(NonNull::from(Box::leak(slot)).cast::<Header>())
}

impl Value {
    /// # Safety
    /// The value may only be materialized back as this same `T`.
    pub unsafe fn erase<T>(table: &VtableRegistry, value: T) -> Value
    where
        T: Send + Sync + 'static,
    {
        if TypeId::of::<T>() == TypeId::of::<Value>() {
            // SAFETY: T is Value; the copy takes over and the original is forgotten.
            let same: Value = unsafe { mem::transmute_copy(&value) };
            mem::forget(value);
            return same;
        }
        if const { is_small::<T>() } {
            let mut bits = 0u64;
            // SAFETY: T fits in the word; low bytes are written and read alike.
            unsafe {
                ptr::copy_nonoverlapping(
                    &value as *const T as *const u8,
                    &mut bits as *mut u64 as *mut u8,
                    mem::size_of::<T>(),
                );
            }
            mem::forget(value);
            Value::Small(bits)
        } else {
            large(table.vtable_of::<T>(), value)
        }
    }

    /// # Safety
    /// `T` is the type this value was erased from.
    pub unsafe fn materialize<T>(self) -> T
    where
        T: Send + Sync + 'static,
    {
        if TypeId::of::<T>() == TypeId::of::<Value>() {
            // SAFETY: T is Value; the copy takes over and self is forgotten.
            let same: T = unsafe { mem::transmute_copy(&self) };
            mem::forget(self);
            return same;
        }
        let out = match &self {
            Value::Small(bits) => {
                debug_assert!(
                    is_small::<T>(),
                    "materialize: {} is not a small type",
                    std::any::type_name::<T>()
                );
                let mut out = MaybeUninit::<T>::uninit();
                // SAFETY: erase wrote T's bytes into the low bytes of the word.
                unsafe {
                    ptr::copy_nonoverlapping(
                        bits as *const u64 as *const u8,
                        out.as_mut_ptr() as *mut u8,
                        mem::size_of::<T>(),
                    );
                    out.assume_init()
                }
            }
            Value::Large(p) => {
                // SAFETY: the header is live.
                debug_assert_eq!(
                    unsafe { p.as_ref() }.vtable.type_id,
                    TypeId::of::<T>(),
                    "materialize: value is not a {}",
                    std::any::type_name::<T>()
                );
                // SAFETY: the payload was allocated by `large` as Box<Slot<T>>.
                let slot = unsafe { Box::from_raw(p.cast::<Slot<T>>().as_ptr()) };
                let Slot { value, .. } = *slot;
                value
            }
            Value::Empty => panic!("materialize: accessed moved-out value"),
            Value::Undef => panic!("materialize: accessed undef value"),
        };
        mem::forget(self);
        out
    }

    fn header(&self) -> &Header {
        match self {
            // SAFETY: the header is live for as long as the value.
            Value::Large(p) => unsafe { p.as_ref() },
            other => panic!("not a large value: {other:?}"),
        }
    }

    pub fn composite(&self) -> Option<Composite> {
        match self {
            Value::Large(_) => self.header().vtable.composite,
            _ => None,
        }
    }

    /// Read a `Large` payload in place.
    ///
    /// # Safety
    /// `T` is the type this value was erased from.
    pub unsafe fn peek<T: 'static>(&self) -> &T {
        debug_assert_eq!(self.header().vtable.type_id, TypeId::of::<T>());
        match self {
            // SAFETY: the payload is a live Slot<T>.
            Value::Large(p) => unsafe { &p.cast::<Slot<T>>().as_ref().value },
            other => panic!("peek: not a large value: {other:?}"),
        }
    }

    /// Mutate a `Large` payload in place.
    ///
    /// # Safety
    /// `T` is the type this value was erased from.
    pub unsafe fn peek_mut<T: 'static>(&mut self) -> &mut T {
        debug_assert_eq!(self.header().vtable.type_id, TypeId::of::<T>());
        match self {
            // SAFETY: the payload is a live Slot<T> and we hold &mut self.
            Value::Large(p) => unsafe { &mut p.cast::<Slot<T>>().as_mut().value },
            other => panic!("peek_mut: not a large value: {other:?}"),
        }
    }

    pub fn small(&self) -> u64 {
        match self {
            Value::Small(bits) => *bits,
            other => panic!("small: not a small value: {other:?}"),
        }
    }

    #[inline]
    pub fn take(&mut self) -> Value {
        mem::replace(self, Value::Empty)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(self, Value::Empty)
    }

    /// An explicit copy, through the vtable; the bridge until context
    /// dump/restore replaces sharing.
    pub fn deep_clone(&self) -> Value {
        match self {
            Value::Empty => panic!("clone: accessed moved-out value"),
            Value::Undef => Value::Undef,
            Value::Small(bits) => Value::Small(*bits),
            Value::Large(p) => {
                let vtable = self.header().vtable;
                let clone = vtable
                    .clone
                    .unwrap_or_else(|| panic!("clone: type {} is not clonable", vtable.name));
                // SAFETY: the payload is a live value of the witnessed type.
                Value::Large(unsafe { clone(*p) })
            }
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Empty => write!(f, "<empty>"),
            Value::Undef => write!(f, "<undef>"),
            Value::Small(bits) => write!(f, "Small({bits:#x})"),
            Value::Large(p) => {
                let vtable = self.header().vtable;
                match vtable.debug {
                    // SAFETY: the payload is a live value of the witnessed type.
                    Some(dbg) => unsafe { dbg(*p, f) },
                    None => write!(f, "<{}>", vtable.name),
                }
            }
        }
    }
}

/// Identity for `Large`, bits for `Small`. Equality of what a value holds is
/// a property of its type: materialize, then compare.
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Small(a), Value::Small(b)) => a == b,
            (Value::Large(pa), Value::Large(pb)) => pa == pb,
            (Value::Empty, Value::Empty) | (Value::Undef, Value::Undef) => true,
            _ => false,
        }
    }
}

// -- The interpreter's own composites --------------------------------

pub struct Array(pub Vec<Value>);
pub struct Tuple(pub Vec<Value>);
pub struct Object(pub FxHashMap<Astr, Value>);

pub struct VariantValue {
    pub tag: Astr,
    pub payload: Option<Box<Value>>,
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
    pub async fn call(&self, arg: Value) -> Result<Value, RuntimeError> {
        crate::interpreter::fn_value_call(self, vec![arg]).await
    }

    pub async fn call2(&self, arg1: Value, arg2: Value) -> Result<Value, RuntimeError> {
        crate::interpreter::fn_value_call(self, vec![arg1, arg2]).await
    }
}

/// A deferred computation handle (spawn result). Consumed exactly once by
/// eval. Move-only. Inner type is executor-specific.
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
    pub fn try_downcast<T: std::any::Any + Send + Sync>(self) -> Result<T, Self> {
        match self.inner.downcast::<T>() {
            Ok(val) => Ok(*val),
            Err(inner) => Err(Self { inner }),
        }
    }
    pub fn into_inner(self) -> Box<dyn std::any::Any + Send + Sync> {
        self.inner
    }
}

impl fmt::Debug for HandleValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Handle")
    }
}

// -- Composite vtables ------------------------------------------------

fn vtable<T: 'static>(
    name: &'static str,
    composite: Composite,
    clone: Option<crate::vtable::CloneFn>,
    debug: Option<crate::vtable::DebugFn>,
) -> Vtable {
    Vtable {
        composite: Some(composite),
        clone,
        debug,
        ..Vtable::drop_only::<T>(name)
    }
}

macro_rules! typed_vtable_fns {
    ($T:ty; $cl:ident = |$c:ident| $cl_body:expr;
             $dg:ident = |$d:ident, $f:ident| $dg_body:expr;) => {
        unsafe fn $cl(p: NonNull<Header>) -> NonNull<Header> {
            // SAFETY: p is the header of a live Slot<$T>.
            let slot = unsafe { p.cast::<Slot<$T>>().as_ref() };
            let $c = &slot.value;
            let copy = Box::new(Slot {
                header: Header {
                    vtable: slot.header.vtable,
                },
                value: $cl_body,
            });
            NonNull::from(Box::leak(copy)).cast::<Header>()
        }
        unsafe fn $dg(p: NonNull<Header>, $f: &mut fmt::Formatter<'_>) -> fmt::Result {
            // SAFETY: p is the header of a live Slot<$T>.
            let $d = unsafe { &p.cast::<Slot<$T>>().as_ref().value };
            $dg_body
        }
    };
}

fn seq_clone(a: &[Value]) -> Vec<Value> {
    a.iter().map(Value::deep_clone).collect()
}

typed_vtable_fns! { String;
    clone_string = |c| c.clone();
    dbg_string = |d, f| write!(f, "{d:?}");
}
typed_vtable_fns! { Array;
    clone_array = |c| Array(seq_clone(&c.0));
    dbg_array = |d, f| f.debug_list().entries(d.0.iter()).finish();
}
typed_vtable_fns! { Tuple;
    clone_tuple = |c| Tuple(seq_clone(&c.0));
    dbg_tuple = |d, f| {
        let mut dt = f.debug_tuple("");
        for v in d.0.iter() {
            dt.field(v);
        }
        dt.finish()
    };
}
typed_vtable_fns! { Object;
    clone_object = |c| Object(c.0.iter().map(|(k, v)| (*k, v.deep_clone())).collect());
    dbg_object = |d, f| f.debug_map().entries(d.0.iter()).finish();
}
typed_vtable_fns! { VariantValue;
    clone_variant = |c| VariantValue { tag: c.tag, payload: c.payload.as_ref().map(|p| Box::new(p.deep_clone())) };
    dbg_variant = |d, f| match &d.payload {
        Some(p) => write!(f, "{:?}({:?})", d.tag, p),
        None => write!(f, "{:?}", d.tag),
    };
}
typed_vtable_fns! { FnValue;
    clone_fn = |c| c.clone();
    dbg_fn = |d, f| write!(f, "Fn({} captures)", d.captures.len());
}

static STRING: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<String>("String", Composite::String, Some(clone_string), Some(dbg_string)));
static ARRAY: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<Array>("Array", Composite::Array, Some(clone_array), Some(dbg_array)));
static TUPLE: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<Tuple>("Tuple", Composite::Tuple, Some(clone_tuple), Some(dbg_tuple)));
static OBJECT: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<Object>("Object", Composite::Object, Some(clone_object), Some(dbg_object)));
static VARIANT: LazyLock<Vtable> = LazyLock::new(|| {
    vtable::<VariantValue>("Variant", Composite::Variant, Some(clone_variant), Some(dbg_variant))
});
static FN: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<FnValue>("Fn", Composite::Fn, Some(clone_fn), Some(dbg_fn)));
static HANDLE: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<HandleValue>("Handle", Composite::Handle, None, None));

// -- Constructors -----------------------------------------------------

impl Value {
    pub fn int(n: i64) -> Self {
        Value::Small(n as u64)
    }
    pub fn float(f: f64) -> Self {
        Value::Small(f.to_bits())
    }
    pub fn bool_(b: bool) -> Self {
        Value::Small(b as u64)
    }
    pub fn unit() -> Self {
        Value::Small(0)
    }
    pub fn byte(b: u8) -> Self {
        Value::Small(b as u64)
    }
    pub fn as_int(&self) -> i64 {
        self.small() as i64
    }
    pub fn as_float(&self) -> f64 {
        f64::from_bits(self.small())
    }
    pub fn as_bool(&self) -> bool {
        self.small() != 0
    }
    pub fn as_byte(&self) -> u8 {
        self.small() as u8
    }

    pub fn string(s: impl Into<String>) -> Self {
        large(&STRING, s.into())
    }
    pub fn array(items: Vec<Value>) -> Self {
        large(&ARRAY, Array(items))
    }
    pub fn tuple(items: Vec<Value>) -> Self {
        large(&TUPLE, Tuple(items))
    }
    pub fn object(fields: FxHashMap<Astr, Value>) -> Self {
        large(&OBJECT, Object(fields))
    }
    pub fn variant(tag: Astr, payload: Option<Value>) -> Self {
        large(
            &VARIANT,
            VariantValue {
                tag,
                payload: payload.map(Box::new),
            },
        )
    }
    pub fn closure(fv: FnValue) -> Self {
        large(&FN, fv)
    }
    pub fn handle(h: HandleValue) -> Self {
        large(&HANDLE, h)
    }

    pub fn is_object(&self) -> bool {
        self.composite() == Some(Composite::Object)
    }
    pub fn is_variant(&self) -> bool {
        self.composite() == Some(Composite::Variant)
    }
    pub fn is_string(&self) -> bool {
        self.composite() == Some(Composite::String)
    }

    /// # Safety
    /// The value is a `String`.
    pub unsafe fn as_str(&self) -> &str {
        unsafe { self.peek::<String>() }
    }
    /// # Safety
    /// The value is an `Array`.
    pub unsafe fn as_array(&self) -> &[Value] {
        unsafe { &self.peek::<Array>().0 }
    }
    /// # Safety
    /// The value is a `Tuple`.
    pub unsafe fn as_tuple(&self) -> &[Value] {
        unsafe { &self.peek::<Tuple>().0 }
    }
    /// # Safety
    /// The value is an `Object`.
    pub unsafe fn as_object(&self) -> &FxHashMap<Astr, Value> {
        unsafe { &self.peek::<Object>().0 }
    }
    /// # Safety
    /// The value is an `Object`.
    pub unsafe fn as_object_mut(&mut self) -> &mut FxHashMap<Astr, Value> {
        unsafe { &mut self.peek_mut::<Object>().0 }
    }
    /// # Safety
    /// The value is a `Variant`.
    pub unsafe fn as_variant(&self) -> &VariantValue {
        unsafe { self.peek::<VariantValue>() }
    }
    /// # Safety
    /// The value is an `Fn`.
    pub unsafe fn as_fn(&self) -> &FnValue {
        unsafe { self.peek::<FnValue>() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_size() {
        let size = mem::size_of::<Value>();
        assert!(size <= 16, "Value is {size} bytes, expected <= 16");
    }

    #[test]
    fn small_types_are_inline() {
        assert!(is_small::<i64>());
        assert!(is_small::<f64>());
        assert!(is_small::<bool>());
        assert!(is_small::<u8>());
        assert!(is_small::<()>());
        assert!(!is_small::<String>());
        assert!(!is_small::<Box<u8>>());
    }

    #[test]
    fn take_leaves_empty() {
        let mut v = Value::int(42);
        let taken = v.take();
        assert!(v.is_empty());
        assert_eq!(taken.as_int(), 42);
    }

    #[test]
    fn erase_materialize_round_trip() {
        let table = VtableRegistry::default();
        let v = unsafe { Value::erase(&table, 7i64) };
        assert_eq!(unsafe { v.materialize::<i64>() }, 7);
        let v = unsafe { Value::erase(&table, String::from("hi")) };
        assert_eq!(unsafe { v.materialize::<String>() }, "hi");
    }

    #[test]
    fn clone_through_the_witness_is_a_second_value() {
        let a = Value::array(vec![Value::int(1), Value::string("x")]);
        let b = a.deep_clone();
        assert_ne!(a, b);
        let (a, b) = unsafe { (a.as_array(), b.as_array()) };
        assert_eq!(a[0].as_int(), b[0].as_int());
        assert_eq!(unsafe { a[1].as_str() }, unsafe { b[1].as_str() });
        assert_ne!(a[1], b[1]);
    }

    #[test]
    fn dropping_a_vec_of_values_releases_them() {
        struct Counted(Arc<()>);
        let alive = Arc::new(());
        let table = VtableRegistry::default();
        let values: Vec<Value> = (0..3)
            .map(|_| unsafe { Value::erase(&table, Counted(Arc::clone(&alive))) })
            .collect();
        assert_eq!(Arc::strong_count(&alive), 4);
        drop(values);
        assert_eq!(Arc::strong_count(&alive), 1);
    }
}
