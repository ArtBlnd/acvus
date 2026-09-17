//! The interpreter's value: a tagged word, a reference, or an erased
//! pointer. The Rust type a value was erased from is recorded in it: the
//! `Tag` of a `Small`, the vtable of a `Large`. The MIR type the interpreter
//! carries per value says what the program reads it as.

use std::any::TypeId;
use std::fmt;
use std::mem::{self, MaybeUninit};
use std::ptr::{self, NonNull};
use std::sync::{Arc, LazyLock};

use acvus_mir::ty::IntTy;
use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::code::Code;
use crate::error::RuntimeError;
use crate::interpreter::InterpreterContext;
use crate::vtable::{Composite, Header, Slot, Vtable, VtableRegistry};

// -- Tag --------------------------------------------------------------

macro_rules! tag {
    ($($name:ident: $t:ty),*) => {
        /// The `Inline` type a `Small` word holds, one variant per type in
        /// `acvus_extern::for_each_inline!`.
        #[repr(u8)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum Tag {
            $($name,)*
        }

        impl Tag {
            pub fn of<T>() -> Option<Tag>
            where
                T: 'static,
            {
                let id = TypeId::of::<T>();
                $(if id == TypeId::of::<$t>() {
                    return Some(Tag::$name);
                })*
                None
            }

            pub fn type_id(self) -> TypeId {
                match self {
                    $(Tag::$name => TypeId::of::<$t>(),)*
                }
            }

            pub fn name(self) -> &'static str {
                match self {
                    $(Tag::$name => stringify!($t),)*
                }
            }
        }
    };
}
acvus_extern::for_each_inline!(tag);

impl Tag {
    pub fn int(k: IntTy) -> Tag {
        match k {
            IntTy::I8 => Tag::I8,
            IntTy::I16 => Tag::I16,
            IntTy::I32 => Tag::I32,
            IntTy::I64 => Tag::I64,
            IntTy::U8 => Tag::U8,
            IntTy::U16 => Tag::U16,
            IntTy::U32 => Tag::U32,
            IntTy::U64 => Tag::U64,
        }
    }
}

// -- Value ------------------------------------------------------------

/// `Empty` is the moved-out sentinel and `Undef` the SSA initial value of a
/// loop-defined variable; neither is a value the program can read.
#[derive(Default)]
pub enum Value {
    #[default]
    Empty,
    Undef,
    Small(Tag, u64),
    Ref(NonNull<Value>),
    Large(NonNull<Header>),
}

// SAFETY: every `Large` payload entered through `erase<T: Send + Sync>` or a
// composite constructor, all `Send + Sync`, and its vtable is a shared
// static. A `Ref` is used only while its target is live (RFC-0018).
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

pub fn is_small<T>() -> bool
where
    T: 'static,
{
    Tag::of::<T>().is_some()
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
        match Tag::of::<T>() {
            Some(tag) => {
                let mut bits = 0u64;
                // SAFETY: an `Inline` T fits the word; low bytes are written and read alike.
                unsafe {
                    ptr::copy_nonoverlapping(
                        &value as *const T as *const u8,
                        &mut bits as *mut u64 as *mut u8,
                        mem::size_of::<T>(),
                    );
                }
                mem::forget(value);
                Value::Small(tag, bits)
            }
            None => large(table.vtable_of::<T>(), value),
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
            Value::Small(tag, bits) => {
                debug_assert_eq!(
                    Some(*tag),
                    Tag::of::<T>(),
                    "materialize: value is not a {}",
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
            Value::Ref(_) => panic!("materialize: a reference was erased from no type"),
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
            Value::Small(_, bits) => *bits,
            other => panic!("small: not a small value: {other:?}"),
        }
    }

    pub fn tag(&self) -> Tag {
        match self {
            Value::Small(tag, _) => *tag,
            other => panic!("tag: not a small value: {other:?}"),
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

    /// The word copy a primitive gets (RFC-0018); a `Large` value moves out
    /// of its storage, which is left `Empty`.
    #[inline]
    pub fn use_from(slot: &mut Value) -> Value {
        match slot {
            Value::Small(..) | Value::Ref(_) => slot.copy_word(),
            Value::Undef => Value::Undef,
            Value::Empty => panic!("use: accessed moved-out value"),
            Value::Large(_) => slot.take(),
        }
    }

    /// The copy of a `Small` or a `Ref`, the two values that are their word.
    pub fn copy_word(&self) -> Value {
        match self {
            Value::Small(tag, bits) => Value::Small(*tag, *bits),
            Value::Ref(p) => Value::Ref(*p),
            other => panic!("copy_word: not a word value: {other:?}"),
        }
    }

    // -- References (RFC-0018) ------------------------------------

    pub fn reference(target: &Value) -> Value {
        Value::Ref(NonNull::from(target))
    }

    /// The storage a reference names.
    ///
    /// # Safety
    /// `self` is a reference made by `Value::reference` whose target is
    /// still live and unmoved.
    pub unsafe fn target<'a>(&self) -> &'a Value {
        match self {
            Value::Ref(p) => unsafe { p.as_ref() },
            other => panic!("target: not a reference: {other:?}"),
        }
    }

    /// # Safety
    /// As `target`, and no other name of the storage is used meanwhile.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn target_mut<'a>(&self) -> &'a mut Value {
        match self {
            Value::Ref(p) => unsafe { &mut *p.as_ptr() },
            other => panic!("target_mut: not a reference: {other:?}"),
        }
    }

    /// The bits of a `Small`, in place.
    pub fn small_ref(&self) -> &u64 {
        match self {
            Value::Small(_, bits) => bits,
            other => panic!("small_ref: not a small value: {other:?}"),
        }
    }

    /// The bits of a `Small`, in place.
    pub fn small_mut(&mut self) -> &mut u64 {
        match self {
            Value::Small(_, bits) => bits,
            other => panic!("small_mut: not a small value: {other:?}"),
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Empty => write!(f, "<empty>"),
            Value::Undef => write!(f, "<undef>"),
            Value::Small(tag, bits) => write!(f, "Small({tag:?}, {bits:#x})"),
            Value::Ref(p) => write!(f, "Ref({p:p})"),
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

/// Identity for `Large` and `Ref`, tag and bits for `Small`. Equality of
/// what a value holds is a property of its type: materialize, then compare.
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Small(ta, a), Value::Small(tb, b)) => ta == tb && a == b,
            (Value::Ref(pa), Value::Ref(pb)) => pa == pb,
            (Value::Large(pa), Value::Large(pb)) => pa == pb,
            (Value::Empty, Value::Empty) | (Value::Undef, Value::Undef) => true,
            _ => false,
        }
    }
}

// -- The interpreter's own composites --------------------------------

/// The language's array is the extern contract's `Arr` at `T = Value`
/// (RFC-0022): it crosses without a copy.
pub type Array = acvus_extern::Arr<Value, ()>;
pub struct Tuple(pub Vec<Value>);
/// The language's object is the extern contract's `Obj` at `V = Value`
/// (RFC-0032).
pub type Object = acvus_extern::Obj<Value>;

/// The language's variant is the extern contract's `Variant` at `V = Value`.
pub type VariantValue = acvus_extern::Variant<Value>;

/// The language's `Option<T>` is Rust's `Option` at `T = Value`, so it
/// crosses the extern boundary as itself (RFC-0022).
pub type OptionValue = Option<Value>;

/// The language's `Result<T, E>` is Rust's `Result` at `T = E = Value`, so
/// it crosses the extern boundary as itself (RFC-0038).
pub type ResultValue = Result<Value, Value>;

/// A self-contained callable: execution context, prepared body, captures.
///
/// Created at `MakeClosure` time. It shares the run's live page: a
/// context read in its body sees the store that precedes the call, as
/// any call does (RFC-0014). Only captures are taken by value.
pub struct FnValue {
    pub shared: Arc<InterpreterContext>,
    pub page: Arc<dyn crate::journal::RuntimeContext>,
    pub code: Arc<Code>,
    pub captures: Arc<[Value]>,
}

impl Clone for FnValue {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            page: Arc::clone(&self.page),
            code: Arc::clone(&self.code),
            captures: Arc::clone(&self.captures),
        }
    }
}

impl fmt::Debug for FnValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn {:?}>", Arc::as_ptr(&self.code))
    }
}

impl FnValue {
    pub async fn call(&self, arg: Value) -> Result<Value, RuntimeError> {
        crate::machine::fn_value_call(self, &mut [arg]).await
    }

    pub async fn call2(&self, arg1: Value, arg2: Value) -> Result<Value, RuntimeError> {
        crate::machine::fn_value_call(self, &mut [arg1, arg2]).await
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
    debug: Option<crate::vtable::DebugFn>,
) -> Vtable {
    Vtable {
        composite: Some(composite),
        debug,
        ..Vtable::drop_only::<T>(name)
    }
}

macro_rules! typed_debug_fn {
    ($T:ty; $dg:ident = |$d:ident, $f:ident| $dg_body:expr;) => {
        unsafe fn $dg(p: NonNull<Header>, $f: &mut fmt::Formatter<'_>) -> fmt::Result {
            // SAFETY: p is the header of a live Slot<$T>.
            let $d = unsafe { &p.cast::<Slot<$T>>().as_ref().value };
            $dg_body
        }
    };
}

typed_debug_fn! { String; dbg_string = |d, f| write!(f, "{d:?}"); }
typed_debug_fn! { Array; dbg_array = |d, f| f.debug_list().entries(d.0.iter()).finish(); }
typed_debug_fn! { Tuple;
    dbg_tuple = |d, f| {
        let mut dt = f.debug_tuple("");
        for v in d.0.iter() {
            dt.field(v);
        }
        dt.finish()
    };
}
typed_debug_fn! { Object; dbg_object = |d, f| f.debug_map().entries(d.0.iter()).finish(); }
typed_debug_fn! { VariantValue;
    dbg_variant = |d, f| match &d.payload {
        Some(p) => write!(f, "{:?}({:?})", d.tag, p),
        None => write!(f, "{:?}", d.tag),
    };
}
typed_debug_fn! { ResultValue;
    dbg_result = |d, f| match d {
        Ok(v) => write!(f, "Ok({v:?})"),
        Err(e) => write!(f, "Err({e:?})"),
    };
}
typed_debug_fn! { OptionValue;
    dbg_option = |d, f| match d {
        Some(p) => write!(f, "Some({p:?})"),
        None => write!(f, "None"),
    };
}
typed_debug_fn! { FnValue; dbg_fn = |d, f| write!(f, "Fn({} captures)", d.captures.len()); }

static STRING: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<String>("String", Composite::String, Some(dbg_string)));
static ARRAY: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<Array>("Array", Composite::Array, Some(dbg_array)));
static TUPLE: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<Tuple>("Tuple", Composite::Tuple, Some(dbg_tuple)));
static OBJECT: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<Object>("Object", Composite::Object, Some(dbg_object)));
static VARIANT: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<VariantValue>("Variant", Composite::Variant, Some(dbg_variant)));
static OPTION: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<OptionValue>("Option", Composite::Option, Some(dbg_option)));
static RESULT: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<ResultValue>("Result", Composite::Result, Some(dbg_result)));
static FN: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<FnValue>("Fn", Composite::Fn, Some(dbg_fn)));
static HANDLE: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<HandleValue>("Handle", Composite::Handle, None));

/// Every composite vtable, in the order of `Composite`.
pub(crate) static COMPOSITE_VTABLES: LazyLock<[&'static Vtable; 9]> = LazyLock::new(|| {
    [
        &STRING, &ARRAY, &TUPLE, &OBJECT, &VARIANT, &OPTION, &RESULT, &FN, &HANDLE,
    ]
});

// -- Constructors -----------------------------------------------------

impl Value {
    pub fn int(n: i64) -> Self {
        Value::Small(Tag::I64, n as u64)
    }
    /// An integer of width `k` from its two's-complement bits, sign- or
    /// zero-extended to the word as `k` says (RFC-0037).
    pub fn from_bits(k: IntTy, bits: u64) -> Self {
        Value::Small(Tag::int(k), bits)
    }
    pub fn float(f: f64) -> Self {
        Value::Small(Tag::F64, f.to_bits())
    }
    pub fn bool_(b: bool) -> Self {
        Value::Small(Tag::Bool, b as u64)
    }
    pub fn unit() -> Self {
        Value::Small(Tag::Unit, 0)
    }
    pub fn byte(b: u8) -> Self {
        Value::Small(Tag::U8, b as u64)
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
        large(&ARRAY, Array::new(items))
    }
    pub fn tuple(items: Vec<Value>) -> Self {
        large(&TUPLE, Tuple(items))
    }
    pub fn object(fields: FxHashMap<Astr, Value>) -> Self {
        large(&OBJECT, acvus_extern::Obj(fields))
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
    pub fn option(payload: OptionValue) -> Self {
        large(&OPTION, payload)
    }
    pub fn result(payload: ResultValue) -> Self {
        large(&RESULT, payload)
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
    pub fn is_array(&self) -> bool {
        self.composite() == Some(Composite::Array)
    }
    pub fn is_tuple(&self) -> bool {
        self.composite() == Some(Composite::Tuple)
    }
    pub fn is_variant(&self) -> bool {
        self.composite() == Some(Composite::Variant)
    }
    pub fn is_string(&self) -> bool {
        self.composite() == Some(Composite::String)
    }
    pub fn is_option(&self) -> bool {
        self.composite() == Some(Composite::Option)
    }
    pub fn is_result(&self) -> bool {
        self.composite() == Some(Composite::Result)
    }
    /// # Safety
    /// The value is a `Result`.
    pub unsafe fn as_result(&self) -> &ResultValue {
        unsafe { self.peek::<ResultValue>() }
    }
    /// # Safety
    /// The value is a `Result`.
    pub unsafe fn as_result_mut(&mut self) -> &mut ResultValue {
        unsafe { self.peek_mut::<ResultValue>() }
    }
    /// # Safety
    /// The value is an `Option`.
    pub unsafe fn as_option(&self) -> &OptionValue {
        unsafe { self.peek::<OptionValue>() }
    }
    /// # Safety
    /// The value is an `Option`.
    pub unsafe fn as_option_mut(&mut self) -> &mut OptionValue {
        unsafe { self.peek_mut::<OptionValue>() }
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
    /// The value is an `Array`.
    pub unsafe fn as_array_mut(&mut self) -> &mut Array {
        unsafe { self.peek_mut::<Array>() }
    }
    /// # Safety
    /// The value is a `Tuple`.
    pub unsafe fn as_tuple_mut(&mut self) -> &mut Tuple {
        unsafe { self.peek_mut::<Tuple>() }
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
    /// The value is a variant.
    pub unsafe fn as_variant_mut(&mut self) -> &mut VariantValue {
        unsafe { self.peek_mut::<VariantValue>() }
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
    fn a_copy_type_outside_the_inline_set_is_large() {
        #[derive(Clone, Copy)]
        struct Word(u32);
        assert!(!is_small::<Word>());
        let table = VtableRegistry::default();
        let v = unsafe { Value::erase(&table, Word(9)) };
        assert!(matches!(v, Value::Large(_)));
        assert_eq!(unsafe { v.materialize::<Word>() }.0, 9);
    }

    #[test]
    fn erase_writes_the_tag_of_the_type() {
        let table = VtableRegistry::default();
        assert_eq!(unsafe { Value::erase(&table, 7i64) }.tag(), Tag::I64);
        assert_eq!(unsafe { Value::erase(&table, 7u8) }.tag(), Tag::U8);
        assert_eq!(unsafe { Value::erase(&table, 1.5f64) }.tag(), Tag::F64);
        assert_eq!(unsafe { Value::erase(&table, true) }.tag(), Tag::Bool);
        assert_eq!(unsafe { Value::erase(&table, ()) }.tag(), Tag::Unit);
    }

    #[test]
    fn every_tag_names_its_type() {
        assert_eq!(Tag::I64.type_id(), TypeId::of::<i64>());
        assert_eq!(Tag::of::<i64>(), Some(Tag::I64));
        assert_eq!(Tag::I64.name(), "i64");
        assert_eq!(Tag::Unit.name(), "()");
        assert_eq!(Tag::int(IntTy::U16), Tag::U16);
        assert_eq!(Tag::of::<u16>(), Some(Tag::U16));
    }

    #[test]
    fn a_reference_is_its_own_variant() {
        let v = Value::int(3);
        let r = Value::reference(&v);
        assert!(matches!(r, Value::Ref(_)));
        assert_eq!(unsafe { r.target() }.as_int(), 3);
        assert_eq!(r.copy_word(), r);
    }

    #[test]
    fn equal_bits_under_different_tags_are_not_equal() {
        assert_ne!(Value::int(1), Value::bool_(true));
        assert_eq!(Value::int(1), Value::int(1));
    }

    #[test]
    fn erased_string_is_the_composite_string() {
        let table = VtableRegistry::default();
        let v = unsafe { Value::erase(&table, String::from("hi")) };
        assert!(v.is_string());
        assert_eq!(unsafe { v.as_str() }, "hi");
        let r = Value::reference(&v);
        assert_eq!(unsafe { r.target().as_str() }, "hi");
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
