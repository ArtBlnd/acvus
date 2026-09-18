//! The interpreter's value and the composites it is built from. A
//! `Value`'s `Kind` says what Rust type it was erased from; the MIR type
//! the interpreter carries beside it says what the program reads it as.

use std::any::TypeId;
use std::fmt;
use std::mem::{self, MaybeUninit};
use std::ops::Deref;
use std::ptr::{self, NonNull};
use std::sync::{Arc, LazyLock};

use acvus_mir::ty::IntTy;
use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::code::Code;
use crate::interpreter::InterpreterContext;
use crate::vtable::{Composite, Header, Slot, Vtable, VtableRegistry};

// -- Kind -------------------------------------------------------------

macro_rules! kind {
    ($($name:ident: $t:ty),*) => {
        /// What a `Value`'s word is. The inline kinds are one variant per
        /// type in `acvus_extern::for_each_inline!`, which expands this macro.
        #[repr(u8)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum Kind {
            /// The register a `Take` moved out of.
            Empty,
            /// The SSA initial value of a loop-defined variable.
            Undef,
            Ref,
            Large,
            /// An option whose payload is a `None`: the word is how many
            /// `Some`s wrap it, and zero is `None` itself (RFC-0022).
            None,
            $($name,)*
        }

        impl Kind {
            pub fn of<T>() -> Option<Kind>
            where
                T: 'static,
            {
                let id = TypeId::of::<T>();
                $(if id == TypeId::of::<$t>() {
                    return Some(Kind::$name);
                })*
                None
            }

            pub fn type_id(self) -> Option<TypeId> {
                match self {
                    $(Kind::$name => Some(TypeId::of::<$t>()),)*
                    Kind::Empty | Kind::Undef | Kind::Ref | Kind::Large | Kind::None => None,
                }
            }

            pub fn name(self) -> Option<&'static str> {
                match self {
                    $(Kind::$name => Some(stringify!($t)),)*
                    Kind::Empty | Kind::Undef | Kind::Ref | Kind::Large | Kind::None => None,
                }
            }

            pub fn is_inline(self) -> bool {
                match self {
                    $(Kind::$name)|* => true,
                    Kind::Empty | Kind::Undef | Kind::Ref | Kind::Large | Kind::None => false,
                }
            }
        }
    };
}
acvus_extern::for_each_inline!(kind);

impl Kind {
    pub fn int(k: IntTy) -> Kind {
        match k {
            IntTy::I8 => Kind::I8,
            IntTy::I16 => Kind::I16,
            IntTy::I32 => Kind::I32,
            IntTy::I64 => Kind::I64,
            IntTy::U8 => Kind::U8,
            IntTy::U16 => Kind::U16,
            IntTy::U32 => Kind::U32,
            IntTy::U64 => Kind::U64,
        }
    }
}

// -- Value ------------------------------------------------------------

#[repr(C)]
pub struct Value {
    kind: Kind,
    word: u64,
}

const _: () = assert!(
    mem::size_of::<Value>() == 16,
    "a Value is one kind byte and one word"
);
const _: () = assert!(
    mem::offset_of!(Value, word) == Value::WORD_OFFSET,
    "a chain's pre-multiplied leaf offset reaches the word of a Value"
);
const _: () = assert!(
    mem::size_of::<Option<Value>>() == 16,
    "Option<Value> takes its discriminant from a spare Kind"
);
const _: () = assert!(
    mem::size_of::<usize>() <= 8,
    "a Ref and a Large address fit the value word"
);

impl Default for Value {
    fn default() -> Value {
        Value::EMPTY
    }
}

// SAFETY: every `Large` payload entered through `erase<T: Send + Sync>` or a
// composite constructor, all `Send + Sync`, and its vtable is a shared
// static. A `Ref` is used only while its target is live (RFC-0018).
unsafe impl Send for Value {}
unsafe impl Sync for Value {}

impl Drop for Value {
    fn drop(&mut self) {
        if self.kind == Kind::Large {
            let p = self.payload();
            // SAFETY: the payload is live and is not used after this.
            unsafe { (p.as_ref().vtable.drop)(p) }
        }
    }
}

pub fn is_inline<T>() -> bool
where
    T: 'static,
{
    Kind::of::<T>().is_some()
}

fn large<T>(vtable: &'static Vtable, value: T) -> Value {
    let slot = Box::new(Slot {
        header: Header { vtable },
        value,
    });
    Value {
        kind: Kind::Large,
        word: Box::into_raw(slot) as *mut Header as u64,
    }
}

impl Value {
    pub const WORD_OFFSET: usize = 8;

    pub const EMPTY: Value = Value {
        kind: Kind::Empty,
        word: 0,
    };
    pub const UNDEF: Value = Value {
        kind: Kind::Undef,
        word: 0,
    };
    pub const NONE: Value = Value {
        kind: Kind::None,
        word: 0,
    };

    #[inline]
    pub fn inline(kind: Kind, bits: u64) -> Value {
        debug_assert!(kind.is_inline(), "inline: {kind:?} does not carry bits");
        Value { kind, word: bits }
    }

    #[inline]
    pub fn kind(&self) -> Kind {
        self.kind
    }

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
        match Kind::of::<T>() {
            Some(kind) => {
                let mut word = 0u64;
                // SAFETY: an `Inline` T fits the word; low bytes are written and read alike.
                unsafe {
                    ptr::copy_nonoverlapping(
                        &value as *const T as *const u8,
                        &mut word as *mut u64 as *mut u8,
                        mem::size_of::<T>(),
                    );
                }
                mem::forget(value);
                Value { kind, word }
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
        match Kind::of::<T>() {
            Some(kind) => {
                debug_assert_eq!(
                    self.kind,
                    kind,
                    "materialize: value is not a {}",
                    std::any::type_name::<T>()
                );
                let word = self.word;
                mem::forget(self);
                let mut out = MaybeUninit::<T>::uninit();
                // SAFETY: erase wrote T's bytes into the low bytes of the word.
                unsafe {
                    ptr::copy_nonoverlapping(
                        &word as *const u64 as *const u8,
                        out.as_mut_ptr() as *mut u8,
                        mem::size_of::<T>(),
                    );
                    out.assume_init()
                }
            }
            None => {
                debug_assert_eq!(
                    self.header().vtable.type_id,
                    TypeId::of::<T>(),
                    "materialize: value is not a {}",
                    std::any::type_name::<T>()
                );
                let p = self.payload();
                mem::forget(self);
                // SAFETY: the payload was allocated by `large` as Box<Slot<T>>.
                let slot = unsafe { Box::from_raw(p.cast::<Slot<T>>().as_ptr()) };
                let Slot { value, .. } = *slot;
                value
            }
        }
    }

    /// The payload a `Large` names.
    ///
    /// The kind check is a `debug_assert!`: every caller reaches here having
    /// already read `Kind::Large`, or under a `# Safety` contract that the
    /// value was erased from a type with no inline kind.
    #[inline]
    fn payload(&self) -> NonNull<Header> {
        debug_assert_eq!(self.kind, Kind::Large, "payload: {self:?} is not large");
        // SAFETY: a `Large` word is the pointer `large` leaked, never null.
        unsafe { NonNull::new_unchecked(self.word as *mut Header) }
    }

    fn header(&self) -> &Header {
        // SAFETY: the header is live for as long as the value.
        unsafe { self.payload().as_ref() }
    }

    /// The vtable a `Large` payload was erased through.
    pub fn vtable(&self) -> &'static Vtable {
        self.header().vtable
    }

    pub fn composite(&self) -> Option<Composite> {
        if self.kind == Kind::Large {
            self.header().vtable.composite
        } else {
            None
        }
    }

    /// Read a `Large` payload in place.
    ///
    /// # Safety
    /// `T` is the type this value was erased from.
    pub unsafe fn peek<T: 'static>(&self) -> &T {
        debug_assert_eq!(self.header().vtable.type_id, TypeId::of::<T>());
        // SAFETY: the payload is a live Slot<T>.
        unsafe { &self.payload().cast::<Slot<T>>().as_ref().value }
    }

    /// Mutate a `Large` payload in place.
    ///
    /// # Safety
    /// `T` is the type this value was erased from.
    pub unsafe fn peek_mut<T: 'static>(&mut self) -> &mut T {
        debug_assert_eq!(self.header().vtable.type_id, TypeId::of::<T>());
        // SAFETY: the payload is a live Slot<T> and we hold &mut self.
        unsafe { &mut self.payload().cast::<Slot<T>>().as_mut().value }
    }

    /// The bits an `Inline` type was erased into.
    ///
    /// The kind check is a `debug_assert!`: the MIR type checker gives every
    /// slot these read a primitive type, and only an inline kind is erased
    /// from one.
    #[inline]
    pub fn bits(&self) -> u64 {
        debug_assert!(self.kind.is_inline(), "bits: {self:?} carries no bits");
        self.word
    }

    /// As `bits`, in place.
    #[inline]
    pub fn bits_ref(&self) -> &u64 {
        debug_assert!(self.kind.is_inline(), "bits_ref: {self:?} carries no bits");
        &self.word
    }

    /// As `bits`, in place and exclusively.
    #[inline]
    pub fn bits_mut(&mut self) -> &mut u64 {
        debug_assert!(self.kind.is_inline(), "bits_mut: {self:?} carries no bits");
        &mut self.word
    }

    #[inline]
    pub fn take(&mut self) -> Value {
        mem::replace(self, Value::EMPTY)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.kind == Kind::Empty
    }

    /// The word copy a primitive gets (RFC-0018); a `Large` value moves out
    /// of its storage, which is left `Empty`.
    #[inline]
    pub fn use_from(slot: &mut Value) -> Value {
        match slot.kind {
            Kind::Large => slot.take(),
            Kind::Empty => panic!("use: accessed moved-out value"),
            Kind::Undef => Value::UNDEF,
            Kind::Ref | Kind::None => slot.copy_word(),
            inline => {
                debug_assert!(inline.is_inline(), "use: {inline:?} is not a word");
                slot.copy_word()
            }
        }
    }

    /// The copy an inline value or a reference gets: the two kinds that are
    /// their word and own nothing.
    ///
    /// The kind check is a `debug_assert!`: `use_from` is the only caller
    /// that does not already hold the kind, and it reaches here only from
    /// `Kind::Ref` and the inline kinds.
    #[inline]
    pub fn copy_word(&self) -> Value {
        debug_assert!(
            self.kind.is_inline() || self.kind == Kind::Ref || self.kind == Kind::None,
            "copy_word: {self:?} does not copy"
        );
        Value {
            kind: self.kind,
            word: self.word,
        }
    }

    // -- References (RFC-0018) ------------------------------------

    #[inline]
    pub fn reference(target: &Value) -> Value {
        Value {
            kind: Kind::Ref,
            word: ptr::from_ref(target) as u64,
        }
    }

    /// The storage a reference names.
    ///
    /// The kind check is a `debug_assert!`: the MIR type checker gives a
    /// slot read through `RefTarget::Through`, and an extern parameter
    /// `&T` / `&mut T`, a reference type, and `Value::reference` is the only
    /// value of that type.
    ///
    /// # Safety
    /// `self` is a reference made by `Value::reference` whose target is
    /// still live and unmoved.
    #[inline]
    pub unsafe fn target<'a>(&self) -> &'a Value {
        debug_assert_eq!(self.kind, Kind::Ref, "target: {self:?} is not a reference");
        // SAFETY: the caller's contract: the target is live and unmoved.
        unsafe { &*(self.word as *const Value) }
    }

    /// # Safety
    /// As `target`, and no other name of the storage is used meanwhile.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub unsafe fn target_mut<'a>(&self) -> &'a mut Value {
        debug_assert_eq!(
            self.kind,
            Kind::Ref,
            "target_mut: {self:?} is not a reference"
        );
        // SAFETY: the caller's contract: the target is live and named once.
        unsafe { &mut *(self.word as *mut Value) }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            Kind::Empty => write!(f, "<empty>"),
            Kind::Undef => write!(f, "<undef>"),
            Kind::None => {
                for _ in 0..self.word {
                    f.write_str("Some(")?;
                }
                f.write_str("None")?;
                for _ in 0..self.word {
                    f.write_str(")")?;
                }
                Ok(())
            }
            Kind::Ref => write!(f, "Ref({:p})", self.word as *const Value),
            Kind::Large => {
                let vtable = self.header().vtable;
                match vtable.debug {
                    // SAFETY: the payload is a live value of the witnessed type.
                    Some(dbg) => unsafe { dbg(self.payload(), f) },
                    None => write!(f, "<{}>", vtable.name),
                }
            }
            kind => write!(f, "{kind:?}({:#x})", self.word),
        }
    }
}

/// Identity for `Large` and `Ref`, kind and bits for an inline value.
/// Equality of what a value holds is a property of its type: materialize,
/// then compare.
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.word == other.word
    }
}

/// Where a read landed: a place inside a value, or the `None` that a
/// `Some`'s payload is when it has no place of its own (RFC-0022).
pub enum Place<'a> {
    At(&'a Value),
    Depth(Value),
}

impl Deref for Place<'_> {
    type Target = Value;

    fn deref(&self) -> &Value {
        match self {
            Place::At(v) => v,
            Place::Depth(v) => v,
        }
    }
}

/// As `Place`, for a write.
pub enum PlaceMut<'a> {
    At(&'a mut Value),
    Depth(Value),
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
    pub async fn call(&self, arg: Value) -> Value {
        crate::machine::fn_value_call(self, &mut [arg]).await
    }

    pub async fn call2(&self, arg1: Value, arg2: Value) -> Value {
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
static RESULT: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<ResultValue>("Result", Composite::Result, Some(dbg_result)));
static FN: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<FnValue>("Fn", Composite::Fn, Some(dbg_fn)));
static HANDLE: LazyLock<Vtable> =
    LazyLock::new(|| vtable::<HandleValue>("Handle", Composite::Handle, None));

/// Every composite vtable, in the order of `Composite`.
pub(crate) static COMPOSITE_VTABLES: LazyLock<[&'static Vtable; 8]> = LazyLock::new(|| {
    [
        &STRING, &ARRAY, &TUPLE, &OBJECT, &VARIANT, &RESULT, &FN, &HANDLE,
    ]
});

// -- Constructors -----------------------------------------------------

impl Value {
    pub fn int(n: i64) -> Self {
        Value::inline(Kind::I64, n as u64)
    }
    /// An integer of width `k` from its two's-complement bits, sign- or
    /// zero-extended to the word as `k` says (RFC-0037).
    pub fn from_bits(k: IntTy, bits: u64) -> Self {
        Value::inline(Kind::int(k), bits)
    }
    pub fn float(f: f64) -> Self {
        Value::inline(Kind::F64, f.to_bits())
    }
    pub fn bool_(b: bool) -> Self {
        Value::inline(Kind::Bool, b as u64)
    }
    pub fn unit() -> Self {
        Value::inline(Kind::Unit, 0)
    }
    pub fn byte(b: u8) -> Self {
        Value::inline(Kind::U8, b as u64)
    }
    pub fn as_int(&self) -> i64 {
        self.bits() as i64
    }
    pub fn as_float(&self) -> f64 {
        f64::from_bits(self.bits())
    }
    pub fn as_bool(&self) -> bool {
        self.bits() != 0
    }
    pub fn as_byte(&self) -> u8 {
        self.bits() as u8
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
    /// `Some(payload)`, in the one shape every option takes: the payload's
    /// own value, unless the payload is itself a `None`, whose depth word
    /// this `Some` raises by one (RFC-0022).
    pub fn some(payload: Value) -> Self {
        if payload.kind != Kind::None {
            return payload;
        }
        Value {
            kind: Kind::None,
            word: payload.word + 1,
        }
    }

    /// The `None` check is a `debug_assert!`: every caller reaches here
    /// having already read `Some` — the extern boundary from `is_none`,
    /// the machine from the `TestVariant` that chose the branch.
    #[inline]
    pub fn some_payload(option: Value) -> Self {
        debug_assert!(!option.is_none(), "some_payload: the value is None");
        if option.kind != Kind::None {
            return option;
        }
        Value {
            kind: Kind::None,
            word: option.word - 1,
        }
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
    pub fn is_none(&self) -> bool {
        self.kind == Kind::None && self.word == 0
    }

    /// The payload a `Some` carries, read where it lies. A `Some` of a
    /// `None` owns nothing, so the payload it yields is a word the reader
    /// holds rather than a borrow of this value.
    pub fn option_payload(&self) -> Option<Place<'_>> {
        let Kind::None = self.kind else {
            return Some(Place::At(self));
        };
        self.word.checked_sub(1).map(|depth| {
            Place::Depth(Value {
                kind: Kind::None,
                word: depth,
            })
        })
    }

    /// The payload a `Some` owns, exclusively. `None` both for the option
    /// `None` and for a `Some` of one, the two values with nothing under
    /// them.
    pub fn option_payload_mut(&mut self) -> Option<&mut Value> {
        (self.kind != Kind::None).then_some(self)
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
    fn a_value_is_a_kind_byte_and_a_word() {
        assert_eq!(mem::size_of::<Value>(), 16);
        assert_eq!(mem::size_of::<Option<Value>>(), 16);
        assert_eq!(mem::align_of::<Value>(), 8);
    }

    #[test]
    fn small_types_are_inline() {
        assert!(is_inline::<i64>());
        assert!(is_inline::<f64>());
        assert!(is_inline::<bool>());
        assert!(is_inline::<u8>());
        assert!(is_inline::<()>());
        assert!(!is_inline::<String>());
        assert!(!is_inline::<Box<u8>>());
    }

    #[test]
    fn a_copy_type_outside_the_inline_set_is_large() {
        #[derive(Clone, Copy)]
        struct Word(u32);
        assert!(!is_inline::<Word>());
        let table = VtableRegistry::default();
        let v = unsafe { Value::erase(&table, Word(9)) };
        assert_eq!(v.kind(), Kind::Large);
        assert_eq!(unsafe { v.materialize::<Word>() }.0, 9);
    }

    #[test]
    fn erase_writes_the_kind_of_the_type() {
        let table = VtableRegistry::default();
        assert_eq!(unsafe { Value::erase(&table, 7i64) }.kind(), Kind::I64);
        assert_eq!(unsafe { Value::erase(&table, 7u8) }.kind(), Kind::U8);
        assert_eq!(unsafe { Value::erase(&table, 1.5f64) }.kind(), Kind::F64);
        assert_eq!(unsafe { Value::erase(&table, true) }.kind(), Kind::Bool);
        assert_eq!(unsafe { Value::erase(&table, ()) }.kind(), Kind::Unit);
    }

    #[test]
    fn every_inline_kind_names_its_type() {
        assert_eq!(Kind::I64.type_id(), Some(TypeId::of::<i64>()));
        assert_eq!(Kind::of::<i64>(), Some(Kind::I64));
        assert_eq!(Kind::I64.name(), Some("i64"));
        assert_eq!(Kind::Unit.name(), Some("()"));
        assert_eq!(Kind::int(IntTy::U16), Kind::U16);
        assert_eq!(Kind::of::<u16>(), Some(Kind::U16));
    }

    #[test]
    fn the_kinds_that_no_rust_type_was_erased_into_name_none() {
        for kind in [Kind::Empty, Kind::Undef, Kind::Ref, Kind::Large, Kind::None] {
            assert_eq!(kind.type_id(), None, "{kind:?}");
            assert_eq!(kind.name(), None, "{kind:?}");
            assert!(!kind.is_inline(), "{kind:?}");
        }
    }

    #[test]
    fn a_reference_is_its_own_kind() {
        let v = Value::int(3);
        let r = Value::reference(&v);
        assert_eq!(r.kind(), Kind::Ref);
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
