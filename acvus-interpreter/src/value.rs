//! The interpreter's value and the composites it is built from. A
//! `Value`'s `Kind` says what Rust type it was erased from; the MIR type
//! the interpreter carries beside it says what the program reads it as.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::any::TypeId;
use std::fmt;
use std::mem::{self, MaybeUninit};
use std::ops::Deref;
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::Arc;

use acvus_extern::{FieldAt, ObjectShape, Owned, Release};
use acvus_mir::ty::IntTy;
use acvus_utils::{Astr, Interner};

use crate::code::{CodeRef, CodeShape, Runs};
use crate::regs::MAX_FRAME_SLOTS;
use crate::runtime::AcvusRuntime;
use crate::vtable::{Composite, DebugFn, HasVtable, Header, NameFn, Slot, Vtable, drop_slot};

// -- Kind -------------------------------------------------------------

macro_rules! kind {
    ($($name:ident: $t:ty),*) => {
        /// What a `Value`'s word is. The inline kinds are one variant per
        /// type in `acvus_extern::for_each_inline!`, which expands this macro.
        #[repr(u8)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum Kind {
            /// The SSA initial value of a loop-defined variable.
            Undef,
            Ref,
            Large,
            /// A run of registers and a heap realization hold one aggregate's
            /// flat layout alike (RFC-0050 rule 4), so a projection into either
            /// is this one kind and a reader of it does not know which it has.
            LargeRef,
            /// An option whose payload is a `None`: the word is how many
            /// `Some`s wrap it, and zero is `None` itself (RFC-0022).
            None,
            /// The word is the address of the instance's mono glue.
            Instance,
            InstanceAwait,
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
                    Kind::Undef
                    | Kind::Ref
                    | Kind::Large
                    | Kind::LargeRef
                    | Kind::None
                    | Kind::Instance
                    | Kind::InstanceAwait => None,
                }
            }

            pub fn name(self) -> Option<&'static str> {
                match self {
                    $(Kind::$name => Some(stringify!($t)),)*
                    Kind::Undef
                    | Kind::Ref
                    | Kind::Large
                    | Kind::LargeRef
                    | Kind::None
                    | Kind::Instance
                    | Kind::InstanceAwait => None,
                }
            }

            pub fn is_inline(self) -> bool {
                match self {
                    $(Kind::$name)|* => true,
                    Kind::Undef
                    | Kind::Ref
                    | Kind::Large
                    | Kind::LargeRef
                    | Kind::None
                    | Kind::Instance
                    | Kind::InstanceAwait => false,
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
#[derive(Clone, Copy)]
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
        Value::UNDEF
    }
}

// SAFETY: every `Large` payload entered through `erase<T: Send + Sync>` or a
// composite constructor, all `Send + Sync`, and its vtable is a shared
// static. A `Ref` is used only while its target is live (RFC-0018).
unsafe impl Send for Value {}
unsafe impl Sync for Value {}

impl Release for Value {
    fn release(self) {
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
    pub unsafe fn erase<T>(value: T) -> Value
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
            None => large(vtable_of::<T>(), value),
        }
    }

    /// # Safety
    /// `T` is the type this value was erased from.
    pub unsafe fn materialize<T>(self) -> T
    where
        T: Send + Sync + 'static,
    {
        if TypeId::of::<T>() == TypeId::of::<Value>() {
            // SAFETY: T is Value, one bit pattern under another name.
            return unsafe { mem::transmute_copy(&self) };
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

    #[inline]
    pub fn instance(at: acvus_extern::InstanceRun) -> Value {
        Value {
            kind: match at.task {
                acvus_extern::Task::Sync => Kind::Instance,
                _ => Kind::InstanceAwait,
            },
            word: at.at as u64,
        }
    }

    /// # Safety
    /// The value was made by `Value::instance`.
    #[inline]
    pub unsafe fn as_instance(&self) -> acvus_extern::InstanceRun {
        debug_assert!(
            matches!(self.kind, Kind::Instance | Kind::InstanceAwait),
            "as_instance: {self:?} is not an instance"
        );
        acvus_extern::InstanceRun {
            at: self.word as usize,
            task: match self.kind {
                Kind::Instance => acvus_extern::Task::Sync,
                _ => acvus_extern::Task::Async,
            },
        }
    }

    /// A projection onto the aggregate whose flat layout begins at `base`.
    ///
    /// Nothing reads through one yet: `prepare` knows every projected web's
    /// base, so it folds each use of a projection onto the register it names.
    /// The read and write through a projection are RFC-0050 rule 3's one
    /// family over a run and a heap `Large` alike, and they arrive with the
    /// flat heap object.
    #[inline]
    pub fn large_ref(base: *mut Value) -> Value {
        Value {
            kind: Kind::LargeRef,
            word: base as u64,
        }
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
            Kind::Instance => write!(f, "Instance({:#x})", self.word),
            Kind::InstanceAwait => write!(f, "InstanceAwait({:#x})", self.word),
            Kind::LargeRef => write!(f, "LargeRef({:p})", self.word as *const Value),
            Kind::Large => {
                let vtable = self.header().vtable;
                match vtable.debug {
                    // SAFETY: the payload is a live value of the witnessed type.
                    Some(dbg) => unsafe { dbg(self.payload(), f) },
                    None => write!(f, "<{}>", (vtable.name)()),
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

/// The language's array is the extern contract's `Arr` at
/// `T = Owned<AcvusRuntime>` (RFC-0022, RFC-0048 §7): it crosses without a
/// copy.
///
/// A crossing erases and materializes this composite by its `TypeId`, so
/// `acvus-extern`'s `Arr<Owned<Rt>, ()>` and this alias must name the same
/// Rust type. Spell one of them `Rt::Value` and both crates still compile
/// while every array that crosses the boundary misses this vtable.
pub type Array = acvus_extern::Arr<Owned<AcvusRuntime>, ()>;
pub struct Tuple(pub Vec<Owned<AcvusRuntime>>);
/// The language's object is the extern contract's `Obj` at
/// `V = Owned<AcvusRuntime>` (RFC-0032), under the same obligation as
/// `Array`.
pub type Object = acvus_extern::Obj<Owned<AcvusRuntime>>;

/// The language's variant is the extern contract's `Variant` at
/// `V = Owned<AcvusRuntime>`, under the same obligation as `Array`.
pub type VariantValue = acvus_extern::Variant<Owned<AcvusRuntime>>;

/// The head of a closure record, whose tail is the captures.
///
/// A closure is one box: `Slot<FnValue>` is the head and `[Value; len]`
/// follows it in the same allocation. A call reaches the record through the
/// `Value`'s word — a borrow of the box, never a copy of its contents — and
/// everything else a call needs comes through `ctx`.
///
/// Obligation across artifacts: `code` names a `Code` the run's `Prepared`
/// owns; `CodeRef::get` states the term.
#[repr(C)]
pub struct FnValue {
    at: *const (),
    shape: CodeShape,
    /// A `u16` because it counts capture registers, which `prepare` colours
    /// out of one frame: the width is what makes `closure_layout`'s size
    /// arithmetic total.
    len: u16,
}

const _: () = assert!(
    mem::size_of::<Slot<FnValue>>() == 24,
    "a closure record's captures begin at offset 24, after its header and head"
);
const _: () = assert!(
    mem::align_of::<Slot<FnValue>>() == mem::align_of::<Value>(),
    "a closure record's tail is laid at the head's own alignment"
);
const _: () = assert!(
    mem::size_of::<Slot<FnValue>>() + (u16::MAX as usize) * mem::size_of::<Value>()
        < isize::MAX as usize,
    "a closure record of the widest capture count the head can name is a valid layout"
);

/// The one layout a closure record is allocated, read and freed with.
#[inline]
fn closure_layout(len: u16) -> Layout {
    // SAFETY: the alignment is `Slot<FnValue>`'s, which `repr(C)` makes a
    // non-zero power of two; the const assert above is that the size of the
    // widest record `len` can name is a valid layout size.
    unsafe {
        Layout::from_size_align_unchecked(
            mem::size_of::<Slot<FnValue>>() + usize::from(len) * mem::size_of::<Value>(),
            mem::align_of::<Slot<FnValue>>(),
        )
    }
}

/// # Safety
/// `p` is the header of a live closure record, and it is not used again.
unsafe fn drop_closure(p: NonNull<Header>) {
    // SAFETY: the caller's contract.
    let head = &unsafe { p.cast::<Slot<FnValue>>().as_ref() }.value;
    let len = head.len;
    for capture in head.captures() {
        capture.release();
    }
    // SAFETY: the same layout `Value::closure` allocated this record with.
    unsafe { dealloc(p.as_ptr().cast::<u8>(), closure_layout(len)) }
}

impl FnValue {
    /// The prepared body this closure runs, and which of the two it is.
    #[inline(always)]
    pub fn runs(&self) -> Runs<'_> {
        // SAFETY: `CodeRef::runs`'s term — the run's `Prepared` is live for
        // as long as any value of the run, this one included.
        unsafe { CodeRef::runs(self.at, self.shape) }
    }

    /// The captures, in the record's own tail.
    #[inline(always)]
    pub fn captures(&self) -> &[Value] {
        // SAFETY: `Value::closure` is the only writer of a record, and it
        // wrote `self.len` captures immediately after this head.
        unsafe {
            slice::from_raw_parts(
                ptr::from_ref(self).add(1).cast::<Value>(),
                usize::from(self.len),
            )
        }
    }
}

impl fmt::Debug for FnValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn {:?}>", self.at)
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

const fn vtable<T: 'static>(name: NameFn, composite: Composite, debug: Option<DebugFn>) -> Vtable {
    Vtable {
        type_id: TypeId::of::<T>(),
        name,
        composite: Some(composite),
        drop: drop_slot::<T>,
        debug,
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
typed_debug_fn! { Object; dbg_object = |d, f| f.debug_map().entries(d.fields()).finish(); }
typed_debug_fn! { VariantValue;
    dbg_variant = |d, f| {
        let tag = Astr::of_bits(d.tag().bits());
        match d.payload().kind() {
            Kind::Undef => write!(f, "{tag:?}"),
            _ => write!(f, "{tag:?}({:?})", d.payload()),
        }
    };
}
typed_debug_fn! { FnValue; dbg_fn = |d, f| write!(f, "Fn({} captures)", d.captures().len()); }

static STRING: Vtable = vtable::<String>(|| "String", Composite::String, Some(dbg_string));
static ARRAY: Vtable = vtable::<Array>(|| "Array", Composite::Array, Some(dbg_array));
static TUPLE: Vtable = vtable::<Tuple>(|| "Tuple", Composite::Tuple, Some(dbg_tuple));
static OBJECT: Vtable = vtable::<Object>(|| "Object", Composite::Object, Some(dbg_object));
static VARIANT: Vtable =
    vtable::<VariantValue>(|| "Variant", Composite::Variant, Some(dbg_variant));
/// The one vtable whose `drop` is not `drop_slot`: a closure record is one
/// allocation with a tail, so it is freed by `drop_closure`.
static FN: Vtable = Vtable {
    type_id: TypeId::of::<FnValue>(),
    name: || "Fn",
    composite: Some(Composite::Fn),
    drop: drop_closure,
    debug: Some(dbg_fn),
};
static HANDLE: Vtable = vtable::<HandleValue>(|| "Handle", Composite::Handle, None);

/// The vtable `T` is erased through: a composite's own static, else the
/// drop-only constant every type has.
pub fn vtable_of<T>() -> &'static Vtable
where
    T: 'static,
{
    composite_vtable::<T>().unwrap_or(&<T as HasVtable>::VTABLE)
}

/// The `TypeId` chain folds at the monomorphization, as `Kind::of` does.
fn composite_vtable<T>() -> Option<&'static Vtable>
where
    T: 'static,
{
    let id = TypeId::of::<T>();
    for vtable in [&STRING, &ARRAY, &TUPLE, &OBJECT, &VARIANT, &FN, &HANDLE] {
        if id == vtable.type_id {
            return Some(vtable);
        }
    }
    None
}

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
    /// A `char`'s word is its scalar value, the `u32` `char as u32` gives
    /// (RFC-0058).
    pub fn char_(c: char) -> Self {
        Value::inline(Kind::Char, u64::from(u32::from(c)))
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
    /// The scalar value this word spells.
    ///
    /// # Panics
    /// When the word is not one, which a `Char` register's is by the
    /// checker: every way into one is `Value::char_` or a `u8 as char`.
    pub fn as_char(&self) -> u32 {
        let code = self.bits() as u32;
        assert!(
            char::from_u32(code).is_some(),
            "a char's word is a Unicode scalar value, found {code:#x}"
        );
        code
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
    pub fn array(items: Vec<Owned<AcvusRuntime>>) -> Self {
        large(&ARRAY, Array::new(items))
    }
    pub fn tuple(items: Vec<Owned<AcvusRuntime>>) -> Self {
        large(&TUPLE, Tuple(items))
    }
    /// A heap object: the shape its type fixes and one value per field of it,
    /// in that order (RFC-0050 rules 4 and 8).
    pub fn object(shape: Arc<ObjectShape>, values: Box<[Owned<AcvusRuntime>]>) -> Self {
        large(&OBJECT, acvus_extern::Obj::new(shape, values))
    }

    /// An object whose field at each position is read where `at` says, which is
    /// how `composite::MakeObject` fills one with no width to compare.
    pub fn object_filled<F>(shape: Arc<ObjectShape>, at: F) -> Self
    where
        F: FnMut(FieldAt) -> Owned<AcvusRuntime>,
    {
        large(&OBJECT, acvus_extern::Obj::filled(shape, at))
    }

    /// An object built from the names it writes rather than from a type: what a
    /// host has when it turns a JSON object into a value, whose language type is
    /// `Written` over exactly those names. Rule 8's order for such a type is
    /// those names sorted, so the order is computed here and nothing has to
    /// have been told it.
    pub fn object_by_name<I>(interner: &Interner, fields: I) -> Self
    where
        I: IntoIterator<Item = (Astr, Owned<AcvusRuntime>)>,
    {
        let mut fields: Vec<(Astr, Owned<AcvusRuntime>)> = fields.into_iter().collect();
        fields.sort_by(|(a, _), (b, _)| interner.resolve(*a).cmp(interner.resolve(*b)));
        let shape = ObjectShape::in_order(fields.iter().map(|(name, _)| *name).collect());
        Value::object(shape, fields.into_iter().map(|(_, v)| v).collect())
    }
    /// RFC-0050 rules 4 and 8's flat variant: the tag register and one payload,
    /// `Undef` where the tag carries none.
    pub fn variant(tag: Astr, payload: Option<Owned<AcvusRuntime>>) -> Self {
        let payload = payload.unwrap_or_else(|| Owned::from_value(Value::UNDEF));
        Value::variant_of(Value::tag(tag), payload)
    }

    pub fn variant_of(tag: Value, payload: Owned<AcvusRuntime>) -> Self {
        large(&VARIANT, VariantValue::of(Owned::from_value(tag), payload))
    }

    /// The word a tag register holds: the one number a run of the program gives
    /// the name, so two tags compare as words and neither side needs the enum's
    /// type to write or read one.
    pub fn tag(tag: Astr) -> Value {
        Value::inline(Kind::U64, tag.bits())
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

    /// A closure record, written once into one allocation. The record owns
    /// the captures it is given.
    pub fn closure<I>(code: CodeRef, captures: I) -> Self
    where
        I: IntoIterator<Item = Value>,
        I::IntoIter: ExactSizeIterator,
    {
        let captures = captures.into_iter();
        let len = captures.len();
        debug_assert!(
            len <= usize::from(MAX_FRAME_SLOTS),
            "a closure captures registers `prepare` coloured out of one frame"
        );
        let len = len as u16;
        let (at, shape) = code.parts();
        let layout = closure_layout(len);
        // SAFETY: the layout's size is the head's at least, so never zero.
        let Some(record) = NonNull::new(unsafe { alloc(layout) }.cast::<Slot<FnValue>>()) else {
            handle_alloc_error(layout)
        };
        // SAFETY: the allocation is fresh, named by nothing else, and sized
        // by `closure_layout(len)` — the head, then `len` capture slots.
        unsafe {
            record.write(Slot {
                header: Header { vtable: &FN },
                value: FnValue { at, shape, len },
            });
            let tail = record.add(1).cast::<Value>();
            for (at, capture) in captures.enumerate() {
                tail.add(at).write(capture);
            }
        }
        Value {
            kind: Kind::Large,
            word: record.as_ptr() as u64,
        }
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
    /// # Safety
    /// The value is a `String`.
    pub unsafe fn as_str(&self) -> &str {
        unsafe { self.peek::<String>() }
    }
    /// # Safety
    /// The value is an `Array`.
    pub unsafe fn as_array(&self) -> &[Owned<AcvusRuntime>] {
        unsafe { &self.peek::<Array>().0 }
    }
    /// # Safety
    /// The value is a `Tuple`.
    pub unsafe fn as_tuple(&self) -> &[Owned<AcvusRuntime>] {
        unsafe { &self.peek::<Tuple>().0 }
    }
    /// The object's fields, flat, in its shape's order: `as_object()[i]` is
    /// the field `as_shape().names()[i]` names (RFC-0050 rule 8).
    ///
    /// # Safety
    /// The value is an `Object`.
    pub unsafe fn as_object(&self) -> &[Owned<AcvusRuntime>] {
        unsafe { &self.peek::<Object>().values }
    }

    /// # Safety
    /// The value is an `Object`.
    pub unsafe fn as_shape(&self) -> &Arc<ObjectShape> {
        unsafe { &self.peek::<Object>().shape }
    }

    /// A field by name, through the object's own shape: the path RFC-0050 rule 6
    /// leaves a reader that holds a name and no type. No operation takes it —
    /// `prepare` resolved every field a body mentions to a position.
    ///
    /// # Safety
    /// The value is an `Object`.
    pub unsafe fn field_by_name(&self, name: Astr) -> Option<&Owned<AcvusRuntime>> {
        let held = unsafe { self.peek::<Object>() };
        held.shape.at(name).map(|at| &held.values[at.index()])
    }

    /// # Safety
    /// The value is an `Object`.
    pub unsafe fn field_by_name_mut(&mut self, name: Astr) -> Option<&mut Owned<AcvusRuntime>> {
        let held = unsafe { self.peek_mut::<Object>() };
        let at = held.shape.at(name)?;
        Some(&mut held.values[at.index()])
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
    pub unsafe fn as_object_mut(&mut self) -> &mut [Owned<AcvusRuntime>] {
        unsafe { &mut self.peek_mut::<Object>().values }
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
    /// The value is a variant's tag register.
    pub unsafe fn as_tag(&self) -> Astr {
        Astr::of_bits(self.bits())
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

    fn assert_composite(vtable: &Vtable, composite: Composite, name: &str) {
        assert_eq!(vtable.composite, Some(composite));
        assert_eq!((vtable.name)(), name);
    }

    #[test]
    fn a_composite_reaches_its_own_vtable_and_not_the_blanket_constant() {
        assert_composite(vtable_of::<String>(), Composite::String, "String");
        assert_composite(vtable_of::<Array>(), Composite::Array, "Array");
        assert_composite(vtable_of::<Tuple>(), Composite::Tuple, "Tuple");
        assert_composite(vtable_of::<Object>(), Composite::Object, "Object");
        assert_composite(vtable_of::<VariantValue>(), Composite::Variant, "Variant");
        assert_composite(vtable_of::<FnValue>(), Composite::Fn, "Fn");
        assert_composite(vtable_of::<HandleValue>(), Composite::Handle, "Handle");
    }

    #[test]
    fn a_type_the_language_does_not_name_reaches_the_drop_only_constant() {
        struct Extension;
        let vtable = vtable_of::<Extension>();
        assert_eq!(vtable.composite, None);
        assert_eq!(vtable.type_id, TypeId::of::<Extension>());
        assert!(*vtable == Vtable::drop_only::<Extension>());
    }

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
        let v = unsafe { Value::erase(Word(9)) };
        assert_eq!(v.kind(), Kind::Large);
        assert_eq!(unsafe { v.materialize::<Word>() }.0, 9);
    }

    #[test]
    fn erase_writes_the_kind_of_the_type() {
        assert_eq!(unsafe { Value::erase(7i64) }.kind(), Kind::I64);
        assert_eq!(unsafe { Value::erase(7u8) }.kind(), Kind::U8);
        assert_eq!(unsafe { Value::erase(1.5f64) }.kind(), Kind::F64);
        assert_eq!(unsafe { Value::erase(true) }.kind(), Kind::Bool);
        assert_eq!(unsafe { Value::erase(()) }.kind(), Kind::Unit);
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
        for kind in [
            Kind::Undef,
            Kind::Ref,
            Kind::Large,
            Kind::LargeRef,
            Kind::None,
        ] {
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
    }

    #[test]
    fn equal_bits_under_different_tags_are_not_equal() {
        assert_ne!(Value::int(1), Value::bool_(true));
        assert_eq!(Value::int(1), Value::int(1));
    }

    #[test]
    fn erased_string_is_the_composite_string() {
        let v = unsafe { Value::erase(String::from("hi")) };
        assert!(v.is_string());
        assert_eq!(unsafe { v.as_str() }, "hi");
        let r = Value::reference(&v);
        assert_eq!(unsafe { r.target().as_str() }, "hi");
    }

    #[test]
    fn erase_materialize_round_trip() {
        let v = unsafe { Value::erase(7i64) };
        assert_eq!(unsafe { v.materialize::<i64>() }, 7);
        let v = unsafe { Value::erase(String::from("hi")) };
        assert_eq!(unsafe { v.materialize::<String>() }, "hi");
    }

    struct Counted(Arc<()>);

    #[test]
    fn dropping_a_vec_of_owned_releases_them() {
        let alive = Arc::new(());
        let values: Vec<Owned<AcvusRuntime>> = (0..3)
            .map(|_| Owned::from_value(unsafe { Value::erase(Counted(Arc::clone(&alive))) }))
            .collect();
        assert_eq!(Arc::strong_count(&alive), 4);
        drop(values);
        assert_eq!(Arc::strong_count(&alive), 1);
    }

    #[test]
    fn a_value_leaving_scope_releases_nothing_and_release_does() {
        let alive = Arc::new(());
        let large = unsafe { Value::erase(Counted(Arc::clone(&alive))) };
        {
            let _copy = large;
        }
        assert_eq!(Arc::strong_count(&alive), 2);
        large.release();
        assert_eq!(Arc::strong_count(&alive), 1);
    }
}
