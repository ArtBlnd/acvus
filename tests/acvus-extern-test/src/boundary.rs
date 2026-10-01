//! The only module that writes `unsafe`: the facts `WordRuntime` signs, the
//! casts between a word and what it holds, and the structural facts of the
//! test types.
//!
//! A value is a kind and a 64-bit word. A scalar lies in the word's first
//! bytes. Anything larger lies in an allocation the word points at: a Rust
//! value or a Rust function's glue in a `Slot` headed by its release and its
//! canonical type. A reference is a pointer in the word: `Ref` to a `Word` the
//! run keeps, `Part` to a part of lent storage, and `WidePart` to an
//! allocation holding the wide pointer of an unsized part. A reference value
//! owns nothing (S2): the allocation a `WidePart` points at is the runtime's,
//! kept until the runtime drops.
//!
//! An opaque value, text, an option, a result and a tuple each lie in a slot:
//! the Rust value itself, and a part of an option, a result or a tuple as a
//! `Held` value the slot owns. An array is a slot holding a `Vec` of `Held`
//! values.
//!
//! Every read at a type takes the entry's `Settled`, the proof that the value
//! is live at the type the checker settled for its site (safety §3). No read
//! checks the kind against the type.

#![allow(unsafe_code)]
#![allow(
    clippy::disallowed_methods,
    reason = "the boundary module: a pointer made from the address a word holds is written here"
)]

use std::any::TypeId;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ptr;
use std::sync::{Arc, Mutex, PoisonError};
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::done::Done;
use acvus_extern::runtime::{
    Crosses, Opaque, OpaqueKind, OptionKind, PrimitiveKind, RefKind, RefTarget, Referent, ResultKind, Runtime,
    StringKind, TupleKind,
};
use acvus_extern::seal::Crossing;
use acvus_extern::{
    LayoutFact, Primitive, Release, Releasing, Settled, SiteShape, SignedS1, SignedS2, SignedS3, Structural, StructuralFacts,
};

use crate::runtime::WordRuntime;

const _: () = assert!(
    size_of::<usize>() <= size_of::<u64>(),
    "an address fits the value word"
);

/// What a value's word is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F64,
    Bool,
    Char,
    Unit,
    Large,
    Closure,
    Ref,
    Part,
    WidePart,
}

impl Kind {
    pub(crate) fn is_scalar(self) -> bool {
        scalars().iter().any(|&(_, kind)| kind == self)
    }
}

fn scalars() -> [(TypeId, Kind); 12] {
    [
        (TypeId::of::<i8>(), Kind::I8),
        (TypeId::of::<i16>(), Kind::I16),
        (TypeId::of::<i32>(), Kind::I32),
        (TypeId::of::<i64>(), Kind::I64),
        (TypeId::of::<u8>(), Kind::U8),
        (TypeId::of::<u16>(), Kind::U16),
        (TypeId::of::<u32>(), Kind::U32),
        (TypeId::of::<u64>(), Kind::U64),
        (TypeId::of::<f64>(), Kind::F64),
        (TypeId::of::<bool>(), Kind::Bool),
        (TypeId::of::<char>(), Kind::Char),
        (TypeId::of::<()>(), Kind::Unit),
    ]
}

/// The scalar kind of the primitive `P`.
fn scalar_kind<P>() -> Kind
where
    P: Primitive,
{
    let canon = TypeId::of::<P>();
    let found = scalars().into_iter().find(|&(id, _)| id == canon);
    found.map(|(_, kind)| kind).expect("every primitive is a scalar")
}

/// A value: its kind and its word, as the interpreter's `Value` on master.
#[derive(Debug, Clone, Copy)]
pub struct Word {
    kind: Kind,
    word: u64,
}

impl Word {
    /// The `i64` this value is.
    pub fn as_i64(self) -> i64 {
        assert_eq!(self.kind, Kind::I64, "the value is an i64");
        // SAFETY: an `I64` value holds an `i64` in its word's first bytes.
        unsafe { ptr::from_ref(&self.word).cast::<i64>().read() }
    }

    #[cfg(test)]
    pub(crate) fn kind(self) -> Kind {
        self.kind
    }

    fn pointing<P>(kind: Kind, at: *mut P) -> Word {
        Word { kind, word: at.expose_provenance() as u64 }
    }

    fn address<P>(self) -> *mut P {
        ptr::with_exposed_provenance_mut(self.word as usize)
    }
}

/// A Rust value in an allocation of its own, headed by what releases it and
/// what its canonical type is.
#[repr(C)]
struct Slot<T> {
    header: Header,
    value: T,
}

struct Header {
    release: unsafe fn(*mut Header),
    canon: TypeId,
}

impl<T> Slot<T> {
    fn allocate(kind: Kind, value: T, canon: TypeId) -> Word {
        let header = Header { release: Slot::<T>::release, canon };
        Word::pointing(kind, Box::into_raw(Box::new(Slot { header, value })))
    }

    /// # Safety
    /// `header` heads a live `Slot<T>` that `allocate` made, and nothing
    /// uses it afterwards.
    unsafe fn release(header: *mut Header) {
        // SAFETY: the caller's contract; `Slot` is `repr(C)` with its header
        // first, so the header's address is the slot's.
        drop(unsafe { Box::from_raw(header.cast::<Slot<T>>()) });
    }

    /// # Safety
    /// As `release`, and `T` is the type the slot was allocated at.
    unsafe fn take(header: *mut Header) -> T {
        // SAFETY: the caller's contract.
        unsafe { Box::from_raw(header.cast::<Slot<T>>()) }.value
    }
}

/// A primitive's value: the scalar in the word's first bytes.
pub(crate) fn scalar<P>(value: P) -> Word
where
    P: Primitive,
{
    let kind = scalar_kind::<P>();
    let mut word = 0u64;
    // SAFETY: `P`'s canonical form is the scalar `kind` names, and `P` has
    // its canonical form's size and alignment by `P`'s `Structural`
    // signature; every scalar is at most 8 bytes aligned to at most 8, so a
    // `P` fits the word's first bytes at the word's alignment.
    unsafe { ptr::from_mut(&mut word).cast::<P>().write(value) };
    Word { kind, word }
}

/// # Safety
/// `v` is a value `scalar` made from a `P`.
unsafe fn read_scalar<P>(v: Word) -> P
where
    P: Primitive,
{
    // SAFETY: `scalar` wrote a `P` in the word's first bytes.
    unsafe { ptr::from_ref(&v.word).cast::<P>().read() }
}

/// A value held in a slot of its own.
fn slotted<T>(value: T) -> Word
where
    T: 'static,
{
    Slot::allocate(Kind::Large, value, TypeId::of::<T>())
}

/// # Safety
/// `v` is a live value `slotted` made from a `T`, and nothing uses it
/// afterwards.
unsafe fn unslotted<T>(v: Word) -> T {
    // SAFETY: the caller's contract.
    unsafe { Slot::<T>::take(v.address()) }
}

/// Where the `P` the value at `place` lends lies: a scalar's own word; a
/// value's slot; through a `Ref`, the word or the slot of the value it names;
/// through a `Part`, its address.
///
/// # Safety
/// `place` holds a live value, and the storage it names is live and holds a
/// `P`.
unsafe fn target<P>(place: *mut Word) -> *mut P {
    // SAFETY: the caller's contract.
    let v = unsafe { place.read() };
    match v.kind {
        // SAFETY: as above; the scalar lies in the word's first bytes.
        kind if kind.is_scalar() => unsafe { (&raw mut (*place).word).cast::<P>() },
        // SAFETY: a live `Large` word points at a live `Slot<P>`.
        Kind::Large => unsafe { &raw mut (*v.address::<Slot<P>>()).value },
        // SAFETY: a `Ref` names a live `Word` holding a `P`.
        Kind::Ref => unsafe { target(v.address::<Word>()) },
        Kind::Part => v.address::<P>(),
        kind => unreachable!("a {kind:?} value names no storage of a Rust type"),
    }
}

/// The wide pointer a `WidePart` holds.
type WideBits = MaybeUninit<[usize; 2]>;

/// The allocations `reference` made for wide pointers, freed when the runtime
/// drops: a reference value owns nothing, so no holder of one frees it, and a
/// copy of it may be read until the runtime is gone.
#[derive(Default)]
pub(crate) struct WideParts(Mutex<Vec<WidePart>>);

/// One allocation `reference` boxed.
struct WidePart(*mut WideBits);

// SAFETY: the pointer is a `Box`'s, which `WideParts` alone frees; the bits
// it holds are an address and a length, readable from any thread.
unsafe impl Send for WidePart {}

impl WideParts {
    fn keep(&self, bits: *mut WideBits) {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).push(WidePart(bits));
    }
}

impl Drop for WideParts {
    fn drop(&mut self) {
        let parts = self.0.get_mut().unwrap_or_else(PoisonError::into_inner);
        for part in parts.drain(..) {
            // SAFETY: `reference` made the pointer by `Box::into_raw`, and
            // `keep` took it once; the runtime is dropping, so no reference
            // value naming it is read again.
            drop(unsafe { Box::from_raw(part.0) });
        }
    }
}

/// # Safety
/// `v` is a live `WidePart` made from a `*mut P`.
unsafe fn wide_target<P>(v: Word) -> *mut P
where
    P: ?Sized,
{
    debug_assert_eq!(v.kind, Kind::WidePart);
    // SAFETY: `reference` wrote a `*mut P` at the start of this allocation.
    unsafe { v.address::<WideBits>().cast::<*mut P>().read() }
}

/// # Safety
/// `v` is live, and nothing uses it afterwards.
unsafe fn free(v: Word) {
    match v.kind {
        Kind::Large | Kind::Closure => {
            let header = v.address::<Header>();
            // SAFETY: the word points at a live slot's header, and the
            // caller's contract makes this its one use.
            unsafe { ((*header).release)(header) }
        }
        // A reference value owns nothing; a wide pointer's allocation is the
        // runtime's (`WideParts`).
        Kind::WidePart
        | Kind::Ref
        | Kind::Part
        | Kind::I8
        | Kind::I16
        | Kind::I32
        | Kind::I64
        | Kind::U8
        | Kind::U16
        | Kind::U32
        | Kind::U64
        | Kind::F64
        | Kind::Bool
        | Kind::Char
        | Kind::Unit => {}
    }
}

impl Release for Word {
    fn release(self, _releasing: Releasing) {
        // SAFETY: only the drop of the one holder that owns a value makes a
        // `Releasing`, so this is the value's one release.
        unsafe { free(self) }
    }
}

/// A value the run owns: a call's result, released by its drop. It stands in
/// for a program's variable, and for a part an option, a result, a tuple or an
/// array holds.
#[repr(transparent)]
pub struct Held(Word);

impl Held {
    /// The result of a call, which the runtime owns from here on.
    pub fn of_the_result(done: Done<WordRuntime>) -> Held {
        Held(done.into_value())
    }

    /// The value handed to a call that takes it by value, which then owns it.
    pub fn hand_over(self) -> Word {
        ManuallyDrop::new(self).0
    }

    /// A `Ref` to this value, for a call it is lent to.
    pub fn lend(&mut self) -> Word {
        Word::pointing(Kind::Ref, ptr::from_mut(&mut self.0))
    }

    /// The value itself, still owned here: a reference value passed to a call
    /// that reads what it names.
    pub fn same(&self) -> Word {
        self.0
    }

    /// A value the runtime made and keeps, released by this holder's drop.
    pub(crate) fn kept(value: Word) -> Held {
        Held(value)
    }

    /// What the option this value is holds, where its part is a reference:
    /// the value the reference names, read at `P` and copied out. The run
    /// reads it as the checker settled its type, `Option<&P>`.
    pub(crate) fn option_referent<P>(&self) -> Option<P::Owned>
    where
        P: ?Sized + ToOwned,
    {
        assert_eq!(self.0.kind, Kind::Large, "an option lies in a slot");
        // SAFETY: the value is an option, a slot holding an `Option<Held>`
        // whose part is a reference value made from a `P` that is still
        // live, as the checker settled for the variable.
        let part = unsafe { &(*self.0.address::<Slot<Option<Held>>>()).value };
        // SAFETY: as above, for the reference the part is.
        part.as_ref().map(|reference| unsafe { (*named_part::<P>(reference.0)).to_owned() })
    }

    /// What the pair this value is holds, where both parts are references:
    /// the values they name, read at `P` and copied out.
    pub(crate) fn pair_referents<P>(&self) -> (P::Owned, P::Owned)
    where
        P: ?Sized + ToOwned,
    {
        assert_eq!(self.0.kind, Kind::Large, "a tuple lies in a slot");
        // SAFETY: the value is a pair, a slot holding `[Held; 2]` whose parts
        // are reference values made from `P`s that are still live, as the
        // checker settled for the variable.
        let [left, right] = unsafe { &(*self.0.address::<Slot<[Held; 2]>>()).value };
        // SAFETY: as above, for the references the parts are.
        unsafe { ((*named_part::<P>(left.0)).to_owned(), (*named_part::<P>(right.0)).to_owned()) }
    }

    /// The function this value is. The run calls a function value it holds
    /// as it passes arguments: trusting that the checker typed it a function.
    pub(crate) fn function(&self) -> &Callee {
        assert_eq!(self.0.kind, Kind::Closure, "a call of a value that is no function");
        // SAFETY: a `Closure` word points at a live slot `closure` made from a
        // boxed function, and the slot lives as long as this holder owns the
        // value.
        unsafe { &*(*self.0.address::<Slot<Box<Callee>>>()).value }
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        // SAFETY: a `Held` owns its value: `of_the_result` took it from the
        // call that made it, and `hand_over` gives it away without this drop.
        unsafe { free(self.0) }
    }
}

// The typed reads. Each takes the entry's `Settled`: under it, `v` is a live
// value of the type the checker settled for its site, which is the type
// asked for here (safety §3), and storage it names stays live and in place
// for as long as the call lends it (S1).

pub(crate) fn give_opaque<T>(value: T) -> Word
where
    T: Opaque,
{
    Slot::allocate(Kind::Large, value, TypeId::of::<T::Canon>())
}

/// A tuple's parts as they lie in its slot: `Held` values, each a `Word` by
/// `repr(transparent)`.
fn parts<const N: usize>(held: &[Held; N]) -> &[Word; N] {
    // SAFETY: `Held` is `repr(transparent)` over `Word`.
    unsafe { &*ptr::from_ref(held).cast::<[Word; N]>() }
}

fn parts_mut<const N: usize>(held: &mut [Held; N]) -> &mut [Word; N] {
    // SAFETY: as `parts`, exclusively.
    unsafe { &mut *ptr::from_mut(held).cast::<[Word; N]>() }
}

pub(crate) fn take_opaque<T>(v: Word, _at: Settled<'_>) -> T
where
    T: Opaque,
{
    // SAFETY: under `Settled`, `v` is a value `give_opaque` made from a `T`,
    // which its one holder moves out.
    unsafe { unslotted(v) }
}

pub(crate) fn take_primitive<P>(v: Word, _at: Settled<'_>) -> P
where
    P: Primitive,
{
    // SAFETY: under `Settled`, `v` is a value `scalar` made from a `P`.
    unsafe { read_scalar(v) }
}

pub(crate) fn give_string(value: String) -> Word {
    slotted(value)
}

pub(crate) fn take_string(v: Word, _at: Settled<'_>) -> String {
    // SAFETY: under `Settled`, `v` is text, a `String` in a slot, which its
    // one holder moves out.
    unsafe { unslotted(v) }
}

pub(crate) fn give_option(value: Option<Word>) -> Word {
    slotted(value.map(Held::kept))
}

pub(crate) fn take_option(v: Word, _at: Settled<'_>) -> Option<Word> {
    // SAFETY: under `Settled`, `v` is an option, an `Option<Held>` in a slot,
    // which its one holder moves out.
    unsafe { unslotted::<Option<Held>>(v) }.map(Held::hand_over)
}

pub(crate) fn give_result(value: Result<Word, Word>) -> Word {
    slotted(value.map(Held::kept).map_err(Held::kept))
}

pub(crate) fn take_result(v: Word, _at: Settled<'_>) -> Result<Word, Word> {
    // SAFETY: under `Settled`, `v` is a result, a `Result<Held, Held>` in a
    // slot, which its one holder moves out.
    unsafe { unslotted::<Result<Held, Held>>(v) }.map(Held::hand_over).map_err(Held::hand_over)
}

pub(crate) fn give_tuple<const N: usize>(parts: [Word; N]) -> Word {
    slotted(parts.map(Held::kept))
}

pub(crate) fn take_tuple<const N: usize>(v: Word, _at: Settled<'_>) -> [Word; N] {
    // SAFETY: under `Settled`, `v` is a tuple of `N` parts, a `[Held; N]` in a
    // slot, which its one holder moves out.
    unsafe { unslotted::<[Held; N]>(v) }.map(Held::hand_over)
}

/// A program's array: a slot holding its elements.
pub(crate) fn array(elements: Vec<Word>) -> Word {
    slotted(elements.into_iter().map(Held::kept).collect::<Vec<Held>>())
}

pub(crate) fn place<'a, T>(v: &'a Word, _at: Settled<'a>) -> &'a T {
    // SAFETY: under `Settled`, `v` lends storage holding a `T` for `'a`, and
    // no exclusive lend of it is live meanwhile.
    unsafe { &*target::<T>(ptr::from_ref(v).cast_mut()) }
}

pub(crate) fn place_mut<'a, T>(v: &'a mut Word, _at: Settled<'a>) -> &'a mut T {
    // SAFETY: under `Settled`, `v` lends storage holding a `T` exclusively for
    // `'a`, and no other lend of it is live meanwhile.
    unsafe { &mut *target::<T>(ptr::from_mut(v)) }
}

/// A reference value to a part of storage the call lent. It names the part
/// where it lies; a writable one carries the exclusive borrow's provenance. A
/// wide pointer is boxed, and the box kept in `wide`.
pub(crate) fn reference<P>(wide: &WideParts, target: RefTarget<'_, P>) -> Word
where
    P: ?Sized,
{
    let part: *mut P = match target {
        RefTarget::Shared(part) => ptr::from_ref(part).cast_mut(),
        RefTarget::Exclusive(part) => ptr::from_mut(part),
    };
    if size_of::<*mut P>() == size_of::<*mut u8>() {
        return Word::pointing(Kind::Part, part.cast::<u8>());
    }
    const {
        assert!(
            size_of::<*mut P>() <= size_of::<WideBits>() && align_of::<*mut P>() <= align_of::<WideBits>(),
            "a wide pointer fits two words"
        )
    };
    let bits = Box::into_raw(Box::new(WideBits::uninit()));
    // SAFETY: the allocation is fresh, and a `*mut P` fits it at its
    // alignment by the assertion above.
    unsafe { bits.cast::<*mut P>().write(part) };
    wide.keep(bits);
    Word::pointing(Kind::WidePart, bits)
}

/// A function value as this runtime holds it: its dispatch, called with an
/// argument run.
pub(crate) type Callee = dyn Fn(&WordRuntime, &[Word], &SiteShape<WordRuntime, 0>) -> Done<WordRuntime> + Send + Sync;

/// A script function value, a function's dispatch.
pub(crate) fn closure(f: Box<Callee>) -> Word {
    Slot::allocate(Kind::Closure, f, TypeId::of::<Box<Callee>>())
}

/// What a reference value `reference` made names, for the loan `'l` it was
/// made under.
///
/// # Safety
/// `v` is a live `Part`, `WidePart` or `Ref` made from a `P`, and the storage it
/// names is live for `'l`.
unsafe fn named_part<P>(v: Word) -> *mut P
where
    P: ?Sized,
{
    match v.kind {
        Kind::Part => {
            assert!(
                size_of::<*mut P>() == size_of::<*mut u8>(),
                "a thin reference value names a thin target"
            );
            let thin = v.address::<u8>();
            // SAFETY: a `*mut P` of a thin `P` is the address alone, as
            // `reference` wrote it.
            unsafe { ptr::from_ref(&thin).cast::<*mut P>().read() }
        }
        // SAFETY: the caller's contract.
        Kind::WidePart => unsafe { wide_target::<P>(v) },
        // SAFETY: a `Ref` names a live `Word` whose storage holds a `P`, or,
        // for a wide `P`, a `String` read as `str` or an array, a `Vec` of
        // words, read as the run of its values.
        Kind::Ref => unsafe { named_whole::<P>(v.address::<Word>()) },
        kind => unreachable!("a {kind:?} value is no reference value"),
    }
}

/// What a `Ref` to the value at `place` names at `P`: its storage for a thin
/// `P`; for a wide one, the text of the `String` or the run of the array's
/// values it holds.
///
/// # Safety
/// `place` holds a live value of `P`, a `String` read as `str`, or an array
/// read as the run of its values.
unsafe fn named_whole<P>(place: *mut Word) -> *mut P
where
    P: ?Sized,
{
    if size_of::<*mut P>() == size_of::<*mut u8>() {
        // SAFETY: the caller's contract, for a thin `P`; a thin `*mut P` is
        // the address alone.
        let thin = unsafe { target::<u8>(place) };
        // SAFETY: as above.
        return unsafe { ptr::from_ref(&thin).cast::<*mut P>().read() };
    }
    // SAFETY: the caller's contract: a wide `P` is read from a slot.
    let slot = unsafe { place.read() };
    assert_eq!(slot.kind, Kind::Large, "a wide target is read from a slot");
    // SAFETY: a live `Large` word points at a live slot's header.
    let canon = unsafe { (*slot.address::<Header>()).canon };
    let wide: *mut [u8] = if canon == TypeId::of::<String>() {
        // SAFETY: the slot holds a `String`, whose text is the `str`.
        let text = unsafe { &mut (*slot.address::<Slot<String>>()).value };
        ptr::from_mut(text.as_mut_str()) as *mut [u8]
    } else {
        // SAFETY: the slot holds an array, a `Vec` of `Held` values, each a
        // `Word` by `repr(transparent)`, whose run is the slice of its values.
        let run = unsafe { &mut (*slot.address::<Slot<Vec<Held>>>()).value };
        ptr::slice_from_raw_parts_mut(run.as_mut_ptr().cast::<u8>(), run.len())
    };
    // SAFETY: `str` and `[Word]` share `*mut [u8]`'s metadata, a length in
    // their own units, and a wide `*mut P` of either is that pair.
    unsafe { ptr::from_ref(&wide).cast::<*mut P>().read() }
}

/// The value a reference value names where the runtime holds its target as a
/// value of its own: a `Ref` names a variable's word, and a `Part` a word in
/// lent storage (a type variable's value, an element of an array's run).
fn named_value(v: Word) -> *mut Word {
    match v.kind {
        Kind::Ref | Kind::Part => v.address::<Word>(),
        kind => unreachable!("a {kind:?} value names no value of the runtime's"),
    }
}

pub(crate) fn named_word<'l>(v: Word, _at: Settled<'l>) -> &'l Word {
    // SAFETY: under `Settled`, `v` is a reference value to a value the runtime
    // holds as its own, which names that value's word until the loan `'l`
    // ends (S2), and no exclusive lend of it is live meanwhile.
    unsafe { &*named_value(v) }
}

pub(crate) fn named_word_mut<'l>(v: Word, _at: Settled<'l>) -> &'l mut Word {
    // SAFETY: as `named_word`'s, for a writable reference value made from an
    // exclusive borrow, whose provenance it carries.
    unsafe { &mut *named_value(v) }
}

pub(crate) fn referent<'l, P>(v: Word, _at: Settled<'l>) -> &'l P
where
    P: ?Sized,
{
    // SAFETY: under `Settled`, `v` is a reference value made from a `P` the
    // call lent, which names it until the loan `'l` ends (S2), and no
    // exclusive lend of it is live meanwhile.
    unsafe { &*named_part::<P>(v) }
}

pub(crate) fn referent_mut<'l, P>(v: Word, _at: Settled<'l>) -> &'l mut P
where
    P: ?Sized,
{
    // SAFETY: as `referent`'s, for a writable reference value made from an
    // exclusive borrow, whose provenance it carries.
    unsafe { &mut *named_part::<P>(v) }
}

/// The Rust function a closure value is.
pub(crate) fn callee<'a>(v: Word, _at: Settled<'a>) -> &'a Callee {
    // SAFETY: under `Settled`, `v` is a closure the call is lent, a slot
    // `closure` made from a boxed function, live for the call.
    unsafe { &*(*v.address::<Slot<Box<Callee>>>()).value }
}

/// `WordRuntime`'s kinds, over the typed reads above.
pub struct Kinds;

impl OpaqueKind<WordRuntime> for Kinds {

    fn give<T: Opaque>(_: &WordRuntime, value: T, _: Settled<'_>) -> Word {
        give_opaque(value)
    }
    fn take<T: Opaque>(_: &WordRuntime, v: Word, at: Settled<'_>) -> T {
        take_opaque(v, at)
    }
    fn shared<'a, T: Opaque + 'a>(_: &'a WordRuntime, place: &'a Word, at: Settled<'a>) -> &'a T {
        self::place(place, at)
    }
    fn exclusive<'a, T: Opaque + 'a>(_: &'a WordRuntime, place: &'a mut Word, at: Settled<'a>) -> &'a mut T {
        place_mut(place, at)
    }
}

impl PrimitiveKind<WordRuntime> for Kinds {

    fn give<P: Primitive>(_: &WordRuntime, value: P, _: Settled<'_>) -> Word {
        scalar(value)
    }
    fn take<P: Primitive>(_: &WordRuntime, v: Word, at: Settled<'_>) -> P {
        take_primitive(v, at)
    }
    fn shared<'a, P: Primitive>(_: &'a WordRuntime, place: &'a Word, at: Settled<'a>) -> &'a P {
        self::place(place, at)
    }
    fn exclusive<'a, P: Primitive>(_: &'a WordRuntime, place: &'a mut Word, at: Settled<'a>) -> &'a mut P {
        place_mut(place, at)
    }
}

impl StringKind<WordRuntime> for Kinds {

    fn give(_: &WordRuntime, value: String, _: Settled<'_>) -> Word {
        give_string(value)
    }
    fn take(_: &WordRuntime, v: Word, at: Settled<'_>) -> String {
        take_string(v, at)
    }
    fn shared<'a>(_: &'a WordRuntime, place: &'a Word, at: Settled<'a>) -> &'a String {
        self::place(place, at)
    }
    fn exclusive<'a>(_: &'a WordRuntime, place: &'a mut Word, at: Settled<'a>) -> &'a mut String {
        place_mut(place, at)
    }
}

impl OptionKind<WordRuntime> for Kinds {
    fn give(_: &WordRuntime, value: Option<Word>, _: Settled<'_>) -> Word {
        give_option(value)
    }
    fn take(_: &WordRuntime, v: Word, at: Settled<'_>) -> Option<Word> {
        take_option(v, at)
    }
    fn shared<'a>(_: &'a WordRuntime, place: &'a Word, at: Settled<'a>) -> Option<&'a Word> {
        self::place::<Option<Held>>(place, at).as_ref().map(|part| &part.0)
    }
    fn exclusive<'a>(_: &'a WordRuntime, place: &'a mut Word, at: Settled<'a>) -> Option<&'a mut Word> {
        place_mut::<Option<Held>>(place, at).as_mut().map(|part| &mut part.0)
    }
}

impl ResultKind<WordRuntime> for Kinds {
    fn give(_: &WordRuntime, value: Result<Word, Word>, _: Settled<'_>) -> Word {
        give_result(value)
    }
    fn take(_: &WordRuntime, v: Word, at: Settled<'_>) -> Result<Word, Word> {
        take_result(v, at)
    }
    fn shared<'a>(_: &'a WordRuntime, place: &'a Word, at: Settled<'a>) -> Result<&'a Word, &'a Word> {
        self::place::<Result<Held, Held>>(place, at).as_ref().map(|part| &part.0).map_err(|part| &part.0)
    }
    fn exclusive<'a>(_: &'a WordRuntime, place: &'a mut Word, at: Settled<'a>) -> Result<&'a mut Word, &'a mut Word> {
        place_mut::<Result<Held, Held>>(place, at).as_mut().map(|part| &mut part.0).map_err(|part| &mut part.0)
    }
}

impl TupleKind<WordRuntime> for Kinds {
    fn give<const N: usize>(_: &WordRuntime, parts: [Word; N], _: Settled<'_>) -> Word {
        give_tuple(parts)
    }
    fn take<const N: usize>(_: &WordRuntime, v: Word, at: Settled<'_>) -> [Word; N] {
        take_tuple(v, at)
    }
    fn shared<'a, const N: usize>(_: &'a WordRuntime, place: &'a Word, at: Settled<'a>) -> &'a [Word; N] {
        parts(self::place(place, at))
    }
    fn exclusive<'a, const N: usize>(_: &'a WordRuntime, place: &'a mut Word, at: Settled<'a>) -> &'a mut [Word; N] {
        parts_mut(place_mut(place, at))
    }
}

impl RefKind<WordRuntime> for Kinds {
    fn reference<'a, P: ?Sized>(rt: &'a WordRuntime, target: RefTarget<'a, P>, _: Settled<'a>) -> Word {
        reference(&rt.wide, target)
    }
    fn target<'l, P: ?Sized>(_: &'l WordRuntime, v: Word, at: Settled<'l>) -> &'l P {
        referent(v, at)
    }
    fn target_mut<'l, P: ?Sized>(_: &'l WordRuntime, v: Word, at: Settled<'l>) -> &'l mut P {
        referent_mut(v, at)
    }
    fn place<'l>(_: &'l WordRuntime, v: Word, at: Settled<'l>) -> &'l Word {
        named_word(v, at)
    }
    fn place_mut<'l>(_: &'l WordRuntime, v: Word, at: Settled<'l>) -> &'l mut Word {
        named_word_mut(v, at)
    }
}

// SAFETY: `WordRuntime` lends three kinds of storage. A scalar is lent in the
// word at the place the guard is given, the holder's own: an argument run the
// run keeps for the call, a `Uniform`, or a variable. A value's slot is
// allocated once by `Slot::allocate` and never moved; it is freed only by the
// release of the one holder that owns the value or by `take` moving the value
// out, and the checker's loan on a value excludes each of these while it
// lasts. A word the run keeps stays where the run put it: `Held` is boxed by
// the run, and this runtime moves no place it is given. `WordRuntime` writes
// none of them after making them except through the `&mut` an exclusive guard
// is, so each keeps its address and is written only under an exclusive loan
// until the checker's loan ends.
unsafe impl SignedS1 for WordRuntime {}
// SAFETY: `reference` keeps the part's own address, in the word or boxed with
// its metadata, and makes no copy of the part. By S1 the part stays at that
// address until the checker's loan ends, so the reference names it.
unsafe impl SignedS2 for WordRuntime {}
// SAFETY: `WordRuntime` starts no run that outlives the call starting it. A
// closure is a Rust function, and calling one, returning or awaited, runs
// its glue to the end before the call returns or its future completes; an
// awaited call's future holds only the values moved into it, which it
// releases when dropped, and lends none of the caller's storage (crossing
// §3). No work is left aloft when a call ends, so storage lent to a callee
// is read only within the lend, and the runtime keeps no value past the
// call that keeps it. It releases an allocation only when the value that
// owns it is released, moved out or handed over.
unsafe impl SignedS3 for WordRuntime {}

#[cfg(test)]
mod tests {
    use std::any::TypeId;
    use std::sync::Arc;

    use acvus_extern::runtime::RefTarget;

    use super::{Kind, Slot, WideParts, Word, free, named_part, read_scalar, reference, scalar, slotted, target, unslotted};

    #[test]
    fn a_scalar_round_trips_through_the_word() {
        let words = [
            scalar(-7i8),
            scalar(u64::MAX),
            scalar(1.5f64),
            scalar('\u{10FFFF}'),
            scalar(true),
        ];

        assert_eq!(words.map(Word::kind), [Kind::I8, Kind::U64, Kind::F64, Kind::Char, Kind::Bool]);
        // SAFETY: each word was made from the type it is read at.
        unsafe {
            assert_eq!(read_scalar::<i8>(words[0]), -7);
            assert_eq!(read_scalar::<u64>(words[1]), u64::MAX);
            assert_eq!(read_scalar::<f64>(words[2]), 1.5);
            assert_eq!(read_scalar::<char>(words[3]), '\u{10FFFF}');
            assert!(read_scalar::<bool>(words[4]));
        }
    }

    #[test]
    fn a_value_larger_than_a_scalar_round_trips_through_an_allocation() {
        let word = slotted(String::from("large"));

        assert_eq!(word.kind(), Kind::Large);
        // SAFETY: the word was slotted from a `String`, and is read once.
        assert_eq!(unsafe { unslotted::<String>(word) }, "large");
    }

    /// A value in a slot whose drop is counted by the strong count of an
    /// `Arc`.
    fn counted(alive: &Arc<()>) -> Word {
        Slot::allocate(Kind::Large, Arc::clone(alive), TypeId::of::<Arc<()>>())
    }

    #[test]
    fn an_allocation_is_released_once_by_its_release() {
        let alive = Arc::new(());
        let word = counted(&alive);
        assert_eq!(Arc::strong_count(&alive), 2);

        // SAFETY: the word is live and released once.
        unsafe { free(word) };

        assert_eq!(Arc::strong_count(&alive), 1);
    }

    #[test]
    fn a_value_moved_out_of_its_allocation_is_dropped_by_its_new_owner_only() {
        let alive = Arc::new(());
        let word = counted(&alive);

        // SAFETY: the slot was allocated at `Arc<()>`, and is taken once.
        let moved = unsafe { Slot::<Arc<()>>::take(word.address()) };
        assert_eq!(Arc::strong_count(&alive), 2);
        drop(moved);

        assert_eq!(Arc::strong_count(&alive), 1);
    }

    #[test]
    fn a_write_through_a_lent_scalar_lands_in_the_word_it_names() {
        let mut kept = scalar(5i16);
        let mut lent = Word::pointing(Kind::Ref, &raw mut kept);

        // SAFETY: `lent` names `kept`, a live `i16`, and nothing else uses
        // `kept` while the write lasts.
        unsafe { *target::<i16>(&raw mut lent) = -300 };

        // SAFETY: `kept` holds an `i16`, read once.
        assert_eq!(unsafe { read_scalar::<i16>(kept) }, -300);
    }

    #[test]
    fn a_write_through_a_lent_allocation_lands_in_its_slot() {
        let mut kept = slotted(String::from("lent"));
        let mut lent = Word::pointing(Kind::Ref, &raw mut kept);

        // SAFETY: as for the scalar, for a `String` in a slot.
        unsafe { (*target::<String>(&raw mut lent)).push_str(" and written") };

        // SAFETY: `kept` holds a `String`, read once.
        assert_eq!(unsafe { unslotted::<String>(kept) }, "lent and written");
    }

    #[test]
    fn a_reference_value_reads_back_the_thin_target_it_was_made_from() {
        let lent = String::from("thin");

        let v = reference(&WideParts::default(), RefTarget::Shared(&lent));

        assert_eq!(v.kind(), Kind::Part);
        // SAFETY: `v` was made from `&lent`, which is live and shared.
        assert!(std::ptr::eq(unsafe { named_part::<String>(v) }, &lent));
    }

    #[test]
    fn a_reference_value_reads_back_the_text_and_run_it_was_made_from() {
        let text = String::from("a wide target");
        let run = [3i64, 5, 8];

        let wide = WideParts::default();
        let text_v = reference(&wide, RefTarget::Shared(&text[2..6]));
        let run_v = reference(&wide, RefTarget::Shared(&run[1..]));

        assert_eq!((text_v.kind(), run_v.kind()), (Kind::WidePart, Kind::WidePart));
        // SAFETY: each value was made from a live shared borrow of its type.
        unsafe {
            assert_eq!(&*named_part::<str>(text_v), "wide");
            assert_eq!(&*named_part::<[i64]>(run_v), &[5, 8]);
        }
    }

    #[test]
    fn a_write_through_a_writable_reference_value_lands_in_its_target() {
        let mut run = [3i64, 5, 8];

        let wide = WideParts::default();
        let v = reference(&wide, RefTarget::Exclusive(&mut run[..2]));
        // SAFETY: `v` was made from the exclusive borrow above, which nothing
        // else uses until the write ends.
        unsafe { (*named_part::<[i64]>(v))[1] = -5 };

        assert_eq!(run, [3, -5, 8]);
    }
}
