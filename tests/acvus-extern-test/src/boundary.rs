//! The only module that writes `unsafe`: the facts `WordRuntime` signs, the
//! casts between a word and what it holds, and the structural facts of the
//! test types.
//!
//! A value is a kind and a 64-bit word. A scalar lies in the word's first
//! bytes. Anything larger lies in an allocation the word points at: a Rust
//! value or a Rust function's glue in a `Slot` headed by its release and its
//! canonical type. A reference value is a `Ref` word: a pointer to the `Word`
//! that is the place of the value it names, a variable the run keeps, a box,
//! or an element of an array. A reference value owns nothing (S2).
//!
//! An opaque value, text, an option, a result and a tuple each lie in a slot:
//! the Rust value itself, and a part of an option, a result or a tuple as a
//! `Held` value the slot owns. An array is a slot holding a `Vec` of `Held`
//! values, the run of its elements' places.
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
use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::Arc;

use acvus_extern::done::Done;
use acvus_extern::runtime::{
    Opaque, OpaqueKind, OptionKind, PrimitiveKind, RefKind, ResultKind, SequenceKind, StringKind, TupleKind,
};
use acvus_extern::{Primitive, Release, Releasing, Settled, SiteShape, SignedS1, SignedS2, SignedS3};

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

/// Where the primitive a scalar value's place holds lies: its word's first
/// bytes.
///
/// # Safety
/// `place` holds a value `scalar` made from a `P`.
unsafe fn scalar_at<P>(place: *mut Word) -> *mut P
where
    P: Primitive,
{
    // SAFETY: the caller's contract; a `P` lies in the word's first bytes at
    // the word's alignment, as `scalar` wrote it.
    unsafe { (&raw mut (*place).word).cast::<P>() }
}

/// Where the Rust value a slotted value's place holds lies: in its slot.
///
/// # Safety
/// `place` holds a live value whose slot holds a `T`.
unsafe fn slot_at<T>(place: *mut Word) -> *mut T {
    // SAFETY: the caller's contract: a live `Large` word points at a live
    // `Slot<T>`.
    unsafe { &raw mut (*place.read().address::<Slot<T>>()).value }
}

/// A reference value naming the place `place`. One made from an exclusive
/// borrow carries its provenance, and is written through.
fn reference(place: *mut Word) -> Word {
    Word::pointing(Kind::Ref, place)
}

/// The place a reference value names.
fn named(v: Word) -> *mut Word {
    match v.kind {
        Kind::Ref => v.address::<Word>(),
        kind => unreachable!("a {kind:?} value is no reference value"),
    }
}

/// What the text value at `place` holds.
///
/// # Safety
/// `place` holds a live value, and the storage it names stays live and
/// unwritten for `'a`.
unsafe fn text_at<'a>(place: *mut Word) -> &'a str {
    // SAFETY: the caller's contract.
    let v = unsafe { place.read() };
    assert_eq!(v.kind, Kind::Large, "text lies in a slot");
    // SAFETY: a live `Large` word points at a live slot's header.
    let canon = unsafe { (*v.address::<Header>()).canon };
    assert_eq!(canon, TypeId::of::<String>(), "the slot holds text");
    // SAFETY: the slot holds a `String`, live for `'a` by the caller's
    // contract.
    unsafe { (*slot_at::<String>(place)).as_str() }
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
        // A reference value owns nothing.
        Kind::Ref
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

    /// A reference value naming this value's place, for a call it is lent
    /// to.
    pub fn lend(&mut self) -> Word {
        reference(ptr::from_mut(&mut self.0))
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

    /// The value the option this value is names through its part, a
    /// reference value, where it is present: the named place's word, copied
    /// out. The run reads it as the checker settled its type, `Option<&P>`.
    pub(crate) fn option_referent(&self) -> Option<Word> {
        assert_eq!(self.0.kind, Kind::Large, "an option lies in a slot");
        // SAFETY: the value is an option, a slot holding an `Option<Held>`
        // whose part is a reference value naming a place that is still live,
        // as the checker settled for the variable.
        let part = unsafe { &(*self.0.address::<Slot<Option<Held>>>()).value };
        // SAFETY: as above, for the place the part names.
        part.as_ref().map(|reference| unsafe { named(reference.0).read() })
    }

    /// The texts the pair this value is names through its parts, both
    /// reference values to text, copied out.
    pub(crate) fn pair_texts(&self) -> (String, String) {
        assert_eq!(self.0.kind, Kind::Large, "a tuple lies in a slot");
        // SAFETY: the value is a pair, a slot holding `[Held; 2]` whose parts
        // are reference values naming places of text that are still live, as
        // the checker settled for the variable.
        let [left, right] = unsafe { &(*self.0.address::<Slot<[Held; 2]>>()).value };
        // SAFETY: as above, for the places the parts name; nothing writes
        // them while this holder is borrowed.
        unsafe { (String::from(text_at(named(left.0))), String::from(text_at(named(right.0)))) }
    }

    /// The function this value is. The run calls a function value it holds
    /// as it passes arguments: trusting that the checker typed it a function.
    pub(crate) fn function(&self) -> &Callee {
        assert_eq!(self.0.kind, Kind::Closure, "a call of a value that is no function");
        // SAFETY: a `Closure` word points at a live slot `closure` or
        // `owned_closure` made from a shared function, and the slot lives as
        // long as this holder owns the value.
        unsafe { &*(*self.0.address::<Slot<Arc<Callee>>>()).value }
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

pub(crate) fn scalar_in<'a, P>(place: &'a Word, _at: Settled<'a>) -> &'a P
where
    P: Primitive,
{
    // SAFETY: under `Settled`, `place` holds a value `scalar` made from a
    // `P`, viewed for as long as the place is borrowed.
    unsafe { &*scalar_at::<P>(ptr::from_ref(place).cast_mut()) }
}

pub(crate) fn scalar_in_mut<'a, P>(place: &'a mut Word, _at: Settled<'a>) -> &'a mut P
where
    P: Primitive,
{
    // SAFETY: as `scalar_in`'s, exclusively: a `P` written through the view
    // is a value of the scalar kind `scalar` gave the word.
    unsafe { &mut *scalar_at::<P>(ptr::from_mut(place)) }
}

pub(crate) fn slot_value<'a, T>(place: &'a Word, _at: Settled<'a>) -> &'a T {
    // SAFETY: under `Settled`, `place` holds a live value whose slot holds a
    // `T`. The slot is freed only by the release of the value's owner or by a
    // take, and the borrow of the place excludes both while the view lasts;
    // no exclusive view of it is live meanwhile.
    unsafe { &*slot_at::<T>(ptr::from_ref(place).cast_mut()) }
}

pub(crate) fn slot_value_mut<'a, T>(place: &'a mut Word, _at: Settled<'a>) -> &'a mut T {
    // SAFETY: as `slot_value`'s, under the exclusive borrow of the place.
    unsafe { &mut *slot_at::<T>(ptr::from_mut(place)) }
}

/// An array's elements' places: the run of `Held` values its slot holds.
pub(crate) fn elements<'a>(place: &'a Word, at: Settled<'a>) -> &'a [Word] {
    let run = slot_value::<Vec<Held>>(place, at).as_slice();
    // SAFETY: `Held` is `repr(transparent)` over `Word`, so a run of held
    // elements is a run of words of the same length.
    unsafe { &*(ptr::from_ref(run) as *const [Word]) }
}

pub(crate) fn elements_mut<'a>(place: &'a mut Word, at: Settled<'a>) -> &'a mut [Word] {
    let run = slot_value_mut::<Vec<Held>>(place, at).as_mut_slice();
    // SAFETY: as `elements`, exclusively. A word written in an element's
    // place is owned by the `Held` there, which releases it once.
    unsafe { &mut *(ptr::from_mut(run) as *mut [Word]) }
}

/// A function value as this runtime holds it: its dispatch, called with an
/// argument run.
pub(crate) type Callee = dyn Fn(&WordRuntime, &[Word], &SiteShape<WordRuntime, 0>) -> Done<WordRuntime> + Send + Sync;

/// A script function value, a function's dispatch, shared by each value of
/// it the runtime owns.
pub(crate) fn closure(f: Box<Callee>) -> Word {
    let f: Arc<Callee> = Arc::from(f);
    Slot::allocate(Kind::Closure, f, TypeId::of::<Arc<Callee>>())
}

/// Another owned value of the function value `v`: the same dispatch, in a
/// slot of its own, released by its own holder.
pub(crate) fn owned_closure(v: Word, _at: Settled<'_>) -> Word {
    // SAFETY: under `Settled`, `v` is a closure the call is lent, a slot
    // `closure` or `owned_closure` made, live for the call.
    let f = unsafe { &(*v.address::<Slot<Arc<Callee>>>()).value };
    Slot::allocate(Kind::Closure, Arc::clone(f), TypeId::of::<Arc<Callee>>())
}

pub(crate) fn named_place<'l>(v: Word, _at: Settled<'l>) -> &'l Word {
    // SAFETY: under `Settled`, `v` is a reference value made from a place the
    // run or a call borrows, which names that place until the loan `'l`
    // ends (S2), and no exclusive borrow of it is live meanwhile.
    unsafe { &*named(v) }
}

pub(crate) fn named_place_mut<'l>(v: Word, _at: Settled<'l>) -> &'l mut Word {
    // SAFETY: as `named_place`'s, for a writable reference value made from an
    // exclusive borrow, whose provenance it carries.
    unsafe { &mut *named(v) }
}

/// The Rust function a closure value is.
pub(crate) fn callee<'a>(v: Word, _at: Settled<'a>) -> &'a Callee {
    // SAFETY: under `Settled`, `v` is a closure the call is lent, a slot
    // `closure` or `owned_closure` made from a shared function, live for
    // the call.
    unsafe { &*(*v.address::<Slot<Arc<Callee>>>()).value }
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
        slot_value(place, at)
    }
    fn exclusive<'a, T: Opaque + 'a>(_: &'a WordRuntime, place: &'a mut Word, at: Settled<'a>) -> &'a mut T {
        slot_value_mut(place, at)
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
        scalar_in(place, at)
    }
    fn exclusive<'a, P: Primitive>(_: &'a WordRuntime, place: &'a mut Word, at: Settled<'a>) -> &'a mut P {
        scalar_in_mut(place, at)
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
        slot_value(place, at)
    }
    fn exclusive<'a>(_: &'a WordRuntime, place: &'a mut Word, at: Settled<'a>) -> &'a mut String {
        slot_value_mut(place, at)
    }
}

impl OptionKind<WordRuntime> for Kinds {
    fn give(_: &WordRuntime, value: Option<Word>, _: Settled<'_>) -> Word {
        give_option(value)
    }
    fn take(_: &WordRuntime, v: Word, at: Settled<'_>) -> Option<Word> {
        take_option(v, at)
    }
}

impl ResultKind<WordRuntime> for Kinds {
    fn give(_: &WordRuntime, value: Result<Word, Word>, _: Settled<'_>) -> Word {
        give_result(value)
    }
    fn take(_: &WordRuntime, v: Word, at: Settled<'_>) -> Result<Word, Word> {
        take_result(v, at)
    }
}

impl TupleKind<WordRuntime> for Kinds {
    fn give<const N: usize>(_: &WordRuntime, parts: [Word; N], _: Settled<'_>) -> Word {
        give_tuple(parts)
    }
    fn take<const N: usize>(_: &WordRuntime, v: Word, at: Settled<'_>) -> [Word; N] {
        take_tuple(v, at)
    }
}

impl SequenceKind<WordRuntime> for Kinds {
    fn shared<'a>(_: &'a WordRuntime, place: &'a Word, at: Settled<'a>) -> &'a [Word] {
        elements(place, at)
    }
    fn exclusive<'a>(_: &'a WordRuntime, place: &'a mut Word, at: Settled<'a>) -> &'a mut [Word] {
        elements_mut(place, at)
    }
}

impl RefKind<WordRuntime> for Kinds {
    fn of<'a>(_: &'a WordRuntime, place: &'a Word, _: Settled<'a>) -> Word {
        reference(ptr::from_ref(place).cast_mut())
    }
    fn of_mut<'a>(_: &'a WordRuntime, place: &'a mut Word, _: Settled<'a>) -> Word {
        reference(ptr::from_mut(place))
    }
    fn place<'l>(_: &'l WordRuntime, v: Word, at: Settled<'l>) -> &'l Word {
        named_place(v, at)
    }
    fn place_mut<'l>(_: &'l WordRuntime, v: Word, at: Settled<'l>) -> &'l mut Word {
        named_place_mut(v, at)
    }
}

// SAFETY: `WordRuntime` views a value at its place in three ways: a scalar
// in the place's own word, a Rust value in the slot the place's word points
// at, and an array's elements in the run its slot holds. A place is a word
// the run keeps, a box's, or an element of an array: `Held` is boxed by the
// run, a box moves only as Rust moves it, which a borrow of it excludes, and
// an array's run is viewed only as a slice, so it never changes length or
// moves. A slot is allocated once by `Slot::allocate` and never moved; it is
// freed only by the release of the one holder that owns the value or by
// `take` moving the value out, and the checker's loan on a value excludes
// each of these while it lasts. `WordRuntime` writes none of them after
// making them except through an exclusive view, so each keeps its address
// and is written only under an exclusive loan until the checker's loan ends.
unsafe impl SignedS1 for WordRuntime {}
// SAFETY: `RefKind::of` and `RefKind::of_mut` keep the place's own address in
// the reference value's word and copy nothing of the value. By S1 the place
// stays at that address until the checker's loan ends, so the reference
// value names it.
unsafe impl SignedS2 for WordRuntime {}
// SAFETY: `WordRuntime` starts no run that outlives the call starting it. A
// closure is a Rust function, and calling one, returning or awaited, runs
// its glue to the end before the call returns or its future completes; an
// awaited call's future holds only the values given to it, which it
// releases when dropped, and reference values naming places its caller
// borrows for as long as the future lives (crossing §3). No work is left
// aloft when a call ends, so storage lent to a callee is read only within
// the lend, and the runtime keeps no value past the call that keeps it. It
// releases an allocation only when the value that owns it is released,
// moved out or handed over.
unsafe impl SignedS3 for WordRuntime {}

#[cfg(test)]
mod tests {
    use std::any::TypeId;
    use std::sync::Arc;

    use super::{
        Held, Kind, Slot, Word, array, free, named, read_scalar, reference, scalar, scalar_at, slot_at, slotted, text_at,
        unslotted,
    };

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
    fn a_reference_value_names_the_place_it_was_made_from() {
        let mut kept = slotted(String::from("named"));

        let v = reference(&raw mut kept);

        assert_eq!(v.kind(), Kind::Ref);
        assert!(std::ptr::eq(named(v), &raw mut kept));
        // SAFETY: `kept` holds a `String` in a slot, live and unwritten while
        // the text is read.
        assert_eq!(unsafe { text_at(named(v)) }, "named");
        // SAFETY: `kept` is live and released once.
        unsafe { free(kept) };
    }

    #[test]
    fn a_write_through_a_reference_to_a_scalar_lands_in_the_word_it_names() {
        let mut kept = scalar(5i16);
        let v = reference(&raw mut kept);

        // SAFETY: `v` names `kept`, a live `i16`, and nothing else uses `kept`
        // while the write lasts.
        unsafe { *scalar_at::<i16>(named(v)) = -300 };

        // SAFETY: `kept` holds an `i16`, read once.
        assert_eq!(unsafe { read_scalar::<i16>(kept) }, -300);
    }

    #[test]
    fn a_write_through_a_reference_to_a_slotted_value_lands_in_its_slot() {
        let mut kept = slotted(String::from("lent"));
        let v = reference(&raw mut kept);

        // SAFETY: as for the scalar, for a `String` in a slot.
        unsafe { (*slot_at::<String>(named(v))).push_str(" and written") };

        // SAFETY: `kept` holds a `String`, read once.
        assert_eq!(unsafe { unslotted::<String>(kept) }, "lent and written");
    }

    #[test]
    fn an_arrays_elements_are_the_places_of_its_run() {
        let mut run = array(vec![scalar(3i64), scalar(5i64), scalar(8i64)]);

        // SAFETY: `run` is an array, a `Vec<Held>` in a slot, and nothing else
        // uses it while its second element is written in place.
        unsafe {
            let elements = &mut *slot_at::<Vec<Held>>(&raw mut run);
            *scalar_at::<i64>(&raw mut elements[1].0) = -5;
        }

        // SAFETY: `run` holds a `Vec<Held>`, taken once; its elements are
        // `i64`s.
        let elements = unsafe { unslotted::<Vec<Held>>(run) };
        assert_eq!(elements.iter().map(|e| e.0.as_i64()).collect::<Vec<_>>(), [3, -5, 8]);
    }
}
