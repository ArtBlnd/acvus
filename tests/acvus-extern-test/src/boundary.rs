//! The only module that writes `unsafe`: the facts `WordRuntime` signs, the
//! casts between a word and what it holds, and the structural facts of the
//! test types.
//!
//! A value is a kind and a 64-bit word. A scalar lies in the word's first
//! bytes. Anything larger lies in an allocation the word points at: a Rust
//! value or a Rust function's glue in a `Slot` headed by its release and its
//! canonical type, or an aggregate's body, whose parts are each a `Word` in an
//! allocation of its own. A reference is a pointer in the word: `Ref` to a
//! `Word` the run keeps or an aggregate keeps as a part, `Part` to a part of
//! lent storage, and `WidePart` to an allocation holding the wide pointer of
//! an unsized part.
//!
//! Every read at a type takes the entry's `Settled`, the proof that the value
//! is live at the type the checker settled for its site (safety §3). No read
//! checks the kind against the type; only `Validate` compares them, for the
//! adapter.

#![allow(unsafe_code)]
#![allow(
    clippy::disallowed_methods,
    reason = "the boundary module: a pointer made from the address a word holds is written here"
)]

use std::any::TypeId;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ptr::{self, NonNull};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::callback::RustFn;
use acvus_extern::done::Done;
use acvus_extern::runtime::{Crosses, ObjectBody, RefTarget, Runtime, TupleBody, VariantBody};
use acvus_extern::{
    LayoutFact, Release, Releasing, Settled, SignedS1, SignedS2, SignedS3, Structural, StructuralFacts,
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
    Tuple,
    Object,
    Variant,
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

/// The scalar kind of `T`, where `T`'s canonical form is a scalar.
pub(crate) fn scalar_kind<T>() -> Option<Kind>
where
    T: Structural,
{
    let canon = TypeId::of::<T::Canon>();
    scalars().into_iter().find(|&(id, _)| id == canon).map(|(_, kind)| kind)
}

/// A value: its kind and its word, as the interpreter's `Value` on master.
#[derive(Debug, Clone, Copy)]
pub struct Word {
    kind: Kind,
    word: u64,
}

impl Word {
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

/// The value of `value`: a scalar in the word, anything else in a slot.
pub(crate) fn erase<T>(value: T) -> Word
where
    T: Crosses,
{
    let Some(kind) = scalar_kind::<T>() else {
        return Slot::allocate(Kind::Large, value, TypeId::of::<T::Canon>());
    };
    let mut word = 0u64;
    // SAFETY: `T`'s canonical form is the scalar `kind` names, and `T` has
    // its canonical form's size and alignment by `T`'s `Structural`
    // signature; every scalar is at most 8 bytes aligned to at most 8, so a
    // `T` fits the word's first bytes at the word's alignment.
    unsafe { ptr::from_mut(&mut word).cast::<T>().write(value) };
    Word { kind, word }
}

/// # Safety
/// `v` is a live value made from a value of `T` at `T`'s canonical type,
/// and nothing uses it afterwards.
unsafe fn materialize<T>(v: Word) -> T
where
    T: Crosses,
{
    match scalar_kind::<T>() {
        // SAFETY: `erase` wrote a `T` in the word's first bytes.
        Some(_) => unsafe { ptr::from_ref(&v.word).cast::<T>().read() },
        // SAFETY: `erase` allocated a `Slot` of a type with `T`'s layout and
        // release by `T`'s `Structural` signature, and the caller's contract
        // makes this its one use.
        None => unsafe { Slot::<T>::take(v.address()) },
    }
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
        // SAFETY: a body's word points at the body `make_*` boxed.
        Kind::Tuple => drop(unsafe { Box::from_raw(v.address::<WordTuple>()) }),
        Kind::Object => drop(unsafe { Box::from_raw(v.address::<WordObject>()) }),
        Kind::Variant => drop(unsafe { Box::from_raw(v.address::<WordVariant>()) }),
        // SAFETY: `reference` boxed the wide pointer.
        Kind::WidePart => drop(unsafe { Box::from_raw(v.address::<WideBits>()) }),
        Kind::Ref
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
/// for a program's variable.
pub struct Held(Word);

impl Held {
    /// The result of a call, which the runtime owns from here on.
    pub fn of_the_result<R>(done: Done<R>) -> Held
    where
        R: Runtime<Value = Word>,
    {
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

pub(crate) fn take<T>(v: Word, _at: Settled<'_>) -> T
where
    T: Crosses,
{
    // SAFETY: under `Settled`, `v` is a value of `T` its one holder moves out.
    unsafe { materialize(v) }
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

/// A `String`'s text: a `String` lent whole, or a reference to text.
pub(crate) fn text<'a>(v: &'a Word, at: Settled<'a>) -> &'a str {
    match v.kind {
        // SAFETY: under `Settled`, a `WidePart` read as text was made from a
        // `str` the call lent, live for `'a`.
        Kind::WidePart => unsafe { &*wide_target::<str>(*v) },
        _ => place::<String>(v, at),
    }
}

/// A program's array lent as a slice: the array whole, its `Vec` in a slot,
/// or a reference to a run of it.
pub(crate) fn run<'a, T>(v: &'a Word, at: Settled<'a>) -> &'a [T] {
    match v.kind {
        // SAFETY: as `text`'s, for a run of `T`.
        Kind::WidePart => unsafe { &*wide_target::<[T]>(*v) },
        _ => place::<Vec<T>>(v, at),
    }
}

pub(crate) fn run_mut<'a, T>(v: &'a mut Word, at: Settled<'a>) -> &'a mut [T] {
    match v.kind {
        // SAFETY: as `run`'s, exclusively lent.
        Kind::WidePart => unsafe { &mut *wide_target::<[T]>(*v) },
        _ => place_mut::<Vec<T>>(v, at),
    }
}

/// A reference value to a part of storage the call lent. It names the part
/// where it lies; a writable one carries the exclusive borrow's provenance.
pub(crate) fn reference<P>(target: RefTarget<'_, P>) -> Word
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
    Word::pointing(Kind::WidePart, bits)
}

/// A script function value made from a Rust function's glue.
pub(crate) fn closure<R>(f: RustFn<R>) -> Word
where
    R: Runtime,
{
    Slot::allocate(Kind::Closure, f, TypeId::of::<RustFn<R>>())
}

// Aggregates.

/// An aggregate's parts, each a `Word` in an allocation of its own, so that
/// a part lent in place is named without borrowing the body.
struct Parts(Box<[NonNull<Word>]>);

// SAFETY: `Parts` owns each allocation it points at, as a `Box<Word>` would,
// and `Word` is `Send` and `Sync`.
unsafe impl Send for Parts {}
// SAFETY: as for `Send`.
unsafe impl Sync for Parts {}

impl Parts {
    /// # Safety
    /// The caller hands over `words`, which it owns: nothing releases them
    /// but these parts.
    unsafe fn adopt(words: impl Iterator<Item = Word>) -> Parts {
        Parts(words.map(|w| NonNull::from(Box::leak(Box::new(w)))).collect())
    }

    /// Every part, handed over at once; the parts own nothing afterwards.
    fn hand_over(mut self) -> std::vec::IntoIter<Word> {
        let parts = std::mem::take(&mut self.0);
        parts
            .iter()
            // SAFETY: `adopt` leaked each box, and taking the parts out of
            // `self` makes this each one's one reclaim.
            .map(|part| *unsafe { Box::from_raw(part.as_ptr()) })
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn lent(&self, i: u32) -> Word {
        Word::pointing(Kind::Ref, self.0[i as usize].as_ptr())
    }
}

impl Drop for Parts {
    fn drop(&mut self) {
        for part in self.0.iter() {
            // SAFETY: the parts own each value they still hold: `adopt` was
            // handed it and `hand_over` has not handed it out.
            unsafe { free(*Box::from_raw(part.as_ptr())) };
        }
    }
}

pub struct WordTuple(Parts);

pub struct WordObject(Parts);

pub struct WordVariant {
    tag: u32,
    payload: Parts,
}

impl TupleBody<Word> for WordTuple {
    fn into_parts(self) -> impl ExactSizeIterator<Item = Word> {
        self.0.hand_over()
    }
}

impl ObjectBody<Word> for WordObject {
    fn into_fields(self) -> impl ExactSizeIterator<Item = Word> {
        self.0.hand_over()
    }
}

impl VariantBody<Word> for WordVariant {
    fn into_payload(self) -> (u32, Option<Word>) {
        (self.tag, self.payload.hand_over().next())
    }
}

pub(crate) fn make_tuple(parts: impl ExactSizeIterator<Item = Word>, _at: Settled<'_>) -> Word {
    // SAFETY: under `Settled`, the crossing hands the parts' values over
    // (runtime contract §2, Aggregates).
    let parts = unsafe { Parts::adopt(parts) };
    Word::pointing(Kind::Tuple, Box::into_raw(Box::new(WordTuple(parts))))
}

pub(crate) fn make_object(fields: impl ExactSizeIterator<Item = Word>, _at: Settled<'_>) -> Word {
    // SAFETY: as `make_tuple`'s.
    let fields = unsafe { Parts::adopt(fields) };
    Word::pointing(Kind::Object, Box::into_raw(Box::new(WordObject(fields))))
}

pub(crate) fn make_variant(tag: u32, payload: Option<Word>, _at: Settled<'_>) -> Word {
    // SAFETY: as `make_tuple`'s.
    let payload = unsafe { Parts::adopt(payload.into_iter()) };
    Word::pointing(Kind::Variant, Box::into_raw(Box::new(WordVariant { tag, payload })))
}

/// # Safety
/// `v` is a live body of kind `kind` boxed as a `B`, and nothing uses it
/// afterwards.
unsafe fn unbox<B>(v: Word, kind: Kind) -> B {
    debug_assert_eq!(v.kind, kind);
    // SAFETY: the caller's contract.
    *unsafe { Box::from_raw(v.address::<B>()) }
}

pub(crate) fn take_tuple(v: Word, _at: Settled<'_>) -> WordTuple {
    // SAFETY: under `Settled`, `v` is a tuple its one holder takes apart.
    unsafe { unbox(v, Kind::Tuple) }
}

pub(crate) fn take_object(v: Word, _at: Settled<'_>) -> WordObject {
    // SAFETY: as `take_tuple`'s, for an object.
    unsafe { unbox(v, Kind::Object) }
}

pub(crate) fn take_variant(v: Word, _at: Settled<'_>) -> WordVariant {
    // SAFETY: as `take_tuple`'s, for a variant.
    unsafe { unbox(v, Kind::Variant) }
}

/// The aggregate a lent aggregate names.
///
/// # Safety
/// `v` is a live `Ref` to a live aggregate.
unsafe fn named(v: Word) -> Word {
    debug_assert_eq!(v.kind, Kind::Ref);
    // SAFETY: the caller's contract.
    unsafe { v.address::<Word>().read() }
}

pub(crate) fn lent_part(v: Word, i: u32, _at: Settled<'_>) -> Word {
    // SAFETY: under `Settled`, `v` is a lent aggregate, a `Ref` to it, and
    // the aggregate's body is live for the call. The body is only read here.
    unsafe {
        let aggregate = named(v);
        match aggregate.kind {
            Kind::Tuple => (*aggregate.address::<WordTuple>()).0.lent(i),
            Kind::Object => (*aggregate.address::<WordObject>()).0.lent(i),
            Kind::Variant => (*aggregate.address::<WordVariant>()).payload.lent(i),
            kind => unreachable!("a {kind:?} value has no parts"),
        }
    }
}

pub(crate) fn lent_tag(v: Word, _at: Settled<'_>) -> u32 {
    // SAFETY: as `lent_part`'s, for a variant.
    unsafe { (*named(v).address::<WordVariant>()).tag }
}

// What `Validate` reads.

/// The value a value names: through a `Ref`, the value it names; any other
/// value is itself.
pub(crate) fn followed(v: Word, _at: Settled<'_>) -> Word {
    match v.kind {
        // SAFETY: under `Settled`, `v` is live, and a live `Ref` names a live
        // `Word`.
        Kind::Ref => unsafe { v.address::<Word>().read() },
        _ => v,
    }
}

/// The canonical type a slot holds.
pub(crate) fn canon_of(v: Word, _at: Settled<'_>) -> TypeId {
    assert!(
        matches!(v.kind, Kind::Large | Kind::Closure),
        "only a value in a slot has a canonical type"
    );
    // SAFETY: under `Settled`, `v` is live, so its word points at a live
    // slot's header.
    unsafe { (*v.address::<Header>()).canon }
}

// SAFETY: `WordRuntime` lends four kinds of storage. A scalar is lent in the
// word at the place the guard is given, the holder's own: an argument run the
// run keeps for the call, a `Uniform`, or a variable. A value's slot is
// allocated once by `Slot::allocate` and never moved; it is freed only by the
// release of the one holder that owns the value, by `take` moving the value
// out, or by the drop of the body that holds it, and the checker's loan on a
// value excludes each of these while it lasts. An aggregate's part lies in an
// allocation of its own that `Parts::adopt` made and only `Parts::hand_over`
// or the body's drop reclaims, likewise excluded while the loan lasts. A word
// the run keeps stays where the run put it: `Held` is boxed by the run, and
// this runtime moves no place it is given. `WordRuntime` writes none of them
// after making them except through the `&mut` an exclusive guard is, so each
// keeps its address and is written only under an exclusive loan until the
// checker's loan ends.
unsafe impl SignedS1 for WordRuntime {}
// SAFETY: `reference` keeps the part's own address, in the word or boxed with
// its metadata, and makes no copy of the part. By S1 the part stays at that
// address until the checker's loan ends, so the reference names it.
unsafe impl SignedS2 for WordRuntime {}
// SAFETY: `WordRuntime` starts no other run: it implements no `CallsBack`,
// so it never calls a closure, and it keeps no value of its own. It lends
// storage in place only to the call it is lent for, and it releases an
// allocation only when the value that owns it is released, moved out or
// handed over. So no storage it lends to another run and no value it keeps
// exists to release early. This argument must be made again with
// `CallsBack`.
unsafe impl SignedS3 for WordRuntime {}

// The test types' structural facts.

/// A value that counts its drops, for the crossing's release tests.
pub struct Tally(pub Arc<AtomicUsize>);

impl Drop for Tally {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// An extern-defined struct, for borrowed results of a field and of an
/// element the struct owns.
pub struct Pair {
    pub name: String,
    pub items: Vec<i64>,
}

// SAFETY: `Canon` is `Self`, so size, alignment, field offsets and release
// are one. `Tally` has no lifetime or type parameter, so it reaches none and
// has no interior mutability over one, as `FACTS` states with an empty
// generic list.
unsafe impl Structural for Tally {
    type Canon = Tally;

    const FACTS: StructuralFacts = StructuralFacts { layout: LayoutFact::OwnCanonical, generics: &[] };
}

// SAFETY: as for `Tally`.
unsafe impl Structural for Pair {
    type Canon = Pair;

    const FACTS: StructuralFacts = StructuralFacts { layout: LayoutFact::OwnCanonical, generics: &[] };
}

#[cfg(test)]
mod tests {
    use std::any::TypeId;
    use std::sync::Arc;

    use super::{Kind, Parts, Slot, Word, erase, free, materialize, target};

    #[test]
    fn a_scalar_round_trips_through_the_word() {
        let words = [
            erase(-7i8),
            erase(u64::MAX),
            erase(1.5f64),
            erase('\u{10FFFF}'),
            erase(true),
        ];

        assert_eq!(words.map(Word::kind), [Kind::I8, Kind::U64, Kind::F64, Kind::Char, Kind::Bool]);
        // SAFETY: each word was erased from the type it is read at, once.
        unsafe {
            assert_eq!(materialize::<i8>(words[0]), -7);
            assert_eq!(materialize::<u64>(words[1]), u64::MAX);
            assert_eq!(materialize::<f64>(words[2]), 1.5);
            assert_eq!(materialize::<char>(words[3]), '\u{10FFFF}');
            assert!(materialize::<bool>(words[4]));
        }
    }

    #[test]
    fn a_value_larger_than_a_scalar_round_trips_through_an_allocation() {
        let word = erase(String::from("large"));

        assert_eq!(word.kind(), Kind::Large);
        // SAFETY: the word was erased from a `String`, and is read once.
        assert_eq!(unsafe { materialize::<String>(word) }, "large");
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
    fn parts_handed_over_are_released_by_their_new_owner_only() {
        let alive = Arc::new(());
        // SAFETY: the two values are fresh and handed over.
        let parts = unsafe { Parts::adopt([counted(&alive), counted(&alive)].into_iter()) };

        let handed: Vec<Word> = parts.hand_over().collect();
        assert_eq!(Arc::strong_count(&alive), 3);
        for part in handed {
            // SAFETY: each part handed out is live and released once.
            unsafe { free(part) };
        }

        assert_eq!(Arc::strong_count(&alive), 1);
    }

    #[test]
    fn parts_never_handed_over_are_released_with_their_body() {
        let alive = Arc::new(());
        // SAFETY: the two values are fresh and handed over.
        let parts = unsafe { Parts::adopt([counted(&alive), counted(&alive)].into_iter()) };

        drop(parts);

        assert_eq!(Arc::strong_count(&alive), 1);
    }

    #[test]
    fn a_write_through_a_lent_scalar_lands_in_the_word_it_names() {
        let mut kept = erase(5i16);
        let mut lent = Word::pointing(Kind::Ref, &raw mut kept);

        // SAFETY: `lent` names `kept`, a live `i16`, and nothing else uses
        // `kept` while the write lasts.
        unsafe { *target::<i16>(&raw mut lent) = -300 };

        // SAFETY: `kept` holds an `i16`, read once.
        assert_eq!(unsafe { materialize::<i16>(kept) }, -300);
    }

    #[test]
    fn a_write_through_a_lent_allocation_lands_in_its_slot() {
        let mut kept = erase(String::from("lent"));
        let mut lent = Word::pointing(Kind::Ref, &raw mut kept);

        // SAFETY: as for the scalar, for a `String` in a slot.
        unsafe { (*target::<String>(&raw mut lent)).push_str(" and written") };

        // SAFETY: `kept` holds a `String`, read once.
        assert_eq!(unsafe { materialize::<String>(kept) }, "lent and written");
    }
}
