//! The only module that writes `unsafe`: the facts `WordRuntime` signs, the
//! casts between a word and what it holds, and the structural facts of the
//! test types.
//!
//! A value is a kind and a 64-bit word. A scalar lies in the word's first
//! bytes. Anything larger lies in an allocation the word points at: a Rust
//! value or a Rust function's glue in a `Slot` headed by its release and its
//! canonical type. A reference is a pointer in the word: `Ref` to a `Word` the
//! run keeps, `Part` to a part of lent storage, and `WidePart` to an
//! allocation holding the wide pointer of an unsized part.
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
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::callback::RustCallee;
use acvus_extern::done::Done;
use acvus_extern::runtime::{Crosses, RefTarget, Runtime};
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

    /// A value the runtime made and keeps, released by this holder's drop.
    pub(crate) fn kept(value: Word) -> Held {
        Held(value)
    }

    /// The Rust function this value is, made at the runtime `R`. The run
    /// calls a function value it holds as it passes arguments: trusting that
    /// the checker typed it a function, and that the host runs one runtime.
    pub(crate) fn rust_fn<R>(&self) -> &RustCallee<R>
    where
        R: Runtime,
    {
        assert_eq!(self.0.kind, Kind::Closure, "a call of a value that is no function");
        // SAFETY: a `Closure` word points at a live slot `closure` made from a
        // `RustCallee<R>` at the one runtime the run uses, and the slot lives
        // as long as this holder owns the value.
        unsafe { &(*self.0.address::<Slot<RustCallee<R>>>()).value }
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
pub(crate) fn closure<R>(f: RustCallee<R>) -> Word
where
    R: Runtime,
{
    Slot::allocate(Kind::Closure, f, TypeId::of::<RustCallee<R>>())
}

/// What a reference value `reference` made names, for the loan `'l` it was
/// made under.
///
/// # Safety
/// `v` is a live `Part` or `WidePart` made from a `P`, and the storage it
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
        kind => unreachable!("a {kind:?} value is no reference value"),
    }
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

/// The Rust function a closure value is, made at the runtime `R`.
pub(crate) fn callee<'a, R>(v: Word, _at: Settled<'a>) -> &'a RustCallee<R>
where
    R: Runtime,
{
    // SAFETY: under `Settled`, `v` is a closure the call is lent, a slot
    // `closure` made from a `RustCallee<R>` at the runtime the call runs at,
    // live for the call.
    unsafe { &(*v.address::<Slot<RustCallee<R>>>()).value }
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

    use acvus_extern::runtime::RefTarget;

    use super::{Kind, Slot, Word, erase, free, materialize, named_part, reference, target};

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

    #[test]
    fn a_reference_value_reads_back_the_thin_target_it_was_made_from() {
        let lent = String::from("thin");

        let v = reference(RefTarget::Shared(&lent));

        assert_eq!(v.kind(), Kind::Part);
        // SAFETY: `v` was made from `&lent`, which is live and shared.
        assert!(std::ptr::eq(unsafe { named_part::<String>(v) }, &lent));
    }

    #[test]
    fn a_reference_value_reads_back_the_text_and_run_it_was_made_from() {
        let text = String::from("a wide target");
        let run = [3i64, 5, 8];

        let text_v = reference(RefTarget::Shared(&text[2..6]));
        let run_v = reference(RefTarget::Shared(&run[1..]));

        assert_eq!((text_v.kind(), run_v.kind()), (Kind::WidePart, Kind::WidePart));
        // SAFETY: each value was made from a live shared borrow of its type,
        // and is released once after its read.
        unsafe {
            assert_eq!(&*named_part::<str>(text_v), "wide");
            assert_eq!(&*named_part::<[i64]>(run_v), &[5, 8]);
            free(text_v);
            free(run_v);
        }
    }

    #[test]
    fn a_write_through_a_writable_reference_value_lands_in_its_target() {
        let mut run = [3i64, 5, 8];

        let v = reference(RefTarget::Exclusive(&mut run[..2]));
        // SAFETY: `v` was made from the exclusive borrow above, which nothing
        // else uses until the write ends, and is released once.
        unsafe {
            (*named_part::<[i64]>(v))[1] = -5;
            free(v);
        }

        assert_eq!(run, [3, -5, 8]);
    }
}
