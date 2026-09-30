//! The only module that writes `unsafe`: the facts `WordRuntime` signs, and
//! the casts between a word and what it holds.
//!
//! A value is a kind and a 64-bit word. A scalar lies in the word's first
//! bytes. Anything larger lies in an allocation the word points at: a Rust
//! value in a `Slot` headed by its release and its canonical type. A
//! reference is a pointer in the word: `Ref` to a `Word` the run keeps, `Part`
//! to a part of lent storage, and `WidePart` to an allocation holding the wide
//! pointer of an unsized part.

#![allow(unsafe_code)]
#![allow(
    clippy::disallowed_methods,
    reason = "the boundary module: a pointer made from the address a word holds is written here"
)]

use std::any::TypeId;
use std::mem::MaybeUninit;
use std::ptr;

use acvus_extern::runtime::Crosses;
use acvus_extern::{Release, Releasing, SignedS1, SignedS2, SignedS3, Structural};

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
    fn allocate(value: T, canon: TypeId) -> Word {
        let header = Header { release: Slot::<T>::release, canon };
        Word::pointing(Kind::Large, Box::into_raw(Box::new(Slot { header, value })))
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
        return Slot::allocate(value, TypeId::of::<T::Canon>());
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

/// Where the `P` a lent value names lies: through a `Ref`, in the word or
/// the slot of the value it names; through a `Part`, at its address.
///
/// # Safety
/// `v` is a live `Ref` or `Part` whose target is live and holds a `P`.
unsafe fn target<P>(v: Word) -> *mut P {
    match v.kind {
        Kind::Ref => {
            let slot = v.address::<Word>();
            // SAFETY: a `Ref` names a live `Word`.
            let named = unsafe { slot.read() };
            if named.kind.is_scalar() {
                // SAFETY: as above; the scalar lies in the word's first bytes.
                unsafe { (&raw mut (*slot).word).cast::<P>() }
            } else {
                // SAFETY: the named value holds a `P`, so it is `Large` and
                // its word points at a live `Slot<P>`.
                unsafe { &raw mut (*named.address::<Slot<P>>()).value }
            }
        }
        Kind::Part => v.address::<P>(),
        kind => unreachable!("a {kind:?} value names no sized storage"),
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
        Kind::Large => {
            let header = v.address::<Header>();
            // SAFETY: a `Large` word points at a live slot's header, and the
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

// SAFETY: to be argued by the fill: an allocation a guard lends keeps its
// address and is written only under an exclusive guard until the loan ends.
unsafe impl SignedS1 for WordRuntime {}
// SAFETY: to be argued by the fill: a reference word names its target, or its
// written-back copy, until the loan ends.
unsafe impl SignedS2 for WordRuntime {}
// SAFETY: to be argued by the fill: an allocation is released only by the
// release of the one value that owns it, after every guard over it dropped.
unsafe impl SignedS3 for WordRuntime {}

#[cfg(test)]
mod tests {
    use std::any::TypeId;
    use std::sync::Arc;

    use super::{Kind, Slot, Word, erase, free, materialize, target};

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
        Slot::allocate(Arc::clone(alive), TypeId::of::<Arc<()>>())
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
        let lent = Word::pointing(Kind::Ref, &raw mut kept);

        // SAFETY: `lent` names `kept`, a live `i16`, and nothing else uses
        // `kept` while the write lasts.
        unsafe { *target::<i16>(lent) = -300 };

        // SAFETY: `kept` holds an `i16`, read once.
        assert_eq!(unsafe { materialize::<i16>(kept) }, -300);
    }

    #[test]
    fn a_write_through_a_lent_allocation_lands_in_its_slot() {
        let mut kept = erase(String::from("lent"));
        let lent = Word::pointing(Kind::Ref, &raw mut kept);

        // SAFETY: as for the scalar, for a `String` in a slot.
        unsafe { (*target::<String>(lent)).push_str(" and written") };

        // SAFETY: `kept` holds a `String`, read once.
        assert_eq!(unsafe { materialize::<String>(kept) }, "lent and written");
    }
}
