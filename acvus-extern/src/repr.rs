//! The representation of the runtime's value word: how an inline scalar,
//! a length read from bytes, and a record with a trailing array are laid
//! out in memory. The conversions live here and nowhere else, so a second
//! target is a change to this one file.
//!
//! This module speaks in words (`u64`), lengths and layouts. It names no
//! runtime value, so it offers no `T -> Value` or `Value -> T`: a runtime
//! builds its value from the word this module gives it.
//!
//! It is defined for little-endian targets whose pointers are 32 or 64 bits
//! wide, and a build for any other target stops here. A big-endian definition
//! that no build and no test exercises would be a convention, not a
//! definition; one arrives as a `cfg` arm in this file together with a
//! target that checks it.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "acvus: the value word is defined for little-endian targets only; `acvus_extern::repr` has no big-endian definition"
);

#[cfg(not(any(target_pointer_width = "32", target_pointer_width = "64")))]
compile_error!("acvus: a pointer must fit the value word and a u32 index must fit a usize");

use std::alloc::Layout;
use std::any::TypeId;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ptr::NonNull;

// -- Inline scalars <-> the word ----------------------------------------

mod sealed {
    /// Unnameable outside this module: `Word`'s impls are the ones below,
    /// one per type `for_each_inline!` lists.
    pub trait Sealed {}
}

/// The one encoding of an `Inline` value in the runtime's value word
/// (RFC-0037 rule 6): a signed integer sign-extended to 64 bits, an unsigned
/// one zero-extended, an `f64` its IEEE bits, a `char` its scalar value
/// zero-extended, a `bool` 0 or 1, and `()` 0.
///
/// Every word of an inline kind is `into_word` of its value, wherever the
/// value came from — a literal, an operation, or an extern's result — so
/// two equal values have one word and a reader compares words. A word
/// written through an in-place view (`view_mut`) holds that encoding again
/// only once `settle` re-encodes it, which is why the view is `unsafe`.
///
/// On a little-endian target the value's own bytes are the word's first
/// bytes under this encoding, which is what `view` reads.
pub trait Word: sealed::Sealed + Copy + Send + Sync + 'static {
    fn into_word(self) -> u64;

    /// # Safety
    /// `word` is `into_word` of some value of this type.
    unsafe fn from_word(word: u64) -> Self;
}

macro_rules! signed {
    ($($t:ty),*) => { $(
        impl sealed::Sealed for $t {}
        impl Word for $t {
            #[inline(always)]
            fn into_word(self) -> u64 {
                i64::from(self).cast_unsigned()
            }
            #[inline(always)]
            unsafe fn from_word(word: u64) -> Self {
                word as $t
            }
        }
    )* };
}

macro_rules! unsigned {
    ($($t:ty),*) => { $(
        impl sealed::Sealed for $t {}
        impl Word for $t {
            #[inline(always)]
            fn into_word(self) -> u64 {
                u64::from(self)
            }
            #[inline(always)]
            unsafe fn from_word(word: u64) -> Self {
                word as $t
            }
        }
    )* };
}

signed!(i8, i16, i32, i64);
unsigned!(u8, u16, u32, u64);

impl sealed::Sealed for f64 {}
impl Word for f64 {
    #[inline(always)]
    fn into_word(self) -> u64 {
        self.to_bits()
    }
    #[inline(always)]
    unsafe fn from_word(word: u64) -> Self {
        f64::from_bits(word)
    }
}

impl sealed::Sealed for char {}
impl Word for char {
    #[inline(always)]
    fn into_word(self) -> u64 {
        u64::from(u32::from(self))
    }
    #[inline(always)]
    unsafe fn from_word(word: u64) -> Self {
        // SAFETY: the caller's contract: the word is a scalar value
        // zero-extended, so its low 32 bits are that scalar value.
        unsafe { char::from_u32_unchecked(word as u32) }
    }
}

impl sealed::Sealed for bool {}
impl Word for bool {
    #[inline(always)]
    fn into_word(self) -> u64 {
        u64::from(self)
    }
    #[inline(always)]
    unsafe fn from_word(word: u64) -> Self {
        word != 0
    }
}

impl sealed::Sealed for () {}
impl Word for () {
    #[inline(always)]
    fn into_word(self) -> u64 {
        0
    }
    #[inline(always)]
    unsafe fn from_word(_: u64) -> Self {}
}

macro_rules! listed {
    ($($name:ident: $t:ty),*) => {
        /// Whether `T` is one of the `Inline` types, whose values live in the
        /// word itself.
        #[inline(always)]
        pub fn is_inline<T>() -> bool
        where
            T: 'static,
        {
            let id = TypeId::of::<T>();
            false $(|| id == TypeId::of::<$t>())*
        }
    };
}
crate::for_each_inline!(listed);

/// The value a word holds, read where it lies: `T`'s bytes are the word's
/// first bytes under `Word`'s encoding on a little-endian target.
///
/// # Safety
/// `T` is an `Inline` type (`is_inline`), and `word` is `into_word` of some
/// value of `T`.
#[inline(always)]
pub unsafe fn view<T>(word: &u64) -> &T
where
    T: 'static,
{
    debug_assert!(is_inline::<T>(), "view: {} is not Inline", std::any::type_name::<T>());
    // SAFETY: an `Inline` type is at most 8 bytes at alignment at most 8
    // (`obj::inline!`'s assertion), so its view fits the word; the caller's
    // contract puts a valid `T` in the first bytes.
    unsafe { &*std::ptr::from_ref(word).cast::<T>() }
}

/// As `view`, exclusively: a write through it lands in the word's first
/// bytes and leaves the rest as they were.
///
/// # Safety
/// As `view`'s, and once the view ends the word's holder re-encodes it with
/// `settle::<T>` before anything reads the word as a whole: a signed
/// integer written through the view is not sign-extended until then.
#[inline(always)]
pub unsafe fn view_mut<T>(word: &mut u64) -> &mut T
where
    T: 'static,
{
    debug_assert!(is_inline::<T>(), "view_mut: {} is not Inline", std::any::type_name::<T>());
    // SAFETY: as `view`'s, exclusively.
    unsafe { &mut *std::ptr::from_mut(word).cast::<T>() }
}

/// Re-encode a word an in-place view of `T` may have written: the value in
/// its first bytes, encoded whole again. On a word `into_word` wrote it
/// changes nothing.
///
/// # Safety
/// The word's first `size_of::<T>()` bytes hold a valid `T`: it was
/// `into_word` of a `T`, written since only through `view_mut::<T>`.
#[inline(always)]
pub unsafe fn settle<T>(word: &mut u64)
where
    T: Word,
{
    // SAFETY: the caller's contract, and `T` is `Inline`: `Word`'s impls are
    // `for_each_inline!`'s types.
    let value = unsafe { *view::<T>(word) };
    *word = value.into_word();
}

// -- Lengths --------------------------------------------------------------

/// A length or an index read as a `u64` that this target's `usize` cannot
/// hold: only on a 32-bit target, from bytes a 64-bit one wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TooWide(pub u64);

impl fmt::Display for TooWide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} does not fit a usize of {} bits",
            self.0,
            usize::BITS
        )
    }
}

impl std::error::Error for TooWide {}

/// A `u64` length or index from untrusted bytes, as a `usize`, refused
/// where it does not fit rather than truncated.
#[inline]
pub fn len_of_word(word: u64) -> Result<usize, TooWide> {
    usize::try_from(word).map_err(|_| TooWide(word))
}

// -- A head with a trailing array -------------------------------------------

/// The layout of one allocation that holds an `H` and then a run of `E`s:
/// `Layout::extend`'s, so the run starts at `E`'s own alignment whatever
/// `H`'s size and alignment are on the target. The run's length is a `u16`,
/// and the widest such record is a valid layout (the assertion in `TAIL`).
pub struct HeadAndTail<H, E>(PhantomData<fn() -> (H, E)>);

impl<H, E> HeadAndTail<H, E> {
    /// The byte offset of the first `E`: `H`'s size rounded up to `E`'s
    /// alignment, which is what `Layout::extend` gives.
    pub const TAIL: usize = {
        let Ok((_, tail)) = Layout::new::<H>().extend(Layout::new::<E>()) else {
            panic!("a head and one element of its tail are a valid layout")
        };
        assert!(
            tail + (u16::MAX as usize) * mem::size_of::<E>() + mem::align_of::<H>() + mem::align_of::<E>()
                < isize::MAX as usize,
            "a record of the widest tail a u16 can count is a valid layout"
        );
        tail
    };

    /// The layout of a record whose tail is `len` elements, padded to its
    /// alignment.
    #[inline]
    pub fn layout(len: u16) -> Layout {
        let head = Layout::new::<H>();
        let tail = Layout::new::<E>();
        let align = if head.align() > tail.align() { head.align() } else { tail.align() };
        let size = Self::TAIL + usize::from(len) * tail.size();
        // SAFETY: `align` is one of two layouts' alignments, a power of two,
        // and `TAIL`'s assertion bounds `size` rounded up to it below
        // `isize::MAX`.
        unsafe { Layout::from_size_align_unchecked(size, align) }.pad_to_align()
    }

    /// The first element of the tail of the record `head` begins.
    ///
    /// # Safety
    /// `head` is the start of an allocation of `layout(len)` for some `len`.
    #[inline(always)]
    pub unsafe fn tail(head: NonNull<H>) -> NonNull<E> {
        // SAFETY: the caller's contract: the allocation reaches `TAIL`.
        unsafe { head.byte_add(Self::TAIL) }.cast::<E>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_signed_width_is_sign_extended_and_an_unsigned_one_zero_extended() {
        assert_eq!((-1i8).into_word(), u64::MAX);
        assert_eq!((-1i16).into_word(), u64::MAX);
        assert_eq!((-1i32).into_word(), u64::MAX);
        assert_eq!((-1i64).into_word(), u64::MAX);
        assert_eq!(i8::MIN.into_word(), (i64::from(i8::MIN)).cast_unsigned());
        assert_eq!(u8::MAX.into_word(), 0xFF);
        assert_eq!(u16::MAX.into_word(), 0xFFFF);
        assert_eq!(u32::MAX.into_word(), 0xFFFF_FFFF);
        assert_eq!(true.into_word(), 1);
        assert_eq!('\u{10FFFF}'.into_word(), 0x10FFFF);
        assert_eq!(().into_word(), 0);
    }

    #[test]
    fn a_word_reads_back_as_its_value() {
        for v in [i8::MIN, -1, 0, 1, i8::MAX] {
            // SAFETY: the word is `into_word` of an `i8`.
            assert_eq!(unsafe { i8::from_word(v.into_word()) }, v);
        }
        // SAFETY: as above, at `char`.
        assert_eq!(unsafe { char::from_word('é'.into_word()) }, 'é');
        assert!(is_inline::<char>());
        assert!(!is_inline::<String>());
    }

    #[test]
    fn a_view_reads_the_value_and_settle_re_encodes_a_write() {
        let mut word = 5i8.into_word();
        // SAFETY: the word is `into_word` of an `i8`.
        assert_eq!(unsafe { *view::<i8>(&word) }, 5);
        // SAFETY: as above; the word is settled below before it is read.
        *unsafe { view_mut::<i8>(&mut word) } = -1;
        assert_eq!(word, 0xFF, "the view wrote the first byte alone");
        // SAFETY: the first byte holds the `i8` the view wrote.
        unsafe { settle::<i8>(&mut word) };
        assert_eq!(word, (-1i8).into_word());

        let mut word = (-1i16).into_word();
        // SAFETY: as above, at `i16`.
        *unsafe { view_mut::<i16>(&mut word) } = 7;
        // SAFETY: as above.
        unsafe { settle::<i16>(&mut word) };
        assert_eq!(word, 7i16.into_word());
    }

    #[test]
    fn a_length_that_does_not_fit_is_refused() {
        assert_eq!(len_of_word(7), Ok(7));
        if usize::BITS < 64 {
            assert_eq!(len_of_word(u64::MAX), Err(TooWide(u64::MAX)));
        } else {
            assert_eq!(len_of_word(u64::MAX), Ok(usize::MAX));
        }
    }

    #[test]
    fn a_tail_starts_at_its_own_alignment() {
        #[repr(C)]
        struct Head {
            _a: u32,
            _b: u32,
            _c: u16,
        }
        assert_eq!(HeadAndTail::<Head, u64>::TAIL, 16);
        assert_eq!(HeadAndTail::<Head, u64>::layout(2).size(), 32);
        assert_eq!(HeadAndTail::<Head, u64>::layout(2).align(), 8);
        assert_eq!(HeadAndTail::<u8, u8>::TAIL, 1);
    }
}
