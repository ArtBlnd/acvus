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

// -- Whether a type can lie in the word -------------------------------------

/// Whether a value of `T` can be one a runtime keeps in its value word.
///
/// It is `false` only where `T`'s own layout rules the word out: `T` is
/// wider than 8 bytes, aligned to more than 8, or has drop glue. It is
/// `true` of every other type, and of every `Word` type among them, which
/// `obj::inline!` asserts of each type `for_each_inline!` lists — the list a
/// runtime's `is_inline` reads.
///
/// It reads the type and not how the type is written: a constant that asks
/// it of a type parameter, an alias or a macro type is evaluated at the type
/// they are filled with. A `true` of a type a runtime keeps behind the word
/// costs one re-encode that changes nothing; a `false` is never a type a
/// runtime keeps in it.
#[inline(always)]
pub const fn may_lie_in_the_word<T>() -> bool {
    mem::size_of::<T>() <= 8 && mem::align_of::<T>() <= 8 && !mem::needs_drop::<T>()
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

#[inline(always)]
fn word_of_len(len: usize) -> u64 {
    len as u64
}

// -- Pointers <-> the word ----------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct PtrWord(u64);

impl PtrWord {
    #[inline(always)]
    pub const fn word(self) -> u64 {
        self.0
    }

    /// # Safety
    /// `word` is `PtrWord::word` of some `PtrWord`.
    #[inline(always)]
    pub const unsafe fn from_word(word: u64) -> PtrWord {
        PtrWord(word)
    }
}

#[inline(always)]
pub fn word_of_ptr<T>(ptr: *const T) -> PtrWord
where
    T: ?Sized,
{
    PtrWord(ptr.expose_provenance() as u64)
}

#[inline(always)]
pub fn ptr_of_word<T>(word: PtrWord) -> *const T {
    debug_assert!(
        usize::try_from(word.0).is_ok(),
        "ptr_of_word: {:#x} is no address of this target",
        word.0
    );
    std::ptr::with_exposed_provenance(word.0 as usize)
}

// -- Function addresses -------------------------------------------------------

/// A function pointer kept by its bytes and not as an integer address: a
/// pointer made back from an integer carries no provenance.
#[derive(Clone, Copy)]
pub struct FnAddr(mem::MaybeUninit<fn()>);

impl fmt::Debug for FnAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FnAddr(..)")
    }
}

#[inline(always)]
pub fn fn_addr<F>(f: F) -> FnAddr
where
    F: Copy,
{
    const {
        assert!(
            mem::size_of::<F>() == mem::size_of::<fn()>(),
            "fn_addr: the type is not the width of a function pointer"
        )
    };
    // SAFETY: `F` is as wide as the destination (the assertion above), and a
    // `MaybeUninit` holds any bytes.
    FnAddr(unsafe { mem::transmute_copy::<F, mem::MaybeUninit<fn()>>(&f) })
}

/// # Safety
/// `addr` is `fn_addr::<F>(f)` for some `f` of this same `F`.
#[inline(always)]
pub unsafe fn fn_of<F>(addr: FnAddr) -> F
where
    F: Copy,
{
    const {
        assert!(
            mem::size_of::<F>() == mem::size_of::<fn()>(),
            "fn_of: the type is not the width of a function pointer"
        )
    };
    // SAFETY: the caller's contract: the bytes are an `F`'s, copied whole.
    unsafe { mem::transmute_copy::<mem::MaybeUninit<fn()>, F>(&addr.0) }
}

// -- A run as two words -------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Words {
    ptr: PtrWord,
    len: u64,
}

impl Words {
    #[inline(always)]
    pub fn of_slice<T>(run: &[T]) -> Words {
        Words {
            ptr: word_of_ptr(run.as_ptr()),
            len: word_of_len(run.len()),
        }
    }

    #[inline(always)]
    pub fn of_slice_mut<T>(run: &mut [T]) -> Words {
        Words {
            ptr: word_of_ptr(run.as_mut_ptr()),
            len: word_of_len(run.len()),
        }
    }

    #[inline(always)]
    pub fn of_str(text: &str) -> Words {
        Self::of_slice(text.as_bytes())
    }

    #[inline(always)]
    pub fn into_pair(self) -> [u64; 2] {
        [self.ptr.0, self.len]
    }

    /// # Safety
    /// `pair` is `into_pair` of some `Words`.
    #[inline(always)]
    pub unsafe fn from_pair(pair: [u64; 2]) -> Words {
        Words {
            ptr: PtrWord(pair[0]),
            len: pair[1],
        }
    }

    #[inline(always)]
    pub fn len(self) -> usize {
        self.len as usize
    }

    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    /// # Safety
    /// The maker was given a run of `len()` elements laid out as `T`s, and
    /// that run is live, and written through no other name, for `'a`.
    #[inline(always)]
    pub unsafe fn slice<'a, T>(self) -> &'a [T] {
        // SAFETY: the caller's contract.
        unsafe { std::slice::from_raw_parts(ptr_of_word::<T>(self.ptr), self.len()) }
    }

    /// # Safety
    /// As `slice`'s, the maker was `of_slice_mut`, and no other name of the
    /// run is used for `'a`.
    #[inline(always)]
    pub unsafe fn slice_mut<'a, T>(self) -> &'a mut [T] {
        // SAFETY: the caller's contract.
        unsafe { std::slice::from_raw_parts_mut(ptr_of_word::<T>(self.ptr).cast_mut(), self.len()) }
    }

    /// # Safety
    /// As `slice::<u8>`'s, and the bytes are UTF-8.
    #[inline(always)]
    pub unsafe fn str<'a>(self) -> &'a str {
        // SAFETY: the caller's contract.
        unsafe { std::str::from_utf8_unchecked(self.slice::<u8>()) }
    }
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
    fn only_a_layout_that_rules_the_word_out_answers_false() {
        macro_rules! in_the_word {
            ($($name:ident: $t:ty),*) => {
                $(assert!(may_lie_in_the_word::<$t>(), "{} is Inline", stringify!($t));)*
            };
        }
        crate::for_each_inline!(in_the_word);

        // A parameter, an alias of one and a macro type are answered at what
        // fills them.
        type Id<T> = T;
        macro_rules! same {
            ($t:ty) => {
                $t
            };
        }
        const fn through_a_parameter<T>() -> bool {
            may_lie_in_the_word::<Id<T>>()
        }
        assert!(through_a_parameter::<i8>());
        assert!(may_lie_in_the_word::<same!(i8)>());
        assert!(!through_a_parameter::<String>());

        // What the layout cannot rule out answers `true`, at the cost of a
        // re-encode that changes nothing.
        assert!(may_lie_in_the_word::<Option<i8>>());
        assert!(may_lie_in_the_word::<(i8,)>());

        assert!(!may_lie_in_the_word::<String>());
        assert!(!may_lie_in_the_word::<Vec<i8>>());
        assert!(!may_lie_in_the_word::<Box<i8>>());
        assert!(!may_lie_in_the_word::<[u64; 2]>());
        assert!(!may_lie_in_the_word::<std::vec::IntoIter<i64>>());
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
    fn a_pointer_reads_back_from_its_word() {
        fn round_trip<T>(ptr: *const T) {
            assert_eq!(ptr_of_word::<T>(word_of_ptr(ptr)), ptr);
        }
        let byte = 7u8;
        let word = 7u64;
        let boxed = Box::new([1u64, 2, 3]);
        let text = String::from("héllo");
        round_trip(&byte);
        round_trip(&word);
        round_trip(&*boxed);
        round_trip(text.as_ptr());
        round_trip(std::ptr::from_ref(&word).cast::<()>());
        let run: &[u64] = &boxed[1..];
        assert_eq!(
            ptr_of_word::<u64>(word_of_ptr(std::ptr::from_ref(run))),
            run.as_ptr(),
            "an unsized pointer's word is its address"
        );
        #[cfg(target_pointer_width = "64")]
        round_trip(std::ptr::without_provenance::<u8>(0xFEDC_BA98_7654_3210));

        // SAFETY: the word is `word_of_ptr`'s, stored bare.
        let again = unsafe { PtrWord::from_word(word_of_ptr(&word).word()) };
        // SAFETY: the pointer is `&word`'s, which is live.
        assert_eq!(unsafe { *ptr_of_word::<u64>(again) }, 7);
    }

    #[test]
    fn a_function_reads_back_from_its_address() {
        fn double(n: u64) -> u64 {
            n * 2
        }
        type Entry = for<'a> unsafe fn(&'a u64) -> u64;
        unsafe fn read(n: &u64) -> u64 {
            *n + 1
        }
        let addr = fn_addr::<fn(u64) -> u64>(double);
        // SAFETY: `addr` is `fn_addr` of a `fn(u64) -> u64`.
        let f: fn(u64) -> u64 = unsafe { fn_of(addr) };
        assert_eq!(f(21), 42);

        let addr = fn_addr::<Entry>(read);
        // SAFETY: as above, at `Entry`.
        let f: Entry = unsafe { fn_of(addr) };
        // SAFETY: `read` has no precondition.
        assert_eq!(unsafe { f(&41) }, 42);
    }

    #[test]
    fn a_run_reads_back_from_its_pair() {
        let values = vec![3u64, 4, 5];
        // SAFETY: the pair is `into_pair`'s.
        let words = unsafe { Words::from_pair(Words::of_slice(&values).into_pair()) };
        assert_eq!(words.len(), 3);
        // SAFETY: `values` is live and unwritten while the view is read.
        assert_eq!(unsafe { words.slice::<u64>() }, &values[..]);

        let mut values = values;
        let words = Words::of_slice_mut(&mut values);
        // SAFETY: `words` is `of_slice_mut`'s, and `values` is named only
        // through it until the view ends.
        let view = unsafe { words.slice_mut::<u64>() };
        view[0] = 9;
        assert_eq!(values, [9, 4, 5]);

        let text = String::from("héllo");
        let words = Words::of_str(&text);
        assert_eq!(words.len(), 6);
        // SAFETY: `text` is live UTF-8.
        assert_eq!(unsafe { words.str() }, "héllo");
        assert!(Words::of_slice::<u8>(&[]).is_empty());
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
