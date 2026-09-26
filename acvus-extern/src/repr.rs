//! The casts the extern contract makes (RFC-0080 rule 5): how an inline
//! scalar, a length read from bytes, a pointer and a run lie in the value
//! word, and the witnesses a cast between two types of one layout takes. The
//! contract serves every runtime, so this module holds no fact of one
//! runtime's frames or values; a runtime keeps its own casts in a private
//! module of its own. Outside the two, `repr_boundary` refuses a cast.
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

#![allow(
    clippy::disallowed_methods,
    reason = "the boundary module: a cast the workspace's lints refuse elsewhere is written here"
)]

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
    /// one per type `for_each_inline!` lists, and `decode` is callable only
    /// through `Encoded::get`.
    pub trait Sealed: Sized {
        fn decode(encoded: super::Encoded<Self>) -> Self;
    }
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
}

/// A `Word` type of which every `u64` reads as some value, so reading one
/// back needs no fact about where the word came from. `char` is the inline
/// type that is not one: its word is read through `Encoded<char>`.
pub trait TotalWord: Word {
    fn from_word(word: u64) -> Self;
}

macro_rules! total {
    ($t:ty, |$value:ident| $into:expr, |$word:ident| $from:expr) => {
        impl sealed::Sealed for $t {
            #[inline(always)]
            fn decode(encoded: Encoded<Self>) -> Self {
                <$t as TotalWord>::from_word(encoded.word())
            }
        }
        impl Word for $t {
            #[inline(always)]
            fn into_word(self) -> u64 {
                let $value = self;
                $into
            }
        }
        impl TotalWord for $t {
            #[inline(always)]
            fn from_word($word: u64) -> Self {
                $from
            }
        }
    };
}

macro_rules! signed {
    ($($t:ty),*) => { $(
        total!($t, |value| i64::from(value).cast_unsigned(), |word| word as $t);
    )* };
}

macro_rules! unsigned {
    ($($t:ty),*) => { $(
        total!($t, |value| u64::from(value), |word| word as $t);
    )* };
}

signed!(i8, i16, i32, i64);
unsigned!(u8, u16, u32, u64);
total!(f64, |value| value.to_bits(), |word| f64::from_bits(word));
total!(bool, |value| u64::from(value), |word| word != 0);
total!((), |_value| 0, |_word| ());

impl sealed::Sealed for char {
    #[inline(always)]
    fn decode(encoded: Encoded<Self>) -> Self {
        // SAFETY: `Encoded::of` is the one maker of an `Encoded<char>`, and
        // it wrote the scalar value zero-extended, so the low 32 bits are
        // that scalar value.
        unsafe { char::from_u32_unchecked(encoded.word() as u32) }
    }
}
impl Word for char {
    #[inline(always)]
    fn into_word(self) -> u64 {
        u64::from(u32::from(self))
    }
}

/// A word that is `into_word` of a `T` (RFC-0102): `of` is its one maker, so
/// reading it back as a `T`, or naming its first bytes as one, is safe at
/// every inline type, `char` and `bool` among them.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Encoded<T>(u64, PhantomData<fn() -> T>);

impl<T> Encoded<T>
where
    T: Word,
{
    #[inline(always)]
    pub fn of(value: T) -> Encoded<T> {
        Encoded(value.into_word(), PhantomData)
    }

    #[inline(always)]
    pub fn word(self) -> u64 {
        self.0
    }

    #[inline(always)]
    pub fn get(self) -> T {
        <T as sealed::Sealed>::decode(self)
    }

    /// The value, read where the word holds it.
    #[inline(always)]
    pub fn view(&self) -> &T {
        const {
            assert!(
                mem::size_of::<T>() <= mem::size_of::<u64>() && mem::align_of::<T>() <= mem::align_of::<u64>(),
                "an inline value's bytes fit its word's first bytes"
            )
        };
        // SAFETY: the word is `into_word` of a `T` (`of`), whose first bytes
        // are that `T`'s own on a little-endian target, the only kind this
        // module builds for; the assertion above fits them in the word.
        unsafe { &*std::ptr::from_ref(&self.0).cast::<T>() }
    }
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

    /// A word read back as the address it holds. Any word is some address;
    /// what reads through the pointer answers for it.
    #[inline(always)]
    pub const fn from_word(word: u64) -> PtrWord {
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

// -- Two types of one layout ------------------------------------------------

/// The witness that `A` and `B` are one layout (RFC-0080 rule 5): one size,
/// one alignment, and each byte of the one at the same offset under the same
/// validity as the other's, so that a place holding a valid `A` holds a
/// valid `B` and the reverse.
///
/// Only `same_layout!` makes one. Its constructor, private to acvus-extern,
/// checks the size and the alignment when the instantiation is compiled, and
/// the macro's caller answers for the rest with the fact that proves it — a
/// `repr(transparent)` wrapper and its field, a type and its `Canonical`
/// form, a derive's `Transparent` impl. Outside acvus-extern a witness is
/// only ever handed out: the field is private and the constructor is
/// `pub(crate)`.
///
/// The witness speaks of layout and of nothing else. What a type promises
/// beyond its bytes — who releases what it holds, the lifetimes it names, an
/// invariant its constructor keeps — is not in it, so each cast below is
/// `unsafe` and its caller names that part.
pub struct SameLayout<A, B>(PhantomData<(fn(A) -> A, fn(B) -> B)>);

impl<A, B> Clone for SameLayout<A, B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<A, B> Copy for SameLayout<A, B> {}

impl<A, B> SameLayout<A, B> {
    /// `same_layout!`'s. The size and the alignment are checked here, at
    /// each instantiation, so no witness names two types that differ in
    /// either.
    ///
    /// # Safety
    /// Each byte of `A` sits at the same offset under the same validity as
    /// `B`'s, so a place holding a valid `A` holds a valid `B` and the
    /// reverse.
    #[inline(always)]
    pub(crate) const unsafe fn vouched() -> Self {
        const {
            assert!(
                Layout::new::<A>().size() == Layout::new::<B>().size()
                    && Layout::new::<A>().align() == Layout::new::<B>().align(),
                "a witness names two types of one size and alignment"
            )
        };
        SameLayout(PhantomData)
    }

    /// `same_layout!`'s, where the fact that gives the rest of one layout is
    /// `F`'s: the witness where `F::HOLDS`, and `None` where not. The size
    /// and the alignment are checked at each instantiation where `F::HOLDS`.
    ///
    /// # Safety
    /// Where `F::HOLDS`, as `vouched`'s.
    #[inline(always)]
    pub(crate) const unsafe fn vouched_where<F>() -> Option<Self>
    where
        F: Fact,
    {
        const {
            assert!(
                !F::HOLDS
                    || Layout::new::<A>().size() == Layout::new::<B>().size()
                        && Layout::new::<A>().align() == Layout::new::<B>().align(),
                "a witness names two types of one size and alignment"
            )
        };
        match F::HOLDS {
            true => Some(SameLayout(PhantomData)),
            false => None,
        }
    }

    /// The same fact read from the other side.
    #[inline(always)]
    pub const fn flip(self) -> SameLayout<B, A> {
        SameLayout(PhantomData)
    }

    /// `a`'s bytes, moved whole, as a `B`.
    ///
    /// # Safety
    /// The witness gives a valid `B`. The caller holds the rest of what a
    /// `B` promises for these bytes, and dropping the `B` releases what
    /// dropping `a` would have.
    #[inline(always)]
    pub unsafe fn cast(self, a: A) -> B {
        let a = mem::ManuallyDrop::new(a);
        // SAFETY: the witness puts a valid `B` in `a`'s place, `ManuallyDrop`
        // keeps `a` from being released, and the caller's contract covers
        // the rest of `B`.
        unsafe { std::ptr::read((&raw const *a).cast::<B>()) }
    }

    /// The place `a` names, named as a `B` for as long.
    ///
    /// # Safety
    /// The witness gives a valid `B`. The caller holds the rest of what a
    /// `B` promises for these bytes for `'a`, the lifetimes `B` names among
    /// them.
    #[inline(always)]
    pub unsafe fn cast_ref<'a>(self, a: &'a A) -> &'a B {
        // SAFETY: the witness gives one size, one alignment and a valid `B`
        // in the place; the caller's contract covers the rest.
        unsafe { &*std::ptr::from_ref(a).cast::<B>() }
    }

    /// As `cast_ref`, exclusively.
    ///
    /// # Safety
    /// As `cast_ref`'s, and whatever is written through the `B` is, read as
    /// an `A` again once the loan ends, what an `A` promises.
    #[inline(always)]
    pub unsafe fn cast_mut<'a>(self, a: &'a mut A) -> &'a mut B {
        // SAFETY: as `cast_ref`'s; the witness runs both ways, and the
        // caller's contract covers what is written.
        unsafe { &mut *std::ptr::from_mut(a).cast::<B>() }
    }

    /// A `Vec` of `A`s as a `Vec` of `B`s over the same buffer: one layout
    /// per element is one layout per buffer, so the allocation is freed at
    /// the layout it was made at.
    ///
    /// # Safety
    /// As `cast`'s, for each element.
    #[inline(always)]
    pub unsafe fn cast_vec(self, items: Vec<A>) -> Vec<B> {
        let mut items = mem::ManuallyDrop::new(items);
        // SAFETY: an element of `B` has `A`'s size and alignment (the
        // witness), so the buffer holds `len` `B`s in an allocation of
        // `capacity` of them; `ManuallyDrop` keeps the `Vec<A>` from freeing
        // it, and the caller's contract covers each element.
        unsafe { Vec::from_raw_parts(items.as_mut_ptr().cast::<B>(), items.len(), items.capacity()) }
    }

    /// A run of `A`s named as a run of `B`s of the same length: one layout
    /// per element is one layout per run.
    ///
    /// # Safety
    /// As `cast_mut`'s, for each element.
    #[inline(always)]
    pub unsafe fn cast_slice_mut<'a>(self, run: &'a mut [A]) -> &'a mut [B] {
        let len = run.len();
        // SAFETY: an element of `B` has `A`'s size and alignment (the
        // witness), so `len` of them cover `run`'s bytes exactly; the
        // caller's contract covers each element.
        unsafe { std::slice::from_raw_parts_mut(run.as_mut_ptr().cast::<B>(), len) }
    }
}

/// A fact of a type that a constant answers, which `SameLayout::vouched_where`
/// reads. It is acvus-extern's: no other crate states one.
pub(crate) trait Fact {
    const HOLDS: bool;
}

// -- One type under two names -------------------------------------------------

/// The witness that `A` and `B` are one type, which only `same_type` makes,
/// from the two `TypeId`s.
pub struct SameType<A, B>(PhantomData<(fn(A) -> A, fn(B) -> B)>);

impl<A, B> Clone for SameType<A, B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<A, B> Copy for SameType<A, B> {}

impl<A, B> SameType<A, B> {
    /// `a` under its other name.
    #[inline(always)]
    pub fn cast(self, a: A) -> B {
        let a = mem::ManuallyDrop::new(a);
        // SAFETY: the witness says `A` is `B` (`same_type`), so the copy is
        // the value itself, and `ManuallyDrop` keeps the original from being
        // released.
        unsafe { mem::transmute_copy::<A, B>(&*a) }
    }
}

/// Whether `A` and `B` are one type, as the witness that casts between them.
/// The check is the two `TypeId`s, which the `'static` bounds make exact: two
/// types that differ in a lifetime alone are not both `'static`.
#[inline(always)]
pub fn same_type<A, B>() -> Option<SameType<A, B>>
where
    A: 'static,
    B: 'static,
{
    (TypeId::of::<A>() == TypeId::of::<B>()).then_some(SameType(PhantomData))
}

// -- A run as an array --------------------------------------------------------

/// A run of `N` elements named as the array it is: `[T; N]` is `N` `T`s in a
/// row with no other requirement.
///
/// # Safety
/// `run.len() == N`.
#[inline(always)]
pub unsafe fn array<T, const N: usize>(run: &[T]) -> &[T; N] {
    debug_assert_eq!(run.len(), N, "array: a run of {} is not [_; {N}]", run.len());
    // SAFETY: the caller's contract: the length is `N`, which is the only
    // way the conversion fails.
    unsafe { <&[T; N]>::try_from(run).unwrap_unchecked() }
}

/// As `array`, exclusively.
///
/// # Safety
/// `run.len() == N`.
#[inline(always)]
pub unsafe fn array_mut<T, const N: usize>(run: &mut [T]) -> &mut [T; N] {
    debug_assert_eq!(run.len(), N, "array_mut: a run of {} is not [_; {N}]", run.len());
    // SAFETY: as `array`'s.
    unsafe { <&mut [T; N]>::try_from(run).unwrap_unchecked() }
}

// -- A lifetime the runtime bounds --------------------------------------------

/// A type named at each lifetime `'a`. `At<'a>` and `At<'static>` are one
/// type expression read at two lifetimes, so they differ in the lifetime
/// alone: a generic associated type cannot pick another type by its
/// lifetime.
pub trait Lifetimed {
    type At<'a>: ?Sized + 'a;
}

/// A pointer to an `F::At<'a>`, with the lifetime erased: a runtime keeps it
/// where no lifetime can follow it, and bounds each use by a fact of its own.
///
/// # Safety
/// Each dereference of the result happens within `'a`. The caller names the
/// fact that bounds them, which the type no longer carries.
#[inline(always)]
pub unsafe fn unbounded<'a, F>(ptr: NonNull<F::At<'a>>) -> NonNull<F::At<'static>>
where
    F: Lifetimed + ?Sized,
{
    // SAFETY: `F::At<'a>` and `F::At<'static>` differ in the lifetime alone
    // (`Lifetimed`), so the two pointers have one layout and one metadata;
    // the caller's contract bounds the uses.
    unsafe { mem::transmute_copy::<NonNull<F::At<'a>>, NonNull<F::At<'static>>>(&ptr) }
}

/// A shared name of a place, at a lifetime its borrow does not give: the
/// place is reached through a copy of a word (a reference word read from a
/// local), and the lifetime of the storage the word names is the runtime's
/// fact, not the copy's.
///
/// # Safety
/// The place `place` names stays live, and no exclusive name of it is used,
/// for `'b`. The caller names the fact that bounds `'b`.
#[inline(always)]
pub unsafe fn unbounded_ref<'a, 'b, T>(place: &'a T) -> &'b T
where
    T: ?Sized,
{
    // SAFETY: the caller's contract: the place is live and shared for `'b`.
    unsafe { &*std::ptr::from_ref(place) }
}

/// As `unbounded_ref`, exclusively.
///
/// # Safety
/// The place `place` names stays live for `'b`, and the result is its only
/// name used meanwhile. The caller names the fact that bounds `'b`.
#[inline(always)]
pub unsafe fn unbounded_mut<'a, 'b, T>(place: &'a mut T) -> &'b mut T
where
    T: ?Sized,
{
    // SAFETY: the caller's contract: the place is live and exclusive for
    // `'b`.
    unsafe { &mut *std::ptr::from_mut(place) }
}

// -- A variant's tag in the word ----------------------------------------------

/// The word a variant's tag register holds for the tag `tag`: its bits
/// (`Astr::bits`), which carry the interner, so two tags are one word
/// exactly when they are one name.
#[inline(always)]
pub fn word_of_tag(tag: acvus_utils::Astr) -> u64 {
    tag.bits()
}

/// The tag a tag register's word names.
///
/// # Panics
/// `word` is no `word_of_tag`'s: its high half, which holds a non-zero
/// interner id in every tag's word, is zero.
#[inline(always)]
pub fn tag_of_word(word: u64) -> acvus_utils::Astr {
    acvus_utils::Astr::of_bits(word)
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
            assert_eq!(i8::from_word(v.into_word()), v);
            assert_eq!(Encoded::of(v).get(), v);
        }
        assert_eq!(Encoded::of('é').get(), 'é');
        assert_eq!(Encoded::of('\u{10FFFF}').word(), 0x10FFFF);
        assert!(is_inline::<char>());
        assert!(!is_inline::<String>());
    }

    #[test]
    fn every_word_reads_as_a_value_of_a_total_type() {
        for word in [0, 1, 2, 0x7F, 0x80, 0xFFFF_FFFF, 0xD800, u64::MAX] {
            assert_eq!(i8::from_word(word), word as i8);
            assert_eq!(u32::from_word(word), word as u32);
            assert_eq!(f64::from_word(word).to_bits(), word);
            assert_eq!(bool::from_word(word), word != 0);
            let () = <()>::from_word(word);
        }
    }

    #[test]
    fn an_encoded_word_is_viewed_as_its_value() {
        assert_eq!(*Encoded::of(-2i16).view(), -2);
        assert_eq!(*Encoded::of(u64::MAX).view(), u64::MAX);
        assert_eq!(*Encoded::of(1.5f64).view(), 1.5);
        assert_eq!(*Encoded::of('\u{10FFFF}').view(), '\u{10FFFF}');
        assert!(*Encoded::of(true).view());
        assert!(!*Encoded::of(false).view());
        let () = *Encoded::of(()).view();
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

        let again = PtrWord::from_word(word_of_ptr(&word).word());
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
    fn a_layout_witness_casts_there_and_back() {
        #[repr(transparent)]
        #[derive(Debug, PartialEq)]
        struct Name(String);

        // SAFETY: `Name` is `repr(transparent)` over `String`.
        let layout = unsafe { SameLayout::<String, Name>::vouched() };

        // SAFETY: a `Name` promises nothing a `String` does not, and
        // releases what the `String` would.
        let name = unsafe { layout.cast(String::from("héllo")) };
        assert_eq!(name, Name(String::from("héllo")));
        // SAFETY: as above, read back the other way.
        let text = unsafe { layout.flip().cast(name) };
        assert_eq!(text, "héllo");

        let mut text = text;
        // SAFETY: as above, for the loan.
        assert_eq!(unsafe { layout.cast_ref(&text) }.0, "héllo");
        // SAFETY: as above; what is written is a `String` again.
        unsafe { layout.cast_mut(&mut text) }.0.push('!');
        assert_eq!(text, "héllo!");

        let mut run = vec![Name(String::from("a")), Name(String::from("b"))];
        // SAFETY: as above, per element.
        let strings = unsafe { layout.flip().cast_slice_mut(&mut run) };
        assert_eq!(strings.len(), 2);
        strings[1].push('c');
        assert_eq!(run, [Name(String::from("a")), Name(String::from("bc"))]);

        // SAFETY: as above, per element; the buffer is the one `run` held.
        let strings = unsafe { layout.flip().cast_vec(run) };
        assert_eq!(strings, ["a", "bc"]);
    }

    #[test]
    fn a_witness_where_a_fact_holds_is_made_and_where_not_is_not() {
        #[repr(transparent)]
        struct Name(String);
        struct Holds;
        impl Fact for Holds {
            const HOLDS: bool = true;
        }
        struct Fails;
        impl Fact for Fails {
            const HOLDS: bool = false;
        }
        // SAFETY: `Name` is `repr(transparent)` over `String`.
        assert!(unsafe { SameLayout::<String, Name>::vouched_where::<Holds>() }.is_some());
        // SAFETY: the fact does not hold, so nothing is vouched; the size
        // check is not asked of `u8` and `String`.
        assert!(unsafe { SameLayout::<u8, String>::vouched_where::<Fails>() }.is_none());
    }

    #[test]
    fn a_type_identity_casts_only_between_one_type() {
        let same = same_type::<String, String>().expect("`String` is `String`");
        let text = same.cast(String::from("héllo"));
        assert_eq!(same.cast(text), "héllo");
        assert_eq!(same_type::<u64, u64>().map(|same| same.cast(7)), Some(7));

        assert!(same_type::<u64, i64>().is_none());
        assert!(same_type::<String, Vec<u8>>().is_none());
        assert!(same_type::<&'static str, String>().is_none());
    }

    #[test]
    fn an_unbounded_pointer_reads_the_place_it_was_made_from() {
        struct TextAt;
        impl Lifetimed for TextAt {
            type At<'a> = dyn AsRef<str> + 'a;
        }
        let mut text = String::from("héllo");
        let place: &mut (dyn AsRef<str> + '_) = &mut text;
        // SAFETY: `text` outlives every dereference below.
        let reach = unsafe { unbounded::<TextAt>(NonNull::from(place)) };
        // SAFETY: as above, and nothing else names `text` meanwhile.
        assert_eq!(unsafe { reach.as_ref() }.as_ref(), "héllo");
    }

    #[test]
    fn an_unbounded_reference_names_the_place_it_was_made_from() {
        let mut word = 7u64;
        let copy = std::ptr::from_mut(&mut word);
        // SAFETY: `word` outlives the result, and nothing else names it
        // while the result is used.
        let place: &mut u64 = unsafe { unbounded_mut(&mut *copy) };
        *place = 9;
        // SAFETY: as above, shared.
        let read: &u64 = unsafe { unbounded_ref(&*copy) };
        assert_eq!(*read, 9);
        assert_eq!(word, 9);
    }

    #[test]
    fn a_run_is_named_as_its_array() {
        let mut run = [1u64, 2, 3];
        // SAFETY: the run is 3 long.
        assert_eq!(unsafe { array::<u64, 3>(&run) }, &[1, 2, 3]);
        // SAFETY: as above.
        let array = unsafe { array_mut::<u64, 3>(&mut run) };
        array[2] = 9;
        assert_eq!(run, [1, 2, 9]);
    }

    #[test]
    fn a_tag_reads_back_from_its_word() {
        let interner = acvus_utils::Interner::new();
        let some = interner.intern("Some");
        let none = interner.intern("None");
        assert_eq!(tag_of_word(word_of_tag(some)), some);
        assert_ne!(word_of_tag(some), word_of_tag(none));
        assert_ne!(word_of_tag(some) >> 32, 0, "a tag's word carries its interner");
    }
}
