//! `Slice<T, Rt>` and `SliceMut<T, Rt>`: the acvus types `&[T]` and
//! `&mut [T]`, the one thing the machine indexes (RFC-0047).
//!
//! A slice is a borrow of the container it was taken from: it holds that
//! container's loan and is never stored beyond it (RFC-0018). Every
//! sliceable container stores the runtime's own values, so the element
//! width is the runtime's `Value` and nothing else (RFC-0047 §1); an
//! `Elements` is therefore the only payload either type carries, and the
//! machine reads it without knowing any container's layout.

use std::marker::PhantomData;

use acvus_mir::ty::{Mutability, PolyTy, TypeArg};
use acvus_utils::Interner;

use crate::obj::Cross;
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg, TyVar};

/// A run as the machine holds it: one word per register of the pair.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Words {
    pub ptr: u64,
    pub len: u64,
}

/// A run of the runtime's values in a storage. Both fields are private and
/// `of` takes a Rust slice, so a pointer and a length that did not come
/// from one run cannot be named.
pub struct Elements<Rt>
where
    Rt: Runtime,
{
    ptr: *const Rt::Value,
    len: usize,
}

// SAFETY: an `Elements` is a borrow of a storage the checker keeps alive
// for as long as the slice value exists (RFC-0018), and the storage holds
// `Rt::Value`s, which are `Send + Sync` by the `Runtime` contract. The raw
// pointer carries no capability the `&[Rt::Value]` it was taken from did
// not already have.
unsafe impl<Rt> Send for Elements<Rt> where Rt: Runtime {}
// SAFETY: as `Send`.
unsafe impl<Rt> Sync for Elements<Rt> where Rt: Runtime {}

/// What a `debug_assert!` here reports: the bound belongs to `Index`'s
/// handler (RFC-0047 §6), which panics with Rust's own text.
const OUT_OF_RANGE: &str = "an element was read past the end of a slice";

impl<Rt> Elements<Rt>
where
    Rt: Runtime,
{
    fn of(values: &[Rt::Value]) -> Self {
        Self {
            ptr: values.as_ptr(),
            len: values.len(),
        }
    }

    /// Obligation across artifacts: `acvus-interpreter`'s `ops::index` keeps
    /// a slice in the register pair these two words are (RFC-0047 amended).
    ///
    /// # Safety
    /// `words.ptr` is the first of `words.len` live `Rt::Value`s of one run,
    /// and that run outlives every `Elements` this makes — the loan the
    /// slice holds is what keeps it so (RFC-0018).
    #[inline(always)]
    pub const unsafe fn from_words(words: Words) -> Self {
        Self {
            ptr: words.ptr as *const Rt::Value,
            len: words.len as usize,
        }
    }

    #[inline(always)]
    pub fn words(&self) -> Words {
        Words {
            ptr: self.ptr as u64,
            len: self.len as u64,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The element at `index`.
    ///
    /// # Safety
    /// `index < self.len()`, and the storage this was taken from is live
    /// and unmoved.
    #[inline]
    pub unsafe fn at<'a>(&self, index: usize) -> &'a Rt::Value {
        debug_assert!(index < self.len, "{OUT_OF_RANGE}");
        // SAFETY: the caller's contract: the run is live and `index` is
        // within it.
        unsafe { &*self.ptr.add(index) }
    }

    /// The element at `index`, exclusively.
    ///
    /// # Safety
    /// As `at`, and this `Elements` came from a `SliceMut::of`, whose
    /// `&mut [Rt::Value]` is the exclusivity this hands on.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub unsafe fn at_mut<'a>(&self, index: usize) -> &'a mut Rt::Value {
        debug_assert!(index < self.len, "{OUT_OF_RANGE}");
        // SAFETY: the caller's contract: the run is live, exclusively named,
        // and `index` is within it.
        unsafe { &mut *self.ptr.cast_mut().add(index) }
    }
}

/// `&[T]`: a shared borrow of a run of a container's elements.
pub struct Slice<T, Rt>(Elements<Rt>, PhantomData<T>)
where
    T: TyVar,
    Rt: Runtime;

/// `&mut [T]`: `Slice`'s exclusive twin. Taking one is an exclusive take of
/// the container, so no `Slice` of it is live (RFC-0047 §2).
pub struct SliceMut<T, Rt>(Elements<Rt>, PhantomData<T>)
where
    T: TyVar,
    Rt: Runtime;

impl<T, Rt> Slice<T, Rt>
where
    T: TyVar,
    Rt: Runtime,
{
    /// The elements of a container read in place; the caller holds the
    /// container's loan for as long as the slice.
    pub fn of(values: &[Rt::Value]) -> Self {
        Self(Elements::of(values), PhantomData)
    }

    pub fn from_elements(elements: Elements<Rt>) -> Self {
        Self(elements, PhantomData)
    }

    pub fn into_elements(self) -> Elements<Rt> {
        self.0
    }
}

impl<T, Rt> SliceMut<T, Rt>
where
    T: TyVar,
    Rt: Runtime,
{
    /// As `Slice::of`, for an exclusive take.
    pub fn of(values: &mut [Rt::Value]) -> Self {
        Self(Elements::of(values), PhantomData)
    }

    pub fn from_elements(elements: Elements<Rt>) -> Self {
        Self(elements, PhantomData)
    }

    pub fn into_elements(self) -> Elements<Rt> {
        self.0
    }
}

/// The acvus type: a reference to the unsized `[T]`.
macro_rules! slice_ty_arg {
    ($t:ident, $m:expr) => {
        impl<T, Rt> TyArg for $t<T, Rt>
        where
            T: TyArg + TyVar,
            Rt: Runtime,
        {
            fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
                PolyTy::Ref(
                    $m,
                    Box::new(TypeArg::uniform(PolyTy::Slice(Box::new(T::poly_ty(
                        i, vars,
                    ))))),
                )
            }
        }
    };
}

slice_ty_arg!(Slice, Mutability::Shared);
slice_ty_arg!(SliceMut, Mutability::Mut);

/// A slice crosses as the register pair the machine keeps it in, and as
/// nothing else: it implements `Cross` and not `OneValue`, so a parameter, a
/// field or an element that names one is a compile error. The two words
/// themselves are the runtime's to build and to read, because only the
/// runtime knows what one of its values is made of.
macro_rules! slice_cross {
    ($t:ident) => {
        impl<T, Rt> Cross<Rt> for $t<T, Rt>
        where
            T: TyVar,
            Rt: Runtime,
        {
            type Form = crate::obj::Pair;

            unsafe fn from_run(rt: &Rt, run: &[Rt::Value]) -> Self {
                // SAFETY: the caller's contract: `run` is the pair a slice
                // was written into, and the elements it names are live.
                Self::from_elements(unsafe { Elements::from_words(rt.slice_from_run(run)) })
            }

            fn into_run(self, rt: &Rt, out: &mut [Rt::Value]) {
                rt.slice_into_run(self.into_elements().words(), out)
            }
        }
    };
}

slice_cross!(Slice);
slice_cross!(SliceMut);
