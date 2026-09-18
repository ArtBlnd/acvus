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

use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg, TyVar};

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

/// The crossing: one boxed `Elements` whatever `T` is (RFC-0047 §6), so
/// the machine reads the box without naming `T` and `Value` is untouched.
macro_rules! slice_cross {
    ($t:ident) => {
        impl<T, Rt> crate::Cross<Rt> for $t<T, Rt>
        where
            T: TyVar,
            Rt: Runtime,
        {
            fn erase(self, rt: &Rt) -> Rt::Value {
                // SAFETY: materialized back as this same `Elements<Rt>`.
                unsafe { rt.erase::<Elements<Rt>>(self.0) }
            }

            unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
                // SAFETY: the caller's contract, and `erase` is
                // `erase::<Elements<Rt>>`.
                Self(
                    unsafe { rt.materialize::<Elements<Rt>>(value) },
                    PhantomData,
                )
            }
        }
    };
}

slice_cross!(Slice);
slice_cross!(SliceMut);
