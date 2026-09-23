//! The strength of a borrow, as a type parameter.
//!
//! A bare marker pair like `Uniform`/`Specialized` was the other candidate and
//! folds nothing: the two impls that differ only in `&`/`&mut` would still be
//! written twice, once per tag. What lets one body serve both bits is that the
//! loan carries the Rust borrow, the runtime calls that open it, and the acvus
//! mutability.

use acvus_mir::ty::Mutability;

use crate::handler::Lends;
use crate::runtime::Runtime;

/// The strength of a borrow: what separated every `X`/`XMut` pair. A
/// handler names its two markers, `Shared` and `Mut`, as type arguments of
/// `Ref` and `Slice`; the methods are the crossing's.
#[doc(hidden)]
pub trait Loan: Send + Sync + 'static {
    const MUTABILITY: Mutability;

    /// A Rust borrow of `T` at this strength.
    type Of<'a, T>
    where
        T: ?Sized + 'a;

    /// # Safety
    /// As `Runtime::deref`, exclusively for `Mut`.
    unsafe fn deref<'a, T, Rt>(rt: &Rt, reference: &'a Rt::Value) -> Self::Of<'a, T>
    where
        T: Send + Sync + 'static,
        Rt: Runtime;

    /// # Safety
    /// As `Borrowable::deref`, exclusively for `Mut`.
    unsafe fn borrow<'a, T, Rep, Rt>(rt: &Rt, reference: &'a Rt::Value) -> Self::Of<'a, T>
    where
        T: Send + Sync + 'static,
        Rep: Lends<T, Rt>,
        Rt: Runtime;

    /// # Safety
    /// As `Runtime::value_as_ref`.
    unsafe fn value_as<'a, T, Rt>(rt: &'a Rt, value: Self::Of<'a, Rt::Value>) -> Self::Of<'a, T>
    where
        T: Send + Sync + 'static,
        Rt: Runtime;
}

pub struct Shared;
pub struct Mut;

impl Loan for Shared {
    const MUTABILITY: Mutability = Mutability::Shared;

    type Of<'a, T>
        = &'a T
    where
        T: ?Sized + 'a;

    unsafe fn deref<'a, T, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a T
    where
        T: Send + Sync + 'static,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract.
        unsafe { rt.deref::<T>(reference) }
    }

    unsafe fn borrow<'a, T, Rep, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a T
    where
        T: Send + Sync + 'static,
        Rep: Lends<T, Rt>,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract.
        unsafe { Rep::deref(rt, reference) }
    }

    unsafe fn value_as<'a, T, Rt>(rt: &'a Rt, value: &'a Rt::Value) -> &'a T
    where
        T: Send + Sync + 'static,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract.
        unsafe { rt.value_as_ref::<T>(value) }
    }
}

impl Loan for Mut {
    const MUTABILITY: Mutability = Mutability::Mut;

    type Of<'a, T>
        = &'a mut T
    where
        T: ?Sized + 'a;

    unsafe fn deref<'a, T, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut T
    where
        T: Send + Sync + 'static,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract, exclusive for this loan.
        unsafe { rt.deref_mut::<T>(reference) }
    }

    unsafe fn borrow<'a, T, Rep, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut T
    where
        T: Send + Sync + 'static,
        Rep: Lends<T, Rt>,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract, exclusive for this loan.
        unsafe { Rep::deref_mut(rt, reference) }
    }

    unsafe fn value_as<'a, T, Rt>(rt: &'a Rt, value: &'a mut Rt::Value) -> &'a mut T
    where
        T: Send + Sync + 'static,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract, exclusive for this loan.
        unsafe { rt.value_as_mut::<T>(value) }
    }
}
