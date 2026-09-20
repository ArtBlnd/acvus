//! The strength of a borrow, as a type parameter.
//!
//! A bare marker pair like `Uniform`/`Specialized` was the other candidate and
//! folds nothing: the two impls that differ only in `&`/`&mut` would still be
//! written twice, once per tag. What lets one body serve both bits is that the
//! loan carries the Rust borrow, the runtime calls that open it, and the acvus
//! mutability.

use acvus_mir::ty::Mutability;

use crate::obj::OneValue;
use crate::runtime::Runtime;

/// The strength of a borrow: what separated every `X`/`XMut` pair.
pub trait Loan: Send + Sync + 'static {
    const MUTABILITY: Mutability;

    /// A Rust borrow of `T` at this strength.
    type Of<'a, T>
    where
        T: ?Sized + 'a;

    fn shared<'a, T>(borrow: &'a Self::Of<'_, T>) -> &'a T
    where
        T: ?Sized;

    /// A borrow of a part of what `borrow` names. The function this loan runs
    /// is the one whose borrow it holds; the other is dead at that
    /// instantiation.
    fn project<'a, T, U>(
        borrow: Self::Of<'a, T>,
        shared: impl FnOnce(&'a T) -> &'a U,
        exclusive: impl FnOnce(&'a mut T) -> &'a mut U,
    ) -> Self::Of<'a, U>
    where
        T: ?Sized + 'a,
        U: ?Sized + 'a;

    /// # Safety
    /// As `Runtime::deref`, exclusively for `Mut`.
    unsafe fn deref<'a, T, Rt>(rt: &Rt, reference: &'a Rt::Value) -> Self::Of<'a, T>
    where
        T: Send + Sync + 'static,
        Rt: Runtime;

    /// # Safety
    /// As `OneValue::deref`, exclusively for `Mut`.
    unsafe fn borrow<'a, T, Rep, Rt>(rt: &Rt, reference: &'a Rt::Value) -> Self::Of<'a, T>
    where
        T: OneValue<Rt, Rep>,
        Rt: Runtime;

    /// # Safety
    /// As `Runtime::value_as_ref`.
    unsafe fn value_as<'a, T, Rt>(rt: &'a Rt, value: Self::Of<'a, Rt::Value>) -> Self::Of<'a, T>
    where
        T: Send + Sync + 'static,
        Rt: Runtime;

    /// # Safety
    /// As `Runtime::some_at`.
    unsafe fn some_at<'a, Rt>(
        rt: &'a Rt,
        value: Self::Of<'a, Rt::Value>,
    ) -> Option<Self::Of<'a, Rt::Value>>
    where
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

    fn shared<'a, T>(borrow: &'a &T) -> &'a T
    where
        T: ?Sized,
    {
        borrow
    }

    fn project<'a, T, U>(
        borrow: &'a T,
        shared: impl FnOnce(&'a T) -> &'a U,
        _: impl FnOnce(&'a mut T) -> &'a mut U,
    ) -> &'a U
    where
        T: ?Sized + 'a,
        U: ?Sized + 'a,
    {
        shared(borrow)
    }

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
        T: OneValue<Rt, Rep>,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract.
        unsafe { <T as OneValue<Rt, Rep>>::deref(rt, reference) }
    }

    unsafe fn value_as<'a, T, Rt>(rt: &'a Rt, value: &'a Rt::Value) -> &'a T
    where
        T: Send + Sync + 'static,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract.
        unsafe { rt.value_as_ref::<T>(value) }
    }

    unsafe fn some_at<'a, Rt>(rt: &'a Rt, value: &'a Rt::Value) -> Option<&'a Rt::Value>
    where
        Rt: Runtime,
    {
        // SAFETY: the caller's contract.
        unsafe { rt.some_at(value) }
    }
}

impl Loan for Mut {
    const MUTABILITY: Mutability = Mutability::Mut;

    type Of<'a, T>
        = &'a mut T
    where
        T: ?Sized + 'a;

    fn shared<'a, T>(borrow: &'a &mut T) -> &'a T
    where
        T: ?Sized,
    {
        borrow
    }

    fn project<'a, T, U>(
        borrow: &'a mut T,
        _: impl FnOnce(&'a T) -> &'a U,
        exclusive: impl FnOnce(&'a mut T) -> &'a mut U,
    ) -> &'a mut U
    where
        T: ?Sized + 'a,
        U: ?Sized + 'a,
    {
        exclusive(borrow)
    }

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
        T: OneValue<Rt, Rep>,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract, exclusive for this loan.
        unsafe { <T as OneValue<Rt, Rep>>::deref_mut(rt, reference) }
    }

    unsafe fn value_as<'a, T, Rt>(rt: &'a Rt, value: &'a mut Rt::Value) -> &'a mut T
    where
        T: Send + Sync + 'static,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract, exclusive for this loan.
        unsafe { rt.value_as_mut::<T>(value) }
    }

    unsafe fn some_at<'a, Rt>(rt: &'a Rt, value: &'a mut Rt::Value) -> Option<&'a mut Rt::Value>
    where
        Rt: Runtime,
    {
        // SAFETY: the caller's contract, exclusive for this loan.
        unsafe { rt.some_at_mut(value) }
    }
}
