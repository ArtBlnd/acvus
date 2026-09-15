//! The crossing at a type's stored shape (RFC-0022). A type whose Rust
//! name carries phantoms names its stored shape as `Repr`; the runtime keys
//! its vtable by that shape and never touches the bytes. Every other type
//! is its own shape. The choice is made where the macro expands, by autoref
//! specialization: `Crossing<T>` has the inherent methods when `T: HasRepr`
//! and the `AsIs` methods otherwise.

use std::marker::PhantomData;
use std::mem::{self, ManuallyDrop};

use crate::runtime::Runtime;

/// # Safety
/// `Self` and `Repr` have the same layout and the same drop, so that a
/// value stored as one is read as the other.
pub unsafe trait HasRepr: Sized {
    type Repr: Send + Sync + 'static;
}

pub struct Crossing<T>(PhantomData<T>);

impl<T> Crossing<T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Default for Crossing<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// # Safety
/// `HasRepr` guarantees the layouts agree.
unsafe fn cast<A, B>(a: A) -> B {
    debug_assert_eq!(mem::size_of::<A>(), mem::size_of::<B>());
    let a = ManuallyDrop::new(a);
    unsafe { mem::transmute_copy(&*a) }
}

impl<T> Crossing<T>
where
    T: HasRepr,
{
    /// # Safety
    /// As `Runtime::erase`.
    pub unsafe fn erase<Rt>(&self, rt: &Rt, value: T) -> Rt::Value
    where
        Rt: Runtime,
    {
        unsafe { rt.erase::<T::Repr>(cast(value)) }
    }

    /// # Safety
    /// As `Runtime::materialize`.
    pub unsafe fn materialize<Rt>(&self, rt: &Rt, value: Rt::Value) -> T
    where
        Rt: Runtime,
    {
        unsafe { cast(rt.materialize::<T::Repr>(value)) }
    }

    /// # Safety
    /// As `Runtime::deref`.
    pub unsafe fn deref<'a, Rt>(&self, rt: &Rt, reference: &'a Rt::Value) -> &'a T
    where
        Rt: Runtime,
    {
        unsafe { &*(rt.deref::<T::Repr>(reference) as *const T::Repr as *const T) }
    }

    /// # Safety
    /// As `Runtime::deref_mut`.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn deref_mut<'a, Rt>(&self, rt: &Rt, reference: &'a Rt::Value) -> &'a mut T
    where
        Rt: Runtime,
    {
        unsafe { &mut *(rt.deref_mut::<T::Repr>(reference) as *mut T::Repr as *mut T) }
    }
}

/// The crossing of a type that is its own shape; reached by autoref when
/// `T: HasRepr` does not hold.
pub trait AsIs<T> {
    /// # Safety
    /// As `Runtime::erase`.
    unsafe fn erase<Rt>(&self, rt: &Rt, value: T) -> Rt::Value
    where
        Rt: Runtime;
    /// # Safety
    /// As `Runtime::materialize`.
    unsafe fn materialize<Rt>(&self, rt: &Rt, value: Rt::Value) -> T
    where
        Rt: Runtime;
    /// # Safety
    /// As `Runtime::deref`.
    unsafe fn deref<'a, Rt>(&self, rt: &Rt, reference: &'a Rt::Value) -> &'a T
    where
        Rt: Runtime;
    /// # Safety
    /// As `Runtime::deref_mut`.
    #[allow(clippy::mut_from_ref)]
    unsafe fn deref_mut<'a, Rt>(&self, rt: &Rt, reference: &'a Rt::Value) -> &'a mut T
    where
        Rt: Runtime;
}

impl<T> AsIs<T> for &Crossing<T>
where
    T: Send + Sync + 'static,
{
    unsafe fn erase<Rt>(&self, rt: &Rt, value: T) -> Rt::Value
    where
        Rt: Runtime,
    {
        unsafe { rt.erase::<T>(value) }
    }

    unsafe fn materialize<Rt>(&self, rt: &Rt, value: Rt::Value) -> T
    where
        Rt: Runtime,
    {
        unsafe { rt.materialize::<T>(value) }
    }

    unsafe fn deref<'a, Rt>(&self, rt: &Rt, reference: &'a Rt::Value) -> &'a T
    where
        Rt: Runtime,
    {
        unsafe { rt.deref::<T>(reference) }
    }

    unsafe fn deref_mut<'a, Rt>(&self, rt: &Rt, reference: &'a Rt::Value) -> &'a mut T
    where
        Rt: Runtime,
    {
        unsafe { rt.deref_mut::<T>(reference) }
    }
}
