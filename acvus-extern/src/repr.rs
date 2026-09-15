//! The crossing at a type's stored shape (RFC-0022, RFC-0032). A type whose
//! Rust name carries phantoms names its stored shape as `Repr`; the runtime
//! keys its vtable by that shape and never touches the bytes. A type that
//! converts (`Cross`) is rebuilt field by field. Every other type is its
//! own shape. The choice is made where the macro expands, by autoref
//! specialization: `Crossing<T, Rt>` has the inherent methods when
//! `T: HasRepr`, the `AsCross` methods when `T: Cross<Rt>`, and the `AsIs`
//! methods otherwise.

use std::marker::PhantomData;
use std::mem::{self, ManuallyDrop};

use crate::obj::Cross;
use crate::runtime::Runtime;

/// # Safety
/// `Self` and `Repr` have the same layout and the same drop, so that a
/// value stored as one is read as the other.
pub unsafe trait HasRepr: Sized {
    type Repr: Send + Sync + 'static;
}

pub struct Crossing<T, Rt>(PhantomData<(T, Rt)>);

impl<T, Rt> Crossing<T, Rt> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T, Rt> Default for Crossing<T, Rt> {
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

impl<T, Rt> Crossing<T, Rt>
where
    T: HasRepr,
    Rt: Runtime,
{
    /// # Safety
    /// As `Runtime::erase`.
    pub unsafe fn erase(&self, rt: &Rt, value: T) -> Rt::Value {
        unsafe { rt.erase::<T::Repr>(cast(value)) }
    }

    /// # Safety
    /// As `Runtime::materialize`.
    pub unsafe fn materialize(&self, rt: &Rt, value: Rt::Value) -> T {
        unsafe { cast(rt.materialize::<T::Repr>(value)) }
    }

    /// # Safety
    /// As `Runtime::deref`.
    pub unsafe fn deref<'a>(&self, rt: &Rt, reference: &'a Rt::Value) -> &'a T {
        unsafe { &*(rt.deref::<T::Repr>(reference) as *const T::Repr as *const T) }
    }

    /// # Safety
    /// As `Runtime::deref_mut`.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn deref_mut<'a>(&self, rt: &Rt, reference: &'a Rt::Value) -> &'a mut T {
        unsafe { &mut *(rt.deref_mut::<T::Repr>(reference) as *mut T::Repr as *mut T) }
    }
}

/// The crossing of a type that converts (RFC-0032); reached by autoref
/// when `T: Cross<Rt>` holds and `T: HasRepr` does not. A converted value
/// has no storage of its own type, so it is not read through a reference.
pub trait AsCross<T, Rt>
where
    Rt: Runtime,
{
    /// # Safety
    /// As `Runtime::erase`.
    unsafe fn erase(&self, rt: &Rt, value: T) -> Rt::Value;
    /// # Safety
    /// As `Runtime::materialize`.
    unsafe fn materialize(&self, rt: &Rt, value: Rt::Value) -> T;
}

impl<T, Rt> AsCross<T, Rt> for Crossing<T, Rt>
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    unsafe fn erase(&self, rt: &Rt, value: T) -> Rt::Value {
        value.erase(rt)
    }

    unsafe fn materialize(&self, rt: &Rt, value: Rt::Value) -> T {
        T::materialize(rt, value)
    }
}

/// The crossing of a type that is its own shape; reached by autoref when
/// neither `T: HasRepr` nor `T: Cross<Rt>` holds.
pub trait AsIs<T, Rt>
where
    Rt: Runtime,
{
    /// # Safety
    /// As `Runtime::erase`.
    unsafe fn erase(&self, rt: &Rt, value: T) -> Rt::Value;
    /// # Safety
    /// As `Runtime::materialize`.
    unsafe fn materialize(&self, rt: &Rt, value: Rt::Value) -> T;
    /// # Safety
    /// As `Runtime::deref`.
    unsafe fn deref<'a>(&self, rt: &Rt, reference: &'a Rt::Value) -> &'a T;
    /// # Safety
    /// As `Runtime::deref_mut`.
    #[allow(clippy::mut_from_ref)]
    unsafe fn deref_mut<'a>(&self, rt: &Rt, reference: &'a Rt::Value) -> &'a mut T;
}

impl<T, Rt> AsIs<T, Rt> for &Crossing<T, Rt>
where
    T: Send + Sync + 'static,
    Rt: Runtime,
{
    unsafe fn erase(&self, rt: &Rt, value: T) -> Rt::Value {
        unsafe { rt.erase::<T>(value) }
    }

    unsafe fn materialize(&self, rt: &Rt, value: Rt::Value) -> T {
        unsafe { rt.materialize::<T>(value) }
    }

    unsafe fn deref<'a>(&self, rt: &Rt, reference: &'a Rt::Value) -> &'a T {
        unsafe { rt.deref::<T>(reference) }
    }

    unsafe fn deref_mut<'a>(&self, rt: &Rt, reference: &'a Rt::Value) -> &'a mut T {
        unsafe { rt.deref_mut::<T>(reference) }
    }
}
