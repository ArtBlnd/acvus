//! An extension type is stored as its payload and read back through it
//! (RFC-0039). These four functions are the whole crossing, and the pointer
//! cast each of them makes is licensed by `Transparent`, which the derive
//! implements only for a `#[repr(transparent)]` struct.

use std::mem::ManuallyDrop;

use crate::runtime::Runtime;

/// `Self` is stored as a `P`.
///
/// # Safety
/// `Self` is `#[repr(transparent)]` with `P` as its one non-zero-sized
/// field, so a `*const Self` and a `*const P` name the same bytes.
pub unsafe trait Transparent<P>: Sized {
    /// Read by every function below, so a `Transparent` impl whose two types
    /// are not one size fails to compile rather than reading past an end.
    const SAME_SIZE: () = assert!(
        std::mem::size_of::<Self>() == std::mem::size_of::<P>(),
        "a transparent type is the size of its payload"
    );
}

pub fn erase<T, P, Rt>(value: T, rt: &Rt) -> Rt::Value
where
    T: Transparent<P>,
    P: Send + Sync + 'static,
    Rt: Runtime,
{
    const { T::SAME_SIZE };
    let held = ManuallyDrop::new(value);
    // SAFETY: `Transparent<P>` licenses the cast and `SAME_SIZE` the extent;
    // `held` is never read again, so the payload moves out exactly once.
    let payload = unsafe { std::ptr::read((&raw const *held).cast::<P>()) };
    // SAFETY: an extension type is stored as its payload.
    unsafe { rt.erase::<P>(payload) }
}

/// # Safety
/// `value` is what `erase` wrote.
pub unsafe fn materialize<T, P, Rt>(rt: &Rt, value: Rt::Value) -> T
where
    T: Transparent<P>,
    P: Send + Sync + 'static,
    Rt: Runtime,
{
    const { T::SAME_SIZE };
    // SAFETY: the caller's contract, and `erase` is `erase::<P>`.
    let payload = ManuallyDrop::new(unsafe { rt.materialize::<P>(value) });
    // SAFETY: as `erase`'s, read back the other way.
    unsafe { std::ptr::read((&raw const payload).cast::<T>()) }
}

/// # Safety
/// `reference` names a live storage holding a `P` erased from a `T`.
pub unsafe fn deref<'a, T, P, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a T
where
    T: Transparent<P>,
    P: Send + Sync + 'static,
    Rt: Runtime,
{
    const { T::SAME_SIZE };
    // SAFETY: the caller's contract, and `Transparent<P>` licenses the cast.
    unsafe { &*(rt.deref::<P>(reference) as *const P).cast::<T>() }
}

/// # Safety
/// As `deref`, exclusively.
#[allow(clippy::mut_from_ref)]
pub unsafe fn deref_mut<'a, T, P, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut T
where
    T: Transparent<P>,
    P: Send + Sync + 'static,
    Rt: Runtime,
{
    const { T::SAME_SIZE };
    // SAFETY: as `deref`, with the caller's exclusive loan.
    unsafe { &mut *(rt.deref_mut::<P>(reference) as *mut P).cast::<T>() }
}
