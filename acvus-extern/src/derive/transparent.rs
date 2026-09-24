//! An extension type is stored as its payload's canonical form and read
//! back through it (RFC-0039, RFC-0076). These six functions are the whole
//! crossing, and the pointer cast each of them makes is licensed by
//! `Transparent`, which the derive implements only for a
//! `#[repr(transparent)]` struct.

use std::mem::ManuallyDrop;

use crate::canonical::same_layout;
use crate::runtime::Runtime;

/// `Self` is stored as a `P`.
///
/// # Safety
/// `Self` is `#[repr(transparent)]` over one non-zero-sized field, and `P`
/// is that field's type with each uniform type parameter `X` replaced by
/// `<X as Canonical<kind::Type>>::Canon` and each lifetime at `'static`. The
/// two differ as `Canonical`'s contract lets a type and its canonical form
/// differ, and the read between them rests on its three layers; the derive
/// proves the payload's part of the third as `UniformPayload`, or
/// `unsafe(uniform_payload)` asserts it.
pub unsafe trait Transparent<P>: Sized {}

pub fn erase<T, P, Rt>(value: T, rt: crate::Crossing<'_, Rt>) -> Rt::Value
where
    T: Transparent<P>,
    P: Send + Sync + 'static,
    Rt: Runtime,
{
    same_layout!(T, P);
    let held = ManuallyDrop::new(value);
    // SAFETY: `Transparent<P>` licenses the cast, with the layout checked
    // above; `held` is never read again, so the payload moves out exactly
    // once.
    let payload = unsafe { std::ptr::read((&raw const *held).cast::<P>()) };
    // SAFETY: an extension type is stored as its payload's canonical form.
    unsafe { rt.erase::<P>(payload) }
}

/// # Safety
/// `value` is what `erase` wrote.
pub unsafe fn materialize<T, P, Rt>(rt: crate::Crossing<'_, Rt>, value: Rt::Value) -> T
where
    T: Transparent<P>,
    P: Send + Sync + 'static,
    Rt: Runtime,
{
    same_layout!(T, P);
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
    same_layout!(T, P);
    // SAFETY: the caller's contract, and `Transparent<P>` licenses the cast,
    // with the layout checked above.
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
    same_layout!(T, P);
    // SAFETY: as `deref`, with the caller's exclusive loan.
    unsafe { &mut *(rt.deref_mut::<P>(reference) as *mut P).cast::<T>() }
}

/// The payload's bytes named as the `T` they were erased from: the derive's
/// `Stored::from_payload`.
pub fn from_payload<T, P>(payload: &P) -> &T
where
    T: Transparent<P>,
{
    same_layout!(T, P);
    // SAFETY: `Transparent<P>` licenses the cast, with the layout checked
    // above.
    unsafe { &*(payload as *const P).cast::<T>() }
}

/// As `from_payload`, exclusively.
pub fn from_payload_mut<T, P>(payload: &mut P) -> &mut T
where
    T: Transparent<P>,
{
    same_layout!(T, P);
    // SAFETY: as `from_payload`'s; `&mut P` is the exclusive name.
    unsafe { &mut *(payload as *mut P).cast::<T>() }
}
