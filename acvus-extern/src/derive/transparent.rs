//! An extension type is stored as its payload's canonical form and read
//! back through it (RFC-0039, RFC-0076). These six functions are the whole
//! crossing. Each casts through `repr::SameLayout`, whose witness `layout`
//! makes from `Transparent`, which the derive implements only for a
//! `#[repr(transparent)]` struct.

use crate::canonical::same_layout;
use crate::repr::SameLayout;
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

/// The layout `Transparent<P>` proves.
#[inline(always)]
fn layout<T, P>() -> SameLayout<T, P>
where
    T: Transparent<P>,
{
    // SAFETY: `Transparent<P>`'s contract: `T` is `repr(transparent)` over
    // `P`'s field, which differs from `P` only as `Canonical` lets a type
    // and its canonical form differ, under its three layers.
    unsafe { same_layout!(T, P) }
}

pub fn erase<T, P, Rt>(value: T, rt: crate::Crossing<'_, Rt>) -> Rt::Value
where
    T: Transparent<P>,
    P: Send + Sync + 'static,
    Rt: Runtime,
{
    // SAFETY: a `P` a derive stores is released as the `T` it came from
    // would be (`Canonical`'s release layer), and nothing reads it at `P`
    // but the runtime's box.
    let payload = unsafe { layout::<T, P>().cast(value) };
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
    // SAFETY: the caller's contract, and `erase` is `erase::<P>`.
    let payload = unsafe { rt.materialize::<P>(value) };
    // SAFETY: the caller's contract: the `P` was erased from a `T`, which it
    // is again.
    unsafe { layout::<T, P>().flip().cast(payload) }
}

/// # Safety
/// `reference` names a live storage holding a `P` erased from a `T`.
pub unsafe fn deref<'a, T, P, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a T
where
    T: Transparent<P>,
    P: Send + Sync + 'static,
    Rt: Runtime,
{
    // SAFETY: the caller's contract, and the `P` was erased from a `T`,
    // which it is again for the loan.
    unsafe { layout::<T, P>().flip().cast_ref(rt.deref::<P>(reference)) }
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
    // SAFETY: as `deref`, with the caller's exclusive loan; what is
    // written as a `T` is a `P` erased from a `T` again.
    unsafe { layout::<T, P>().flip().cast_mut(rt.deref_mut::<P>(reference)) }
}

/// The payload's bytes named as the `T` they were erased from: the derive's
/// `Stored::from_payload`.
pub fn from_payload<T, P>(payload: &P) -> &T
where
    T: Transparent<P>,
{
    // SAFETY: `Transparent<P>` licenses naming a `P` as a `T`.
    unsafe { layout::<T, P>().flip().cast_ref(payload) }
}

/// As `from_payload`, exclusively.
pub fn from_payload_mut<T, P>(payload: &mut P) -> &mut T
where
    T: Transparent<P>,
{
    // SAFETY: as `from_payload`'s; `&mut P` is the exclusive name.
    unsafe { layout::<T, P>().flip().cast_mut(payload) }
}
