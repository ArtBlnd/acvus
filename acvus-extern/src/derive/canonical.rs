//! A type stored as itself is boxed at its canonical form and read back
//! through it (`Canonical`).

use std::mem::ManuallyDrop;

use crate::canonical::{Canonical, same_layout};
use crate::runtime::Runtime;
use crate::ty_arg::kind;

pub fn erase<T, Rt>(rt: &Rt, value: T) -> Rt::Value
where
    T: Canonical<kind::Type>,
    Rt: Runtime,
{
    same_layout!(T, T::Canon);
    let held = ManuallyDrop::new(value);
    // SAFETY: `Canonical`'s contract, with the layout checked above; `held`
    // is never read again, so the value moves out exactly once.
    let canon = unsafe { std::ptr::read((&raw const *held).cast::<T::Canon>()) };
    // SAFETY: a type stored as itself is stored as its canonical form.
    unsafe { rt.erase::<T::Canon>(canon) }
}

/// # Safety
/// `value` is what `erase::<T>` wrote.
pub unsafe fn materialize<T, Rt>(rt: &Rt, value: Rt::Value) -> T
where
    T: Canonical<kind::Type>,
    Rt: Runtime,
{
    same_layout!(T, T::Canon);
    // SAFETY: the caller's contract, and `erase` is `erase::<T::Canon>`.
    let canon = ManuallyDrop::new(unsafe { rt.materialize::<T::Canon>(value) });
    // SAFETY: as `erase`'s, read back the other way.
    unsafe { std::ptr::read((&raw const *canon).cast::<T>()) }
}

/// # Safety
/// `reference` names a live storage `erase::<T>` wrote.
pub unsafe fn deref<'a, T, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a T
where
    T: Canonical<kind::Type>,
    Rt: Runtime,
{
    same_layout!(T, T::Canon);
    // SAFETY: the caller's contract, and `Canonical`'s with the layout
    // checked above.
    unsafe { &*(rt.deref::<T::Canon>(reference) as *const T::Canon).cast::<T>() }
}

/// # Safety
/// As `deref`, exclusively.
#[allow(clippy::mut_from_ref)]
pub unsafe fn deref_mut<'a, T, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut T
where
    T: Canonical<kind::Type>,
    Rt: Runtime,
{
    same_layout!(T, T::Canon);
    // SAFETY: as `deref`, with the caller's exclusive loan.
    unsafe { &mut *(rt.deref_mut::<T::Canon>(reference) as *mut T::Canon).cast::<T>() }
}

/// The canonical form's bytes named as `T`: `Stored::from_payload` of a
/// type stored as itself.
pub fn from_canon<T>(canon: &T::Canon) -> &T
where
    T: Canonical<kind::Type>,
{
    same_layout!(T, T::Canon);
    // SAFETY: `Canonical`'s contract, with the layout checked above.
    unsafe { &*(canon as *const T::Canon).cast::<T>() }
}

/// As `from_canon`, exclusively.
pub fn from_canon_mut<T>(canon: &mut T::Canon) -> &mut T
where
    T: Canonical<kind::Type>,
{
    same_layout!(T, T::Canon);
    // SAFETY: as `from_canon`'s; `&mut T::Canon` is the exclusive name.
    unsafe { &mut *(canon as *mut T::Canon).cast::<T>() }
}
