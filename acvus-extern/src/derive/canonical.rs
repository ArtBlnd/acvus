//! A type stored as itself is boxed at its canonical form and read back
//! through it (`Canonical`).

use crate::canonical::{Canonical, same_layout};
use crate::repr::SameLayout;
use crate::runtime::Runtime;
use crate::ty_arg::kind;

/// The layout `Canonical` proves.
#[inline(always)]
fn layout<T>() -> SameLayout<T, T::Canon>
where
    T: Canonical<kind::Type>,
{
    // SAFETY: `Canonical`'s contract: `T::Canon` is `T` with each uniform
    // part's `X` at `Never` and each lifetime at `'static`, one layout under
    // its three layers.
    unsafe { same_layout!(T, T::Canon) }
}

pub fn erase<T, Rt>(rt: crate::Crossing<'_, Rt>, value: T) -> Rt::Value
where
    T: Canonical<kind::Type>,
    Rt: Runtime,
{
    // SAFETY: the canonical form releases what `T` would (`Canonical`'s
    // release layer), and nothing reads it at `T::Canon` but the runtime's
    // box, whose `'static` no loan of `T`'s outlives: RFC-0079 rule 7.
    let canon = unsafe { layout::<T>().cast(value) };
    // SAFETY: a type stored as itself is stored as its canonical form.
    unsafe { rt.erase::<T::Canon>(canon) }
}

/// # Safety
/// `value` is what `erase::<T>` wrote.
pub unsafe fn materialize<T, Rt>(rt: crate::Crossing<'_, Rt>, value: Rt::Value) -> T
where
    T: Canonical<kind::Type>,
    Rt: Runtime,
{
    // SAFETY: the caller's contract, and `erase` is `erase::<T::Canon>`.
    let canon = unsafe { rt.materialize::<T::Canon>(value) };
    // SAFETY: the caller's contract: the canonical form was erased from a
    // `T`, which it is again.
    unsafe { layout::<T>().flip().cast(canon) }
}

/// # Safety
/// `reference` names a live storage `erase::<T>` wrote.
pub unsafe fn deref<'a, T, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a T
where
    T: Canonical<kind::Type>,
    Rt: Runtime,
{
    // SAFETY: the caller's contract: the storage was erased from a `T`,
    // which it is again for the loan.
    unsafe { layout::<T>().flip().cast_ref(rt.deref::<T::Canon>(reference)) }
}

/// # Safety
/// As `deref`, exclusively.
#[allow(clippy::mut_from_ref)]
pub unsafe fn deref_mut<'a, T, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut T
where
    T: Canonical<kind::Type>,
    Rt: Runtime,
{
    // SAFETY: as `deref`, with the caller's exclusive loan; what is
    // written as a `T` is a canonical form erased from a `T` again.
    unsafe { layout::<T>().flip().cast_mut(rt.deref_mut::<T::Canon>(reference)) }
}

/// The canonical form's bytes named as `T`: `Stored::from_payload` of a
/// type stored as itself.
pub fn from_canon<'c, T, Rt>(_: crate::Holding<'_, Rt>, canon: &'c T::Canon) -> &'c T
where
    T: Canonical<kind::Type>,
    Rt: Runtime,
{
    // SAFETY: `Canonical`'s contract: a `T::Canon` differs from a `T` in a
    // uniform part's `X`, which only a `PhantomData` holds, and in lifetimes
    // at `'static`, which read at `'c` hold for `'c`.
    unsafe { layout::<T>().flip().cast_ref(canon) }
}

/// As `from_canon`, exclusively.
pub fn from_canon_mut<'c, T, Rt>(_: crate::Holding<'_, Rt>, canon: &'c mut T::Canon) -> &'c mut T
where
    T: Canonical<kind::Type>,
    Rt: Runtime,
{
    // SAFETY: as `from_canon`'s; `&mut T::Canon` is the exclusive name.
    unsafe { layout::<T>().flip().cast_mut(canon) }
}
