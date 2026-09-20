//! `Ref<T, M, Rt>`: the acvus types `&T` and `&mut T` in an extern
//! declaration, carrying the reference value itself (RFC-0018, RFC-0028). A
//! Rust parameter `&T` / `&mut T` declares the same type and is read at entry;
//! a carrier is for a body that keeps the reference, returns it, or takes it
//! inside a lambda or an iterator. Its region is the caller's.

use std::marker::PhantomData;

use acvus_mir::ty::{PolyTy, TypeArg};
use acvus_utils::Interner;

use crate::len::Arr;
use crate::loan::Loan;
use crate::obj::TransparentOver;
use crate::owned::Owned;
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg, Var, kind};

pub struct Ref<T, M, Rt>(Rt::Value, PhantomData<(T, M)>)
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime;

impl<T, M, Rt> Ref<T, M, Rt>
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    pub fn new(value: Rt::Value) -> Self {
        Self(value, PhantomData)
    }

    /// A reference to a runtime value the caller holds, for a call that reads
    /// it: the callee cannot keep the reference past the call (RFC-0018), so
    /// it never outlives `target`.
    pub fn lend(rt: &Rt, target: M::Of<'_, Rt::Value>) -> Self {
        // SAFETY: `target` is live for the call, and RFC-0018 keeps the
        // reference within it.
        Self::new(unsafe { rt.reference(M::shared(&target)) })
    }

    pub fn into_value(self) -> Rt::Value {
        self.0
    }

    /// Read, and for a `Mut` loan write, through the reference.
    pub fn with<'a, R>(&'a self, rt: &Rt, f: impl FnOnce(M::Of<'a, T>) -> R) -> R {
        // SAFETY: the storage holds a `T` and is live: the compiler keeps the
        // loan this reference carries for as long as the value exists, and an
        // exclusive loan is the only live name of it.
        f(unsafe { M::deref::<T, Rt>(rt, &self.0) })
    }

    /// A reference to a part of what this reference names; `f`'s signature
    /// proves the part lives in the same storage.
    pub fn map<'a, U>(
        &'a self,
        rt: &Rt,
        f: impl FnOnce(M::Of<'a, T>) -> M::Of<'a, U>,
    ) -> Ref<U, M, Rt>
    where
        U: TransparentOver<Rt>,
    {
        self.with(rt, |target| reference_to(rt, M::shared(&f(target))))
    }

    pub fn try_map<'a, U>(
        &'a self,
        rt: &Rt,
        f: impl FnOnce(M::Of<'a, T>) -> Option<M::Of<'a, U>>,
    ) -> Option<Ref<U, M, Rt>>
    where
        U: TransparentOver<Rt>,
    {
        self.with(rt, |target| {
            f(target).map(|part| reference_to(rt, M::shared(&part)))
        })
    }
}

impl<T, M, Rt> Ref<Vec<T>, M, Rt>
where
    T: TransparentOver<Rt>,
    M: Loan,
    Rt: Runtime,
{
    /// The elements, read in place: the storage is the runtime's
    /// `Vec<Owned<Rt>>`, and a `Vec<T>` of them is not a view the language
    /// promises, so the typed view is the slice.
    pub fn as_slice<'a>(&'a self, rt: &Rt) -> M::Of<'a, [T]> {
        // SAFETY: as `with`: the storage holds a `Vec<Owned<Rt>>` and is live.
        let values = unsafe { M::deref::<Vec<Owned<Rt>>, Rt>(rt, &self.0) };
        M::project(values, transparent, transparent_mut)
    }
}

/// A sliceable container's storage is a `Vec<Owned<Rt>>` whatever its element
/// type is (RFC-0039), which is why one slice width serves every container
/// (RFC-0047 §1). `elements` is that storage, read in place; `as_slice` above
/// it is the same run seen at an element type that promises the layout.
impl<T, M, Rt> Ref<Vec<T>, M, Rt>
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    pub fn elements<'a>(&'a self, rt: &Rt) -> M::Of<'a, [Rt::Value]> {
        // SAFETY: as `with`: the storage holds a `Vec<Owned<Rt>>` and is live.
        let owned = unsafe { M::deref::<Vec<Owned<Rt>>, Rt>(rt, &self.0) };
        M::project(owned, |o| borrowed(o), |o| borrowed_mut(o))
    }
}

impl<T, N, M, Rt> Ref<Arr<T, N>, M, Rt>
where
    T: Send + Sync + 'static,
    N: Var<kind::Length>,
    M: Loan,
    Rt: Runtime,
{
    pub fn elements<'a>(&'a self, rt: &Rt) -> M::Of<'a, [Rt::Value]> {
        // SAFETY: as `Ref::<Vec<T>>::elements`; the language's array is
        // `Arr<Owned<Rt>, ()>` (RFC-0022, RFC-0048 §7).
        let arr = unsafe { M::deref::<Arr<Owned<Rt>, ()>, Rt>(rt, &self.0) };
        M::project(arr, |a| borrowed(&a.0), |a| borrowed_mut(&mut a.0))
    }
}

/// A run of owned elements seen as the run of values it is: `Elements` is on
/// the ABI side of RFC-0048 §1, where a value owes no release, and the storage
/// keeps the obligation.
fn borrowed<Rt>(owned: &[Owned<Rt>]) -> &[Rt::Value]
where
    Rt: Runtime,
{
    // SAFETY: `Owned<Rt>` is `repr(transparent)` over `Rt::Value`.
    unsafe { std::slice::from_raw_parts(owned.as_ptr().cast::<Rt::Value>(), owned.len()) }
}

/// As `borrowed`, exclusively. The storage still owes the release for every
/// element, so a caller that writes an element through this slice releases the
/// one it replaced; the interpreter's `IndexSet` is that caller (RFC-0048 §1).
fn borrowed_mut<Rt>(owned: &mut [Owned<Rt>]) -> &mut [Rt::Value]
where
    Rt: Runtime,
{
    // SAFETY: as `borrowed`, with the caller's exclusive loan.
    unsafe { std::slice::from_raw_parts_mut(owned.as_mut_ptr().cast::<Rt::Value>(), owned.len()) }
}

fn transparent<T, Rt>(values: &Vec<Owned<Rt>>) -> &[T]
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    // SAFETY: `T: TransparentOver<Rt>` and `Owned<Rt>` is `repr(transparent)`
    // over `Rt::Value`: `[T]` and `[Owned<Rt>]` are one layout.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<T>(), values.len()) }
}

fn transparent_mut<T, Rt>(values: &mut Vec<Owned<Rt>>) -> &mut [T]
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    // SAFETY: as `transparent`, with the caller's exclusive loan.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<T>(), values.len()) }
}

/// A reference to `part`, which lives in storage a live loan names.
fn reference_to<U, M, Rt>(rt: &Rt, part: &U) -> Ref<U, M, Rt>
where
    U: TransparentOver<Rt>,
    M: Loan,
    Rt: Runtime,
{
    // SAFETY: `part` is borrowed from storage the caller's reference names, so
    // the reference is to a live `U`.
    Ref::new(unsafe { rt.reference(value_of::<U, Rt>(part)) })
}

impl<T, M, Rt> Var<kind::Type> for Ref<T, M, Rt>
where
    T: Var<kind::Type>,
    M: Loan,
    Rt: Runtime,
{
}

impl<T, M, Rt> TyArg for Ref<T, M, Rt>
where
    T: TyArg + Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Ref(
            M::MUTABILITY,
            Box::new(TypeArg::uniform(T::poly_ty(i, vars))),
        )
    }
}

crate::cross_one_value!(Ref<T, M, __Rt>, T: Send + Sync + 'static, M: crate::Loan);

impl<T, M, Rt> crate::OneValue<Rt> for Ref<T, M, Rt>
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    fn erase(self, _: &Rt) -> Rt::Value {
        self.0
    }

    unsafe fn materialize(_: &Rt, value: Rt::Value) -> Self {
        Self::new(value)
    }
}

fn value_of<U, Rt>(item: &U) -> &Rt::Value
where
    U: TransparentOver<Rt>,
    Rt: Runtime,
{
    // SAFETY: `U: TransparentOver<Rt>` is the promise that `U` is
    // `repr(transparent)` over `Rt::Value`.
    unsafe { &*(item as *const U).cast::<Rt::Value>() }
}
