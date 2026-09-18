//! `Ref<T, Rt>` and `RefMut<T, Rt>`: the acvus types `&T` and `&mut T` in
//! an extern declaration, each carrying the reference value itself
//! (RFC-0018, RFC-0028). A Rust parameter `&T` / `&mut T` declares the
//! same type and is read at entry; a carrier is for a body that keeps the
//! reference, returns it, or takes it inside a lambda or an iterator. Its
//! region is the caller's.

use std::marker::PhantomData;

use acvus_mir::ty::{Mutability, PolyTy, TypeArg};
use acvus_utils::Interner;

use crate::len::{Arr, LenVar};
use crate::obj::TransparentOver;
use crate::owned::Owned;
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg, TyVar};

pub struct Ref<T, Rt>(Rt::Value, PhantomData<T>)
where
    T: TyVar,
    Rt: Runtime;

pub struct RefMut<T, Rt>(Rt::Value, PhantomData<T>)
where
    T: TyVar,
    Rt: Runtime;

impl<T, Rt> Ref<T, Rt>
where
    T: TyVar,
    Rt: Runtime,
{
    pub fn new(value: Rt::Value) -> Self {
        Self(value, PhantomData)
    }

    /// A reference to a runtime value the caller holds, for a call that
    /// reads it: the callee cannot keep the reference past the call
    /// (RFC-0018), so it never outlives `target`.
    pub fn lend(rt: &Rt, target: &Rt::Value) -> Self {
        // SAFETY: `target` is live for the call, and RFC-0018 keeps the
        // reference within it.
        Self::new(unsafe { rt.reference(target) })
    }

    pub fn into_value(self) -> Rt::Value {
        self.0
    }

    /// Read through the reference.
    pub fn with<R>(&self, rt: &Rt, f: impl FnOnce(&T) -> R) -> R {
        // SAFETY: the storage holds a `T` and is live: the compiler keeps
        // the loan this reference carries for as long as the value exists.
        f(unsafe { rt.deref::<T>(&self.0) })
    }

    /// A reference to a part of what this reference names; `f`'s
    /// signature proves the part lives in the same storage.
    pub fn map<U>(&self, rt: &Rt, f: impl for<'a> FnOnce(&'a T) -> &'a U) -> Ref<U, Rt>
    where
        U: TransparentOver<Rt>,
    {
        self.with(rt, |target| reference_to(rt, f(target)))
    }

    /// As `map`, for a part that may be absent.
    pub fn try_map<U>(
        &self,
        rt: &Rt,
        f: impl for<'a> FnOnce(&'a T) -> Option<&'a U>,
    ) -> Option<Ref<U, Rt>>
    where
        U: TransparentOver<Rt>,
    {
        self.with(rt, |target| f(target).map(|part| reference_to(rt, part)))
    }
}

impl<T, Rt> RefMut<T, Rt>
where
    T: TyVar,
    Rt: Runtime,
{
    pub fn new(value: Rt::Value) -> Self {
        Self(value, PhantomData)
    }

    pub fn into_value(self) -> Rt::Value {
        self.0
    }

    /// Read or write through the reference.
    pub fn with_mut<R>(&self, rt: &Rt, f: impl FnOnce(&mut T) -> R) -> R {
        // SAFETY: as `Ref::with`; the loan is exclusive, so nothing else
        // reads or writes the storage during `f`.
        f(unsafe { rt.deref_mut::<T>(&self.0) })
    }

    /// A mutable reference to a part of what this reference names.
    pub fn map_mut<U>(
        &self,
        rt: &Rt,
        f: impl for<'a> FnOnce(&'a mut T) -> &'a mut U,
    ) -> RefMut<U, Rt>
    where
        U: TransparentOver<Rt>,
    {
        self.with_mut(rt, |target| {
            // SAFETY: as `reference_to`.
            RefMut::new(unsafe { rt.reference(value_of::<U, Rt>(f(target))) })
        })
    }
}

impl<T, Rt> Ref<Vec<T>, Rt>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    /// The elements, read in place: the storage is the runtime's
    /// `Vec<Owned<Rt>>`, and a `Vec<T>` of them is not a view the language
    /// promises, so the typed view is the slice.
    pub fn as_slice<'a>(&'a self, rt: &Rt) -> &'a [T] {
        // SAFETY: as `with`: the storage holds a `Vec<Owned<Rt>>` and is live.
        let values = unsafe { rt.deref::<Vec<Owned<Rt>>>(&self.0) };
        // SAFETY: `T: TransparentOver<Rt>` and `Owned<Rt>` is
        // `repr(transparent)` over `Rt::Value`: `[T]` and `[Owned<Rt>]` are
        // one layout.
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<T>(), values.len()) }
    }
}

impl<T, Rt> RefMut<Vec<T>, Rt>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    /// As `Ref::as_slice`; the elements are edited in place, and the
    /// length is changed only through the stored `Vec<Owned<Rt>>`.
    pub fn as_mut_slice<'a>(&'a mut self, rt: &Rt) -> &'a mut [T] {
        // SAFETY: as `with_mut`: the storage holds a `Vec<Owned<Rt>>`, is
        // live, and the loan is exclusive.
        let values = unsafe { rt.deref_mut::<Vec<Owned<Rt>>>(&self.0) };
        // SAFETY: `T: TransparentOver<Rt>` and `Owned<Rt>` is
        // `repr(transparent)` over `Rt::Value`: `[T]` and `[Owned<Rt>]` are
        // one layout.
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<T>(), values.len()) }
    }
}

/// A sliceable container's storage is a `Vec<Owned<Rt>>` whatever its
/// element type is (RFC-0039), which is why one slice width
/// serves every container (RFC-0047 §1). `elements` is that storage, read
/// in place; `as_slice` above it is the same run seen at an element type
/// that promises the layout.
impl<T, Rt> Ref<Vec<T>, Rt>
where
    T: TyVar,
    Rt: Runtime,
{
    pub fn elements<'a>(&'a self, rt: &Rt) -> &'a [Rt::Value] {
        // SAFETY: as `with`: the storage holds a `Vec<Owned<Rt>>` and is live.
        let owned = unsafe { rt.deref::<Vec<Owned<Rt>>>(&self.0) };
        borrowed(owned)
    }
}

impl<T, Rt> RefMut<Vec<T>, Rt>
where
    T: TyVar,
    Rt: Runtime,
{
    pub fn elements_mut<'a>(&'a self, rt: &Rt) -> &'a mut [Rt::Value] {
        // SAFETY: as `with_mut`: the loan is exclusive, so nothing else
        // names the storage while this slice lives.
        let owned = unsafe { rt.deref_mut::<Vec<Owned<Rt>>>(&self.0) };
        borrowed_mut(owned)
    }
}

impl<T, N, Rt> Ref<Arr<T, N>, Rt>
where
    T: TyVar,
    N: LenVar,
    Rt: Runtime,
{
    pub fn elements<'a>(&'a self, rt: &Rt) -> &'a [Rt::Value] {
        // SAFETY: as `Ref::<Vec<T>>::elements`; the language's array is
        // `Arr<Owned<Rt>, ()>` (RFC-0022, RFC-0048 §7).
        borrowed(&unsafe { rt.deref::<Arr<Owned<Rt>, ()>>(&self.0) }.0)
    }
}

impl<T, N, Rt> RefMut<Arr<T, N>, Rt>
where
    T: TyVar,
    N: LenVar,
    Rt: Runtime,
{
    pub fn elements_mut<'a>(&'a self, rt: &Rt) -> &'a mut [Rt::Value] {
        // SAFETY: as `RefMut::<Vec<T>>::elements_mut`.
        borrowed_mut(&mut unsafe { rt.deref_mut::<Arr<Owned<Rt>, ()>>(&self.0) }.0)
    }
}

/// A run of owned elements seen as the run of values it is: `Elements` is
/// on the ABI side of RFC-0048 §1, where a value owes no release, and the
/// storage keeps the obligation.
fn borrowed<Rt>(owned: &[Owned<Rt>]) -> &[Rt::Value]
where
    Rt: Runtime,
{
    // SAFETY: `Owned<Rt>` is `repr(transparent)` over `Rt::Value`.
    unsafe { std::slice::from_raw_parts(owned.as_ptr().cast::<Rt::Value>(), owned.len()) }
}

/// As `borrowed`, exclusively. The storage still owes the release for
/// every element, so a caller that writes an element through this slice
/// releases the one it replaced; the interpreter's `IndexSet` is that
/// caller (RFC-0048 §1).
fn borrowed_mut<Rt>(owned: &mut [Owned<Rt>]) -> &mut [Rt::Value]
where
    Rt: Runtime,
{
    // SAFETY: as `borrowed`, with the caller's exclusive loan.
    unsafe { std::slice::from_raw_parts_mut(owned.as_mut_ptr().cast::<Rt::Value>(), owned.len()) }
}

/// A reference to `part`, which lives in storage a live loan names.
fn reference_to<U, Rt>(rt: &Rt, part: &U) -> Ref<U, Rt>
where
    U: TransparentOver<Rt>,
    Rt: Runtime,
{
    // SAFETY: `part` is borrowed from storage the caller's reference
    // names, so the reference is to a live `U`.
    Ref::new(unsafe { rt.reference(value_of::<U, Rt>(part)) })
}

impl<T, Rt> TyArg for Ref<T, Rt>
where
    T: TyArg + TyVar,
    Rt: Runtime,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Ref(
            Mutability::Shared,
            Box::new(TypeArg::uniform(T::poly_ty(i, vars))),
        )
    }
}

impl<T, Rt> TyArg for RefMut<T, Rt>
where
    T: TyArg + TyVar,
    Rt: Runtime,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Ref(
            Mutability::Mut,
            Box::new(TypeArg::uniform(T::poly_ty(i, vars))),
        )
    }
}

impl<T, Rt> crate::Cross<Rt> for Ref<T, Rt>
where
    T: TyVar,
    Rt: Runtime,
{
    fn erase(self, _: &Rt) -> Rt::Value {
        self.0
    }

    unsafe fn materialize(_: &Rt, value: Rt::Value) -> Self {
        Self::new(value)
    }
}

impl<T, Rt> crate::Cross<Rt> for RefMut<T, Rt>
where
    T: TyVar,
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
