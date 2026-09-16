//! `Ref<T, Rt>` and `RefMut<T, Rt>`: the acvus types `&T` and `&mut T` in
//! an extern declaration, each carrying the reference value itself
//! (RFC-0018, RFC-0028). A Rust parameter `&T` / `&mut T` declares the
//! same type and is read at entry; a carrier is for a body that keeps the
//! reference, returns it, or takes it inside a lambda or an iterator. Its
//! region is the caller's.

use std::marker::PhantomData;

use acvus_mir::ty::{Mutability, PolyTy, TypeArg};
use acvus_utils::Interner;

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
        U: TyVar,
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
        U: TyVar,
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
        U: TyVar,
    {
        self.with_mut(rt, |target| {
            // SAFETY: as `reference_to`.
            RefMut::new(unsafe { rt.reference(value_of::<U, Rt>(f(target))) })
        })
    }
}

/// A reference to `part`, which lives in storage a live loan names.
fn reference_to<U, Rt>(rt: &Rt, part: &U) -> Ref<U, Rt>
where
    U: TyVar,
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

    fn materialize(_: &Rt, value: Rt::Value) -> Self {
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

    fn materialize(_: &Rt, value: Rt::Value) -> Self {
        Self::new(value)
    }
}

/// The runtime value a type variable's item is: `T` is a type variable of
/// the ExternFn, so it is `Rt::Value` at run time (RFC-0022).
///
/// # Safety
/// `T` is a type variable of the calling ExternFn, not a concrete type.
unsafe fn value_of<T, Rt>(item: &T) -> &Rt::Value
where
    T: TyVar,
    Rt: Runtime,
{
    unsafe { &*(item as *const T as *const Rt::Value) }
}
