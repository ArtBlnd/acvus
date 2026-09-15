//! `Ref<T>` and `RefMut<T>`: the acvus types `&T` and `&mut T` in an
//! extern signature (RFC-0018). A Rust parameter `&T` / `&mut T` declares
//! one; `Fn1<Ref<T>, R>` declares a lambda that takes a reference. A
//! reference is never materialized: the glue reads the storage it names
//! through the runtime's `deref` / `deref_mut`. `Lent<T, Rt>` is the
//! reference value itself, for a body that keeps the loan past the call
//! (an iterator over a borrowed container); its region is the caller's.

use std::marker::PhantomData;

use acvus_mir::ty::{Mutability, PolyTy};
use acvus_utils::Interner;

use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg, TyVar};

pub struct Ref<T>(PhantomData<T>)
where
    T: TyVar;

pub struct RefMut<T>(PhantomData<T>)
where
    T: TyVar;

impl<T> TyArg for Ref<T>
where
    T: TyArg + TyVar,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Ref(Mutability::Shared, Box::new(T::poly_ty(i, vars)))
    }
}

impl<T> TyArg for RefMut<T>
where
    T: TyArg + TyVar,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Ref(Mutability::Mut, Box::new(T::poly_ty(i, vars)))
    }
}

pub struct Lent<T, Rt>(Rt::Value, PhantomData<T>)
where
    T: TyVar,
    Rt: Runtime;

impl<T, Rt> Lent<T, Rt>
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

    /// # Safety
    /// The storage the reference names holds a `T` and is live.
    pub unsafe fn get(&self, rt: &Rt) -> &T {
        unsafe { rt.deref::<T>(&self.0) }
    }
}

impl<T, Rt> TyArg for Lent<T, Rt>
where
    T: TyArg + TyVar,
    Rt: Runtime,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Ref(Mutability::Shared, Box::new(T::poly_ty(i, vars)))
    }
}

/// The runtime value a type variable's item is: `T` is a type variable of
/// the ExternFn, so it is `Rt::Value` at run time (RFC-0022).
///
/// # Safety
/// `T` is a type variable of the calling ExternFn, not a concrete type.
pub unsafe fn value_of<T, Rt>(item: &T) -> &Rt::Value
where
    T: TyVar,
    Rt: Runtime,
{
    unsafe { &*(item as *const T as *const Rt::Value) }
}
