//! `Ref<T>` and `RefMut<T>`: the acvus types `&T` and `&mut T` in an
//! extern signature (RFC-0018). A Rust parameter `&T` / `&mut T` declares
//! one; `Fn1<Ref<T>, R>` declares a lambda that takes a reference. A
//! reference is never materialized: the glue reads the storage it names
//! through the runtime's `deref` / `deref_mut`.

use std::marker::PhantomData;

use acvus_mir::ty::{Mutability, PolyTy};
use acvus_utils::Interner;

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
