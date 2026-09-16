//! `Erased<R, T>`: a runtime value that was erased from a `T`, read and
//! edited as a `T` in place.

use std::marker::PhantomData;

use acvus_mir::ty::PolyTy;
use acvus_utils::Interner;

use crate::obj::{Cross, Inline, Stored};
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg};

/// The bound is `Stored`, not `Cross`, and there is no check in `as_ref`:
/// a type converted on the way in (a derived struct, stored as an `Obj`)
/// has no `T` in storage to read, so it is refused at the type.
#[repr(transparent)]
pub struct Erased<R, T>(R::Value, PhantomData<T>)
where
    R: Runtime,
    T: Stored<R>;

impl<R, T> Erased<R, T>
where
    R: Runtime,
    T: Stored<R>,
{
    pub fn new(rt: &R, value: T) -> Self {
        Self(value.erase(rt), PhantomData)
    }

    pub fn as_ref<'a>(&'a self, rt: &'a R) -> &'a T {
        // SAFETY: `new` erased the value from a `T`, and `T: Stored<R>`
        // makes that the runtime's own `erase::<T>`.
        unsafe { rt.value_as_ref::<T>(&self.0) }
    }

    pub fn as_mut<'a>(&'a mut self, rt: &'a R) -> &'a mut T {
        // SAFETY: as in `as_ref`; `&mut self` is the exclusive name.
        unsafe { rt.value_as_mut::<T>(&mut self.0) }
    }

    pub fn into_inner(self, rt: &R) -> T {
        // SAFETY: `new` erased the value from a `T`.
        unsafe { T::materialize(rt, self.0) }
    }

    pub fn get(&self) -> T
    where
        T: Inline,
    {
        **self
    }
}

impl<R, T> std::ops::Deref for Erased<R, T>
where
    R: Runtime,
    T: Stored<R> + Inline,
{
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: `new` erased the value from a `T`, and an `Inline` `T`
        // lives in the word, which needs no runtime state to read.
        unsafe { R::inline_ref::<T>(&self.0) }
    }
}

impl<R, T> std::ops::DerefMut for Erased<R, T>
where
    R: Runtime,
    T: Stored<R> + Inline,
{
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as in `deref`; `&mut self` is the exclusive name.
        unsafe { R::inline_mut::<T>(&mut self.0) }
    }
}

impl<R, T> PartialEq for Erased<R, T>
where
    R: Runtime,
    T: Stored<R> + Inline + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<R, T> PartialOrd for Erased<R, T>
where
    R: Runtime,
    T: Stored<R> + Inline + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<R, T> std::fmt::Debug for Erased<R, T>
where
    R: Runtime,
    T: Stored<R> + Inline + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

impl<R, T> Cross<R> for Erased<R, T>
where
    R: Runtime,
    T: Stored<R>,
{
    const STORED_AS_VALUE: bool = true;

    fn erase(self, _: &R) -> R::Value {
        self.0
    }

    unsafe fn materialize(_: &R, value: R::Value) -> Self {
        Self(value, PhantomData)
    }

    unsafe fn deref<'a>(rt: &R, reference: &'a R::Value) -> &'a Self {
        // SAFETY: the caller's contract, and `repr(transparent)` over
        // `R::Value`.
        unsafe {
            &*(<R::Value as Cross<R>>::deref(rt, reference) as *const R::Value as *const Self)
        }
    }

    unsafe fn deref_mut<'a>(rt: &R, reference: &'a R::Value) -> &'a mut Self {
        // SAFETY: as in `deref`, exclusively.
        unsafe {
            &mut *(<R::Value as Cross<R>>::deref_mut(rt, reference) as *mut R::Value as *mut Self)
        }
    }
}

impl<R, T> TyArg for Erased<R, T>
where
    R: Runtime,
    T: Stored<R> + TyArg,
{
    fn poly_ty(interner: &Interner, vars: &PolyVars) -> PolyTy {
        T::poly_ty(interner, vars)
    }
}
