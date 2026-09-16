//! Array-length positions in ExternFn signatures.
//!
//! `LenArg` names a length variable; `LenVar` is the bound of a generic
//! parameter that is one, and `()` fills it at runtime. `Arr<T, N>` is an
//! array whose length the script decides.

use std::marker::PhantomData;

use acvus_mir::ty::{LenTerm, Poly, PolyTy};
use acvus_utils::Interner;

use crate::ty_arg::{PolyVars, TyArg, TyVar};

pub trait LenArg: Send + Sync + 'static {
    fn poly_len(vars: &PolyVars) -> LenTerm<Poly>;
}

pub trait LenVar: Send + Sync + 'static {}

impl<N: LenArg> LenVar for N {}
impl LenVar for () {}

/// The K-th length variable of a declaration. Uninhabited.
pub enum Len<const K: usize> {}

impl<const K: usize> LenArg for Len<K> {
    fn poly_len(vars: &PolyVars) -> LenTerm<Poly> {
        vars.lens[K]
    }
}

/// `Array<T, N>` with N a length variable. Holds the elements at runtime.
///
/// `Cross::deref` in `obj.rs` reads the runtime's `Arr<Value, ()>` as an
/// `Arr<Value, N>` through this `repr(transparent)`; drop the attribute and
/// that file still compiles while the read stands on nothing the language
/// promises.
#[repr(transparent)]
pub struct Arr<T, N>(pub Vec<T>, PhantomData<N>)
where
    T: TyVar,
    N: LenVar;

impl<T, N> Arr<T, N>
where
    T: TyVar,
    N: LenVar,
{
    pub fn new(items: Vec<T>) -> Self {
        Self(items, PhantomData)
    }
}

impl<T, N> IntoIterator for Arr<T, N>
where
    T: TyVar,
    N: LenVar,
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T, N> TyArg for Arr<T, N>
where
    T: TyArg + TyVar,
    N: LenArg,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Array(Box::new(T::poly_ty(i, vars)), N::poly_len(vars))
    }
}
