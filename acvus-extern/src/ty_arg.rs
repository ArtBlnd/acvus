//! `TyArg`: a Rust type that names an acvus type.
//! `TyVar`: a generic parameter that is an acvus type variable.
//! `Typeck<N>`: the compile-time stand-in for the N-th type variable.

use acvus_interpreter::{FromValue, IntoValue, RuntimeError, Value};
use acvus_mir::ty::{EffectTerm, LenTerm, Poly, PolyBuilder, PolyTy};
use acvus_utils::Interner;

/// The variables a polymorphic ExternFn type ranges over, by kind and
/// position. Built once per declaration from a `PolyBuilder`.
pub struct PolyVars {
    pub tys: Vec<PolyTy>,
    pub effects: Vec<EffectTerm<Poly>>,
    pub lens: Vec<LenTerm<Poly>>,
}

impl PolyVars {
    pub fn empty() -> Self {
        Self {
            tys: Vec::new(),
            effects: Vec::new(),
            lens: Vec::new(),
        }
    }

    pub fn fresh(tys: usize, effects: usize, lens: usize) -> Self {
        let mut b = PolyBuilder::new();
        Self {
            tys: (0..tys).map(|_| b.fresh_ty_var()).collect(),
            effects: (0..effects).map(|_| b.fresh_effect_var()).collect(),
            lens: (0..lens).map(|_| b.fresh_len_var()).collect(),
        }
    }
}

/// A Rust type that names an acvus type.
pub trait TyArg: 'static {
    fn poly_ty(interner: &Interner, vars: &PolyVars) -> PolyTy;
}

/// A generic parameter that is an acvus type variable: anything that
/// crosses the runtime boundary. Filled by any convertible `TyArg` in a
/// signature, by `Typeck<N>` while the type is built, and by `Value` at
/// runtime.
pub trait TyVar: FromValue + IntoValue + 'static {}

impl<T: FromValue + IntoValue + 'static> TyVar for T {}

/// Compile-time stand-in for the N-th type variable of a declaration.
/// Uninhabited: it names a type and is never a value.
pub enum Typeck<const N: usize> {}

impl<const N: usize> TyArg for Typeck<N> {
    fn poly_ty(_: &Interner, vars: &PolyVars) -> PolyTy {
        vars.tys[N].clone()
    }
}

impl<const N: usize> FromValue for Typeck<N> {
    fn from_value(_: Value, _: &Interner) -> Result<Self, RuntimeError> {
        Err(RuntimeError::internal("Typeck is a compile-time stand-in, never a runtime value"))
    }
}

impl<const N: usize> IntoValue for Typeck<N> {
    fn into_value(self, _: &Interner) -> Value {
        match self {}
    }
}

macro_rules! impl_scalar_ty_arg {
    ($T:ty, $variant:ident) => {
        impl TyArg for $T {
            fn poly_ty(_: &Interner, _: &PolyVars) -> PolyTy {
                PolyTy::$variant
            }
        }
    };
}

impl_scalar_ty_arg!(i64, Int);
impl_scalar_ty_arg!(f64, Float);
impl_scalar_ty_arg!(String, String);
impl_scalar_ty_arg!(bool, Bool);
impl_scalar_ty_arg!(u8, Byte);
impl_scalar_ty_arg!((), Unit);

impl<T: TyArg, const N: usize> TyArg for [T; N] {
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Array(Box::new(T::poly_ty(i, vars)), LenTerm::Known(N))
    }
}

impl<T: TyArg> TyArg for Option<T> {
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Option(Box::new(T::poly_ty(i, vars)))
    }
}

macro_rules! impl_tuple_ty_arg {
    ($($T:ident),+) => {
        impl<$($T: TyArg),+> TyArg for ($($T,)+) {
            fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
                PolyTy::Tuple(vec![$($T::poly_ty(i, vars)),+])
            }
        }
    }
}

impl_tuple_ty_arg!(A);
impl_tuple_ty_arg!(A, B);
impl_tuple_ty_arg!(A, B, C);
impl_tuple_ty_arg!(A, B, C, D);
