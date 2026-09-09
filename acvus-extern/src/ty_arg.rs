//! `TyArg`: a Rust type that names an acvus type.
//! `TyVar`: a generic parameter that is an acvus type variable.
//! `Typeck<N>`: the compile-time stand-in for the N-th type variable.

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

/// A generic parameter that is an acvus type variable. The body never opens
/// it; a body that must cross the runtime boundary says so with `FromValue`
/// or `IntoValue`. Filled by `Typeck<N>` while the type is built and by the
/// runtime's value at runtime.
pub trait TyVar: Send + Sync + 'static {}

impl<T: Send + Sync + 'static> TyVar for T {}

/// Compile-time stand-in for the N-th type variable of a declaration.
/// Uninhabited: it names a type and is never a value.
pub enum Typeck<const N: usize> {}

impl<const N: usize> TyArg for Typeck<N> {
    fn poly_ty(_: &Interner, vars: &PolyVars) -> PolyTy {
        vars.tys[N].clone()
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

impl<T, const N: usize> TyArg for [T; N]
where
    T: TyArg,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Array(Box::new(T::poly_ty(i, vars)), LenTerm::Known(N))
    }
}

impl<T> TyArg for Option<T>
where
    T: TyArg,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Option(Box::new(T::poly_ty(i, vars)))
    }
}

macro_rules! impl_tuple_ty_arg {
    ($($T:ident),+) => {
        impl<$($T),+> TyArg for ($($T,)+)
        where
            $($T: TyArg,)+
        {
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

/// The bound of a generic parameter that ranges over a finite set of
/// concrete types: `A: Monomorphize<(i64, f64)>`. The declaration carries
/// the set; the handler is compiled once per member. Every type satisfies
/// the Rust trait; the macro reads the set.
pub trait Monomorphize<Types>: TyVar {}

impl<T: TyVar, Types> Monomorphize<Types> for T {}
