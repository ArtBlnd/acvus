//! `TyArg`: a Rust type that names an acvus type.
//! `TyVar`: a generic parameter that is an acvus type variable.
//! `Typeck<N>`: the compile-time stand-in for the N-th type variable.

use acvus_mir::ty::{EffectTerm, IdentityTerm, IntTy, LenTerm, Poly, PolyBuilder, PolyTy};
use acvus_utils::Interner;

/// The variables a polymorphic ExternFn type ranges over, by kind and
/// position. Built once per declaration from a `PolyBuilder`.
pub struct PolyVars {
    pub tys: Vec<PolyTy>,
    pub effects: Vec<EffectTerm<Poly>>,
    pub lens: Vec<LenTerm<Poly>>,
    pub identities: Vec<IdentityTerm<Poly>>,
}

/// How many variables of each kind a declaration has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VarCounts {
    pub tys: usize,
    pub effects: usize,
    pub lens: usize,
    pub identities: usize,
}

impl PolyVars {
    pub fn empty() -> Self {
        Self::fresh(VarCounts::default())
    }

    pub fn fresh(counts: VarCounts) -> Self {
        let mut b = PolyBuilder::new();
        Self {
            tys: (0..counts.tys).map(|_| b.fresh_ty_var()).collect(),
            effects: (0..counts.effects).map(|_| b.fresh_effect_var()).collect(),
            lens: (0..counts.lens).map(|_| b.fresh_len_var()).collect(),
            identities: (0..counts.identities)
                .map(|_| b.fresh_identity_var())
                .collect(),
        }
    }
}

/// A Rust type that names an acvus type.
pub trait TyArg: 'static {
    fn poly_ty(interner: &Interner, vars: &PolyVars) -> PolyTy;
}

/// A generic parameter that is an acvus type variable. The body never opens
/// it; a body that must cross the runtime boundary goes through the
/// runtime's `materialize`/`erase`. Filled by `Typeck<N>` while the type is
/// built and by the runtime's value at runtime.
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
    ($T:ty, $ty:expr) => {
        impl TyArg for $T {
            fn poly_ty(_: &Interner, _: &PolyVars) -> PolyTy {
                $ty
            }
        }
    };
}

impl_scalar_ty_arg!(i8, PolyTy::I8);
impl_scalar_ty_arg!(i16, PolyTy::I16);
impl_scalar_ty_arg!(i32, PolyTy::I32);
impl_scalar_ty_arg!(i64, PolyTy::I64);
impl_scalar_ty_arg!(u8, PolyTy::U8);
impl_scalar_ty_arg!(u16, PolyTy::U16);
impl_scalar_ty_arg!(u32, PolyTy::U32);
impl_scalar_ty_arg!(u64, PolyTy::U64);
impl_scalar_ty_arg!(f64, PolyTy::Float);
impl_scalar_ty_arg!(String, PolyTy::String);
impl_scalar_ty_arg!(bool, PolyTy::Bool);
impl_scalar_ty_arg!((), PolyTy::Unit);

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
