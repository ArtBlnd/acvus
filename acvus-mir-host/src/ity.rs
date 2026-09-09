//! ITy: Bridge between Rust host types and MIR compiler types.
//! Hosted: Runtime opaque type marker.
//! Typeck<N>: Compile-time stand-in for generic type parameters.

use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::{InferTy, LenTerm, PolyTy, Ty, lift_to_poly, lift_ty};
use acvus_utils::Interner;

// ── ITy ────────────────────────────────────────────────────────────

/// Bridge between Rust host types and MIR compiler types.
///
/// `type_vars`: pre-allocated type entries (from Solver::fresh_ty_var).
///
/// Concrete types ignore the slice. Typeck<N> indexes into type_vars.
pub trait ITy: Sized + 'static {
    fn ty(interner: &Interner, type_vars: &[Ty]) -> Ty;

    /// Infer-phase version: returns `InferTy` with solver variables.
    /// Default: lifts the concrete type (works for non-generic types).
    fn infer_ty(interner: &Interner, type_vars: &[InferTy]) -> InferTy {
        let _ = type_vars;
        lift_ty(&Self::ty(interner, &[]))
    }

    /// Poly-phase version: returns `PolyTy` with positional Var placeholders.
    /// Default: lifts the concrete type (works for non-generic types).
    fn poly_ty(interner: &Interner, type_vars: &[PolyTy]) -> PolyTy {
        let _ = type_vars;
        lift_to_poly(&Self::ty(interner, &[]))
    }
}

// ── Builtin ITy impls ─────────────────────────────────────────────

impl ITy for i64    { fn ty(_: &Interner, _: &[Ty]) -> Ty { Ty::Int } }
impl ITy for f64    { fn ty(_: &Interner, _: &[Ty]) -> Ty { Ty::Float } }
impl ITy for String { fn ty(_: &Interner, _: &[Ty]) -> Ty { Ty::String } }
impl ITy for bool   { fn ty(_: &Interner, _: &[Ty]) -> Ty { Ty::Bool } }
impl ITy for u8     { fn ty(_: &Interner, _: &[Ty]) -> Ty { Ty::Byte } }
impl ITy for ()     { fn ty(_: &Interner, _: &[Ty]) -> Ty { Ty::Unit } }

impl<T: ITy> ITy for Vec<T> {
    fn ty(i: &Interner, tv: &[Ty]) -> Ty {
        Ty::UserDefined {
            id: QualifiedRef::root(i.intern("List")),
            type_args: vec![T::ty(i, tv)],
            effect_args: vec![],
        }
    }
    fn infer_ty(i: &Interner, tv: &[InferTy]) -> InferTy {
        InferTy::UserDefined {
            id: QualifiedRef::root(i.intern("List")),
            type_args: vec![T::infer_ty(i, tv)],
            effect_args: vec![],
        }
    }
    fn poly_ty(i: &Interner, tv: &[PolyTy]) -> PolyTy {
        PolyTy::UserDefined {
            id: QualifiedRef::root(i.intern("List")),
            type_args: vec![T::poly_ty(i, tv)],
            effect_args: vec![],
        }
    }
}

impl<T: ITy, const N: usize> ITy for [T; N] {
    fn ty(i: &Interner, tv: &[Ty]) -> Ty {
        Ty::Array(Box::new(T::ty(i, tv)), LenTerm::Known(N))
    }
    fn infer_ty(i: &Interner, tv: &[InferTy]) -> InferTy {
        InferTy::Array(Box::new(T::infer_ty(i, tv)), LenTerm::Known(N))
    }
    fn poly_ty(i: &Interner, tv: &[PolyTy]) -> PolyTy {
        PolyTy::Array(Box::new(T::poly_ty(i, tv)), LenTerm::Known(N))
    }
}

impl<T: ITy> ITy for Option<T> {
    fn ty(i: &Interner, tv: &[Ty]) -> Ty {
        Ty::Option(Box::new(T::ty(i, tv)))
    }
    fn infer_ty(i: &Interner, tv: &[InferTy]) -> InferTy {
        InferTy::Option(Box::new(T::infer_ty(i, tv)))
    }
    fn poly_ty(i: &Interner, tv: &[PolyTy]) -> PolyTy {
        PolyTy::Option(Box::new(T::poly_ty(i, tv)))
    }
}

macro_rules! impl_ity_tuple {
    ($($T:ident),+) => {
        impl<$($T: ITy),+> ITy for ($($T,)+) {
            fn ty(i: &Interner, tv: &[Ty]) -> Ty {
                Ty::Tuple(vec![$($T::ty(i, tv)),+])
            }
            fn infer_ty(i: &Interner, tv: &[InferTy]) -> InferTy {
                InferTy::Tuple(vec![$($T::infer_ty(i, tv)),+])
            }
            fn poly_ty(i: &Interner, tv: &[PolyTy]) -> PolyTy {
                PolyTy::Tuple(vec![$($T::poly_ty(i, tv)),+])
            }
        }
    }
}

impl_ity_tuple!(A);
impl_ity_tuple!(A, B);
impl_ity_tuple!(A, B, C);
impl_ity_tuple!(A, B, C, D);

// ── Hosted: runtime opaque type marker ─────────────────────────────

/// Marker for types that represent runtime opaque values.
///
/// Only `Typeck<N>` (compile-time) and `S::Owned` (runtime) implement this.
/// Concrete types (i64, String, etc.) do NOT.
///
/// # Safety
/// Implementors must be valid runtime opaque representations.
pub unsafe trait Hosted: ITy {}

// Vec<T> is Hosted when T: Hosted (enables nesting).
// SAFETY: Vec inherits opaque status from element type.
unsafe impl<T: Hosted> Hosted for Vec<T> {}

// ── Inferrable: restricted return type for infer handlers ──────────

/// The restricted set of values an `infer`-tagged ExternFn can return.
///
/// Only data types — no Function, Iterator, Handle, etc.
/// The system validates Inferrable against the expected type (dst's Ty)
/// and wraps in Option: match → Some, mismatch → None.
///
/// `Failed` forces None unconditionally (e.g., parse error).
#[derive(Debug, Clone)]
pub enum Inferrable {
    Int(i64),
    Float(f64),
    String(std::string::String),
    Bool(bool),
    Byte(u8),
    Unit,
    List(Vec<Inferrable>),
    Object(Vec<(std::string::String, Inferrable)>),
    /// Unconditional None. Handler couldn't produce a value.
    Failed,
}

impl ITy for Inferrable {
    fn ty(_: &Interner, _: &[Ty]) -> Ty {
        // Inferrable is system-internal. Its Ty is determined by the system, not by ITy.
        panic!("Inferrable::ty should not be called — type is determined by infer system")
    }
}

// ── Callable: function signature constraint ────────────────────────

/// Declares that a type is callable with the given argument/return types.
///
/// Used in ExternFn signatures to express function parameter constraints.
///
/// Supertrait: `Hosted`. At runtime, F = S::Owned (just a value).
///
/// # Safety
/// Same as Hosted — implementors must be valid runtime opaque representations.
pub unsafe trait Callable<Args, Ret: ITy>: Hosted {}

// All Hosted types implement Callable for all Args/Ret.
// SAFETY: Hosted types are opaque runtime values. Callable is a constraint marker,
// not a runtime dispatch mechanism. The acvus type system guarantees correctness.
unsafe impl<T: Hosted, Args, Ret: ITy> Callable<Args, Ret> for T {}

// ── Monomorphize: constrained type marker ──────────────────────────

/// Declares which concrete types a generic parameter can be monomorphized to.
///
/// The `#[extern_fn]` macro reads this bound and generates one concrete
/// handler per type in the tuple.
///
/// ```ignore
/// #[extern_fn(name = "add_num", AddNumFn)]
/// fn add_num<A: Monomorphize<(i64, f64)> + Add<Output = A>>(a: A, b: A) -> (A,)
/// ```
pub trait Monomorphize<Types> {}

macro_rules! impl_monomorphize {
    ($($T:ident),+) => {
        impl<__Self, $($T: ITy),+> Monomorphize<($($T,)+)> for __Self {}
    }
}

impl_monomorphize!(A);
impl_monomorphize!(A, B);
impl_monomorphize!(A, B, C);
impl_monomorphize!(A, B, C, D);
impl_monomorphize!(A, B, C, D, E);
impl_monomorphize!(A, B, C, D, E, F);

// ── Typeck<N>: type variable stand-in ──────────────────────────────

/// Compile-time stand-in for generic type parameters.
///
/// The `#[extern_fn]` macro assigns `Typeck<0>`, `Typeck<1>`, ... to each
/// generic type parameter (those with `Hosted` bound).
/// `ty()` returns the pre-allocated type variable from `type_vars[N]`.
pub struct Typeck<const N: usize>;

impl<const N: usize> ITy for Typeck<N> {
    fn ty(_: &Interner, type_vars: &[Ty]) -> Ty {
        type_vars[N].clone()
    }
    fn infer_ty(_: &Interner, type_vars: &[InferTy]) -> InferTy {
        type_vars[N].clone()
    }
    fn poly_ty(_: &Interner, type_vars: &[PolyTy]) -> PolyTy {
        type_vars[N].clone()
    }
}

// SAFETY: Typeck<N> is compile-time only. Never instantiated at runtime.
unsafe impl<const N: usize> Hosted for Typeck<N> {}
