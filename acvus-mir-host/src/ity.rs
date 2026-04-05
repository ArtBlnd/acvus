//! ITy: Bridge between Rust host types and MIR compiler types.
//! Hosted: Runtime opaque type marker.
//! Typeck<N>: Compile-time stand-in for generic type parameters.
//! EffectParam + Eff<N>: Effect variable system.

use acvus_mir::ty::{Effect, Ty};
use acvus_utils::Interner;

// ── ITy ────────────────────────────────────────────────────────────

/// Bridge between Rust host types and MIR compiler types.
///
/// `type_vars`: pre-allocated Ty::Param entries (from TySubst::fresh_param).
/// `effect_vars`: pre-allocated Effect::Var entries (from TySubst::fresh_effect_var).
///
/// Concrete types ignore both slices. Typeck<N> indexes into type_vars.
/// Eff<N> indexes into effect_vars.
pub trait ITy: Sized + 'static {
    fn ty(interner: &Interner, type_vars: &[Ty], effect_vars: &[Effect]) -> Ty;
}

// ── Builtin ITy impls ─────────────────────────────────────────────

impl ITy for i64    { fn ty(_: &Interner, _: &[Ty], _: &[Effect]) -> Ty { Ty::Int } }
impl ITy for f64    { fn ty(_: &Interner, _: &[Ty], _: &[Effect]) -> Ty { Ty::Float } }
impl ITy for String { fn ty(_: &Interner, _: &[Ty], _: &[Effect]) -> Ty { Ty::String } }
impl ITy for bool   { fn ty(_: &Interner, _: &[Ty], _: &[Effect]) -> Ty { Ty::Bool } }
impl ITy for u8     { fn ty(_: &Interner, _: &[Ty], _: &[Effect]) -> Ty { Ty::Byte } }
impl ITy for ()     { fn ty(_: &Interner, _: &[Ty], _: &[Effect]) -> Ty { Ty::Unit } }

impl<T: ITy> ITy for Vec<T> {
    fn ty(i: &Interner, tv: &[Ty], ev: &[Effect]) -> Ty {
        Ty::List(Box::new(T::ty(i, tv, ev)))
    }
}

impl<T: ITy> ITy for Option<T> {
    fn ty(i: &Interner, tv: &[Ty], ev: &[Effect]) -> Ty {
        Ty::Option(Box::new(T::ty(i, tv, ev)))
    }
}

macro_rules! impl_ity_tuple {
    ($($T:ident),+) => {
        impl<$($T: ITy),+> ITy for ($($T,)+) {
            fn ty(i: &Interner, tv: &[Ty], ev: &[Effect]) -> Ty {
                Ty::Tuple(vec![$($T::ty(i, tv, ev)),+])
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
    fn ty(_: &Interner, _: &[Ty], _: &[Effect]) -> Ty {
        // Inferrable is system-internal. Its Ty is determined by the system, not by ITy.
        panic!("Inferrable::ty should not be called — type is determined by infer system")
    }
}

// ── Callable: function signature constraint ────────────────────────

/// Declares that a type is callable with the given argument/return/effect types.
///
/// Used in ExternFn signatures to express function parameter constraints:
/// ```ignore
/// fn filter<T: Hosted, E: EffectParam, F: Callable<(T,), bool, E>>(v: Vec<T>, pred: F) -> Vec<T>
/// fn map<T: Hosted, U: Hosted, E: EffectParam, F: Callable<(T,), U, E>>(
///     it: AcvusIter<T, E>, f: F
/// ) -> AcvusIter<U, E>
/// ```
///
/// Same `E` across params → same `Effect::Var` → unification merges effects.
/// Supertrait: `Hosted`. At runtime, F = S::Owned (just a value).
///
/// # Safety
/// Same as Hosted — implementors must be valid runtime opaque representations.
pub unsafe trait Callable<Args, Ret: ITy, E: EffectParam>: Hosted {}

// All Hosted types implement Callable for all Args/Ret/E.
// SAFETY: Hosted types are opaque runtime values. Callable is a constraint marker,
// not a runtime dispatch mechanism. The acvus type system guarantees correctness.
unsafe impl<T: Hosted, Args, Ret: ITy, E: EffectParam> Callable<Args, Ret, E> for T {}

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

// ── EffectParam: effect variable marker ────────────────────────────

/// Marker for types that represent effect variables in ExternFn signatures.
///
/// Only `Eff<N>` implements this. The macro uses this bound to distinguish
/// effect parameters from type parameters.
pub trait EffectParam: 'static {
    fn effect(effect_vars: &[Effect]) -> Effect;
}

// ── Typeck<N>: type variable stand-in ──────────────────────────────

/// Compile-time stand-in for generic type parameters.
///
/// The `#[extern_fn]` macro assigns `Typeck<0>`, `Typeck<1>`, ... to each
/// generic type parameter (those with `Hosted` bound).
/// `ty()` returns the pre-allocated `Ty::Param` from `type_vars[N]`.
pub struct Typeck<const N: usize>;

impl<const N: usize> ITy for Typeck<N> {
    fn ty(_: &Interner, type_vars: &[Ty], _: &[Effect]) -> Ty {
        type_vars[N].clone()
    }
}

// SAFETY: Typeck<N> is compile-time only. Never instantiated at runtime.
unsafe impl<const N: usize> Hosted for Typeck<N> {}

// ── Eff<N>: effect variable stand-in ───────────────────────────────

/// Compile-time stand-in for effect parameters.
///
/// The `#[extern_fn]` macro assigns `Eff<0>`, `Eff<1>`, ... to each
/// generic effect parameter (those with `EffectParam` bound).
/// Indexes into the pre-allocated `effect_vars` slice.
pub struct Eff<const N: usize>;

impl<const N: usize> EffectParam for Eff<N> {
    fn effect(effect_vars: &[Effect]) -> Effect {
        effect_vars[N].clone()
    }
}

// Eff<N> needs ITy for type-level usage (e.g., as phantom param in AcvusIter<T, E>).
// Returns Unit — effect params don't occupy value slots.
impl<const N: usize> ITy for Eff<N> {
    fn ty(_: &Interner, _: &[Ty], _: &[Effect]) -> Ty { Ty::Unit }
}
