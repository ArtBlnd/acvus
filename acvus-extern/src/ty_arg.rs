//! A variable has a kind.
//!
//! `kind::{Type, Effect, Length, Identity}` are the four kinds a
//! declaration's variables have. `Var<K>` is the bound of a generic
//! parameter that is a variable of kind `K`; `Term<K>` is a Rust type that
//! names a term of that kind; `Nth<K, N>` is the stand-in that fills the
//! `N`-th variable of kind `K` while the declaration's type is built.
//!
//! `TyArg` names an acvus type and is not `Term<kind::Type>`: it takes the
//! interner and carries `SLOT`, the representation a specializing slot
//! gives its argument, and `Term` takes and carries neither.

use std::marker::PhantomData;

use acvus_mir::ty::{EffectTerm, IdentityTerm, LenTerm, Poly, PolyBuilder, PolyTy, Repr, TypeArg};
use acvus_utils::Interner;

/// The variables a polymorphic ExternFn type ranges over, by kind and
/// position. Built once per declaration from a `PolyBuilder`.
pub struct PolyVars {
    pub tys: Vec<PolyTy>,
    pub effects: Vec<EffectTerm<Poly>>,
    pub lens: Vec<LenTerm<Poly>>,
    pub identities: Vec<IdentityTerm<Poly>>,
}

impl PolyVars {
    pub fn empty() -> Self {
        Self::fresh(0, 0, 0, 0)
    }

    /// The declaration's variables: `tys` of the type kind, `effects` of the
    /// effect kind, `lens` of the length kind, `identities` of the identity
    /// kind. How many of each a declaration has is the length of its vector.
    pub fn fresh(tys: usize, effects: usize, lens: usize, identities: usize) -> Self {
        let mut b = PolyBuilder::new();
        Self {
            tys: (0..tys).map(|_| b.fresh_ty_var()).collect(),
            effects: (0..effects).map(|_| b.fresh_effect_var()).collect(),
            lens: (0..lens).map(|_| b.fresh_len_var()).collect(),
            identities: (0..identities).map(|_| b.fresh_identity_var()).collect(),
        }
    }
}

/// The kinds a declaration's variables have. Each is uninhabited: a kind
/// names a class of variables and is never a value.
pub mod kind {
    /// A variable ranging over acvus types.
    pub enum Type {}
    /// A variable ranging over call effects.
    pub enum Effect {}
    /// A variable ranging over array lengths.
    pub enum Length {}
    /// A variable ranging over identity sources.
    pub enum Identity {}
}

/// One of the four kinds, with the term its variables take in a polymorphic
/// type and the declaration's variables of that kind.
pub trait Kind: Send + Sync + 'static {
    /// This kind's term in a polymorphic type.
    type Poly;

    /// The declaration's `n`-th variable of this kind.
    fn nth(vars: &PolyVars, n: usize) -> Self::Poly;
}

impl Kind for kind::Type {
    type Poly = PolyTy;

    fn nth(vars: &PolyVars, n: usize) -> PolyTy {
        vars.tys[n].clone()
    }
}

impl Kind for kind::Effect {
    type Poly = EffectTerm<Poly>;

    fn nth(vars: &PolyVars, n: usize) -> EffectTerm<Poly> {
        vars.effects[n].clone()
    }
}

impl Kind for kind::Length {
    type Poly = LenTerm<Poly>;

    fn nth(vars: &PolyVars, n: usize) -> LenTerm<Poly> {
        vars.lens[n]
    }
}

impl Kind for kind::Identity {
    type Poly = IdentityTerm<Poly>;

    fn nth(vars: &PolyVars, n: usize) -> IdentityTerm<Poly> {
        vars.identities[n]
    }
}

/// A generic parameter of a declaration that is a variable of kind `K`. The
/// body never opens it; a body that must cross the runtime boundary goes
/// through the runtime's `materialize`/`erase`. Filled by `Nth<K, N>` while
/// the declaration's type is built, and at runtime by what the runtime
/// carries a value of that kind in.
///
/// No kind has an impl over an unbounded parameter: a type fills a variable
/// only where this crate wrote the impl, so a type the language does not
/// know cannot reach a declaration by satisfying `Send + Sync`.
pub trait Var<K>: Send + Sync + 'static
where
    K: Kind,
{
}

/// A Rust type that names a term of kind `K` in a declaration's type: a
/// known term such as `Pure`, or the declaration's `N`-th variable of that
/// kind as `Nth<K, N>`.
pub trait Term<K>: Send + Sync + 'static
where
    K: Kind,
{
    fn poly(vars: &PolyVars) -> K::Poly;
}

/// The `N`-th variable of kind `K` of a declaration, as a Rust type:
/// the stand-in that fills the parameter while the declaration's type is
/// built. Uninhabited: it names a variable and is never a value.
pub struct Nth<K, const N: usize>(PhantomData<fn() -> K>, Never);

impl<K, const N: usize> Term<K> for Nth<K, N>
where
    K: Kind,
{
    fn poly(vars: &PolyVars) -> K::Poly {
        K::nth(vars, N)
    }
}

/// The stand-in fills the variable it names. Written once per kind rather
/// than once over `K`, which would be an impl over an unbounded parameter.
impl<const N: usize> Var<kind::Type> for Nth<kind::Type, N> {}
impl<const N: usize> Var<kind::Effect> for Nth<kind::Effect, N> {}
impl<const N: usize> Var<kind::Length> for Nth<kind::Length, N> {}
impl<const N: usize> Var<kind::Identity> for Nth<kind::Identity, N> {}

/// The representation of a specializing slot by what it holds
/// (hash-types.md, Signatures); a composite takes the strongest of its
/// parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SlotRepr {
    Ground,
    Var,
    Member,
}

/// The one `ρ` of a signature (RFC-0041).
const SIGNATURE_RHO: u32 = 0;

impl SlotRepr {
    pub const fn join(self, other: Self) -> Self {
        if (self as u8) >= (other as u8) {
            self
        } else {
            other
        }
    }

    pub const fn repr(self) -> Repr<Poly> {
        match self {
            SlotRepr::Ground => Repr::Uniform,
            SlotRepr::Var => Repr::Var(SIGNATURE_RHO),
            SlotRepr::Member => Repr::Specialized,
        }
    }
}

/// A Rust type that names an acvus type.
pub trait TyArg: Var<kind::Type> {
    const SLOT: SlotRepr = SlotRepr::Ground;

    fn poly_ty(interner: &Interner, vars: &PolyVars) -> PolyTy;

    /// This type as the argument of a specializing slot.
    fn slot(interner: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        TypeArg::new(Self::SLOT.repr(), Self::poly_ty(interner, vars))
    }

    /// This type as an argument of a derived extension type, whose payload
    /// holds it as the Rust type it is.
    fn held(interner: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        let repr = match Self::SLOT {
            SlotRepr::Var => Repr::Uniform,
            SlotRepr::Ground | SlotRepr::Member => Repr::Specialized,
        };
        TypeArg::new(repr, Self::poly_ty(interner, vars))
    }
}

impl<const N: usize> TyArg for Nth<kind::Type, N> {
    const SLOT: SlotRepr = SlotRepr::Var;

    fn poly_ty(_: &Interner, vars: &PolyVars) -> PolyTy {
        vars.tys[N].clone()
    }
}

/// The member stand-in: a `Monomorphize` member in a member instance's
/// type. Its parameter is the member and its `SLOT` is `Member`; that is
/// the whole difference from `Nth<kind::Type, N>`, whose parameter is an
/// index and whose `SLOT` is `Var`.
pub struct Spec<T>(PhantomData<fn() -> T>, Never);

impl<T> Var<kind::Type> for Spec<T> where T: Var<kind::Type> {}

impl<T> TyArg for Spec<T>
where
    T: TyArg,
{
    const SLOT: SlotRepr = SlotRepr::Member;

    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        T::poly_ty(i, vars)
    }
}

crate::cross_one_value!(Nth<kind::Type, N>, const N: usize);

/// The stand-ins appear inside types that ask their element to be
/// `Stored`, such as `Erased<Rt, T>`; being uninhabited, they never cross.
impl<const N: usize, Rt> crate::OneValue<Rt> for Nth<kind::Type, N>
where
    Rt: crate::Runtime,
{
    fn erase(self, _: &Rt) -> Rt::Value {
        match self {}
    }

    unsafe fn materialize(_: &Rt, _: Rt::Value) -> Self {
        panic!("a value of a compile-time stand-in type was materialized")
    }
}

impl<const N: usize, Rt> crate::Stored<Rt> for Nth<kind::Type, N>
where
    Rt: crate::Runtime,
{
    crate::stored_as_itself!();
}

impl<const N: usize, Rt> crate::Borrowable<Rt> for Nth<kind::Type, N>
where
    Rt: crate::Runtime,
{
    crate::whole_box_in_place!(Self, Rt);
}

// SAFETY: `Nth<kind::Type, N>` is uninhabited, so no `&Nth<kind::Type, N>` exists and the
// layout claim is never read.
unsafe impl<const N: usize, Rt> crate::TransparentOver<Rt> for Nth<kind::Type, N> where
    Rt: crate::Runtime
{
}

crate::cross_one_value!(Spec<T>, T: Send + Sync + 'static);

impl<T, Rt> crate::OneValue<Rt> for Spec<T>
where
    T: Send + Sync + 'static,
    Rt: crate::Runtime,
{
    fn erase(self, _: &Rt) -> Rt::Value {
        match self.1 {}
    }

    unsafe fn materialize(_: &Rt, _: Rt::Value) -> Self {
        panic!("a value of a compile-time stand-in type was materialized")
    }
}

impl<T, Rt> crate::Stored<Rt> for Spec<T>
where
    T: Send + Sync + 'static,
    Rt: crate::Runtime,
{
    crate::stored_as_itself!();
}

impl<T, Rt> crate::Borrowable<Rt> for Spec<T>
where
    T: Send + Sync + 'static,
    Rt: crate::Runtime,
{
    crate::whole_box_in_place!(Self, Rt);
}

// SAFETY: as `Nth<kind::Type, N>`: `Spec<T>` holds a `Never` and is uninhabited.
unsafe impl<T, Rt> crate::TransparentOver<Rt> for Spec<T>
where
    T: Send + Sync + 'static,
    Rt: crate::Runtime,
{
}

macro_rules! impl_scalar_ty_arg {
    ($T:ty, $ty:expr) => {
        impl TyArg for $T {
            fn poly_ty(_: &Interner, _: &PolyVars) -> PolyTy {
                $ty
            }
        }

        impl Var<kind::Type> for $T {}
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
impl_scalar_ty_arg!(char, PolyTy::Char);
impl_scalar_ty_arg!(String, PolyTy::String);
impl_scalar_ty_arg!(bool, PolyTy::Bool);
impl_scalar_ty_arg!((), PolyTy::Unit);

/// The language's `!`: an extern fn returning `Never` panics instead of
/// returning, so its call is typed `!` and nothing runs after it
/// (RFC-0038).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Never {}

impl_scalar_ty_arg!(Never, PolyTy::Never);

impl<T, const N: usize> Var<kind::Type> for [T; N] where T: Var<kind::Type> {}

impl<T, const N: usize> TyArg for [T; N]
where
    T: TyArg,
{
    const SLOT: SlotRepr = T::SLOT;

    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Array(Box::new(T::poly_ty(i, vars)), LenTerm::Known(N))
    }
}

impl<T> Var<kind::Type> for Option<T> where T: Var<kind::Type> {}

impl<T> TyArg for Option<T>
where
    T: TyArg,
{
    const SLOT: SlotRepr = T::SLOT;

    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Option(Box::new(T::poly_ty(i, vars)))
    }
}

impl<T, E> Var<kind::Type> for Result<T, E>
where
    T: Var<kind::Type>,
    E: Var<kind::Type>,
{
}

impl<T, E> TyArg for Result<T, E>
where
    T: TyArg,
    E: TyArg,
{
    const SLOT: SlotRepr = T::SLOT.join(E::SLOT);

    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Result(Box::new(T::poly_ty(i, vars)), Box::new(E::poly_ty(i, vars)))
    }
}

macro_rules! impl_tuple_ty_arg {
    ($($T:ident),+) => {
        impl<$($T),+> Var<kind::Type> for ($($T,)+)
        where
            $($T: Var<kind::Type>,)+
        {
        }

        impl<$($T),+> TyArg for ($($T,)+)
        where
            $($T: TyArg,)+
        {
            const SLOT: SlotRepr = SlotRepr::Ground$(.join($T::SLOT))+;

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

/// The bound of a type variable that ranges over a finite set of concrete
/// types: `A: Monomorphize<(i64, f64)>`. The declaration carries the set;
/// the handler is compiled once per member. The macro reads the set.
pub trait Monomorphize<Types>: Var<kind::Type> {}

/// The run-time fill. `Owned<R>: TyArg` is not written and is not missing:
/// the carrier holds a value of whatever acvus type the caller passed, so
/// there is no `poly_ty` for it to answer with.
impl<R> Var<kind::Type> for crate::Owned<R> where R: crate::Runtime {}

/// One concrete type a `Monomorphize` parameter ranges over: the handler is
/// compiled at it, so it fills the parameter there. This list is the reach
/// of `Monomorphize` without a blanket — a member type not named here
/// cannot be one, and the build says so at the declaration.
macro_rules! mono_member {
    ($T:ty) => {
        impl<Types> Monomorphize<Types> for $T {}
    };
}

mono_member!(i64);
mono_member!(f64);
mono_member!(bool);
mono_member!(u8);
mono_member!(String);

/// A `Monomorphize` parameter carrying no bound the erased value fails also
/// compiles one handler for the runtime's own value.
impl<R, Types> Monomorphize<Types> for crate::Owned<R> where R: crate::Runtime {}
