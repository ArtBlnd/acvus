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

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;

use acvus_mir::ty::{
    EffectArg, EffectTerm, HeldTy, IdentityTerm, LenTerm, Poly, PolyBuilder, PolyTy, TypeArg,
};
use acvus_utils::{Interner, QualifiedRef};

use crate::canonical::Canonical;
use crate::laid::Layout;
use crate::name::Named;
use crate::registry::ExternTypeDecl;

/// The variables a polymorphic ExternFn type ranges over, by kind and
/// position. Built once per declaration from a `PolyBuilder`.
pub struct PolyVars {
    pub tys: Vec<PolyTy>,
    pub effects: Vec<EffectTerm<Poly>>,
    pub lens: Vec<LenTerm<Poly>>,
    pub identities: Vec<IdentityTerm<Poly>>,
    /// The next representation variable, and the one each slot type has
    /// been given: a `ρ` binds the whole tree of what it stands over
    /// (RFC-0041), so two slots of one type share one and two of different
    /// types have one each.
    next_repr: Cell<u32>,
    slot_reprs: RefCell<Vec<(PolyTy, u32)>>,
    names: RefCell<Vec<Named>>,
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
            next_repr: Cell::new(0),
            slot_reprs: RefCell::new(Vec::new()),
            names: RefCell::new(Vec::new()),
        }
    }

    /// The name of the extension type `T`, which this declaration now
    /// reaches.
    pub fn extension<T>(&self, interner: &Interner) -> QualifiedRef
    where
        T: ExternTypeDecl + ?Sized,
    {
        self.reach(Named::extension::<T>(interner))
    }

    /// The name of the derived struct or enum `T`, which this declaration
    /// now reaches.
    pub fn derived<T>(&self, qref: QualifiedRef) -> QualifiedRef
    where
        T: 'static,
    {
        self.reach(Named::derived::<T>(qref))
    }

    fn reach(&self, named: Named) -> QualifiedRef {
        let mut names = self.names.borrow_mut();
        if !names.contains(&named) {
            names.push(named);
        }
        named.qref
    }

    /// Every name this declaration's types reach.
    pub fn names(&self) -> Vec<Named> {
        self.names.borrow().clone()
    }

    /// A representation variable nothing in this declaration has yet.
    pub fn fresh_repr(&self) -> u32 {
        let repr = self.next_repr.get();
        self.next_repr.set(repr + 1);
        repr
    }

    /// The representation variable of a specializing slot holding `ty`.
    fn slot_repr(&self, ty: &PolyTy) -> u32 {
        let known = self
            .slot_reprs
            .borrow()
            .iter()
            .find_map(|(slot, repr)| (slot == ty).then_some(*repr));
        match known {
            Some(repr) => repr,
            None => {
                let repr = self.fresh_repr();
                self.slot_reprs.borrow_mut().push((ty.clone(), repr));
                repr
            }
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
pub trait Var<K>: Canonical<K> + Send + Sync
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
    const SLOT: SlotRepr = SlotRepr::Ground;

    fn poly(vars: &PolyVars) -> K::Poly;
}

/// The `N`-th variable of kind `K` of a declaration, as a Rust type:
/// the stand-in that fills the parameter while the declaration's type is
/// built. Uninhabited: it names a variable and is never a value.
pub struct Nth<K, const N: usize>(PhantomData<fn() -> K>, Bottom);

impl<K, const N: usize> Term<K> for Nth<K, N>
where
    K: Kind,
{
    const SLOT: SlotRepr = SlotRepr::Var;

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

// SAFETY: a stand-in holds no `Erased`.
unsafe impl<const N: usize> Canonical<kind::Type> for Nth<kind::Type, N> {
    type Canon = Self;
}
// SAFETY: as above.
unsafe impl<const N: usize> Canonical<kind::Effect> for Nth<kind::Effect, N> {
    type Canon = Self;
}
// SAFETY: as above.
unsafe impl<const N: usize> Canonical<kind::Length> for Nth<kind::Length, N> {
    type Canon = Self;
}
// SAFETY: as above.
unsafe impl<const N: usize> Canonical<kind::Identity> for Nth<kind::Identity, N> {
    type Canon = Self;
}

/// What a specializing slot's argument holds (hash-types.md,
/// Signatures), which picks the slot's form in `TyArg::slot`: a member
/// part, else a variable part, else neither. A composite takes the
/// strongest of its parts; the tree the form builds keeps each part's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SlotRepr {
    Ground,
    Var,
    Member,
}

impl SlotRepr {
    pub const fn join(self, other: Self) -> Self {
        if (self as u8) >= (other as u8) {
            self
        } else {
            other
        }
    }
}

/// A Rust type that names an acvus type.
pub trait TyArg: Var<kind::Type> {
    const SLOT: SlotRepr = SlotRepr::Ground;
    const LAYOUT: Layout<Self> = Layout::UNREAD;

    fn poly_ty(interner: &Interner, vars: &PolyVars) -> PolyTy;

    /// This type as the argument of a specializing slot (RFC-0041). With a
    /// member part, the member instance's type is the Rust type written
    /// out, so it is the held tree: `#` at every part but a variable's.
    /// Otherwise, with a variable part, a `ρ` stands over the whole
    /// argument until an instance fixes it, the one `ρ` of this slot type.
    /// Otherwise it is uniform.
    fn slot(interner: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        match Self::SLOT {
            SlotRepr::Member => Self::held(interner, vars),
            SlotRepr::Var => {
                let ty = Self::poly_ty(interner, vars);
                TypeArg::Open(vars.slot_repr(&ty), ty)
            }
            SlotRepr::Ground => TypeArg::uniform(Self::poly_ty(interner, vars)),
        }
    }

    /// This type as an argument of an extension type the runtime keeps as
    /// one box of its Rust type: a derived type, or a container stored as
    /// itself. The representation follows the Rust type part by part: a
    /// variable is filled with its run-time instantiation, which is
    /// uniform, and any other part is the Rust type it names, `#`. A type
    /// whose acvus head is a tuple, option, result or array states its
    /// parts; any other head is one `#` leaf. An array states its Rust head
    /// too: `Arr` is the language's array, `[T; N]` a Rust array.
    fn held(interner: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        TypeArg::specialized(Self::poly_ty(interner, vars))
    }
}

/// An effect as an argument of an extension type the runtime keeps as one
/// box of its Rust type, as `TyArg::held` gives a type: a variable is
/// uniform, and a written effect is `#`.
pub fn held_effect<E>(vars: &PolyVars) -> EffectArg<Poly>
where
    E: Term<kind::Effect>,
{
    match E::SLOT {
        SlotRepr::Var => EffectArg::uniform(E::poly(vars)),
        SlotRepr::Ground | SlotRepr::Member => EffectArg::specialized(E::poly(vars)),
    }
}

impl<const N: usize> TyArg for Nth<kind::Type, N> {
    const SLOT: SlotRepr = SlotRepr::Var;

    fn poly_ty(_: &Interner, vars: &PolyVars) -> PolyTy {
        vars.tys[N].clone()
    }

    fn held(interner: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        TypeArg::uniform(Self::poly_ty(interner, vars))
    }
}

/// The member stand-in: a `Monomorphize` member in a member instance's
/// type. Its parameter is the member and its `SLOT` is `Member`; that is
/// the whole difference from `Nth<kind::Type, N>`, whose parameter is an
/// index and whose `SLOT` is `Var`.
pub struct Spec<T>(PhantomData<fn() -> T>, Bottom);

impl<T> Var<kind::Type> for Spec<T> where T: Var<kind::Type> + 'static {}

// SAFETY: a stand-in holds no `Erased`.
unsafe impl<T> Canonical<kind::Type> for Spec<T>
where
    T: Var<kind::Type> + 'static,
{
    type Canon = Self;
}

impl<T> TyArg for Spec<T>
where
    T: TyArg + 'static,
{
    const SLOT: SlotRepr = SlotRepr::Member;

    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        T::poly_ty(i, vars)
    }

    fn held(i: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        T::held(i, vars)
    }
}

/// The stand-in of a signature's type variable bounded by `Chosen`: `Nth`
/// but for its slots. Each instance fills the variable with a Rust type
/// of its own, so the variable's slot is not uniform but a `ρ` the
/// instance's tree binds, the form `TyArg::slot` gives a variable
/// (RFC-0041).
pub struct ChosenNth<const N: usize>(Bottom);

impl<const N: usize> Var<kind::Type> for ChosenNth<N> {}

// SAFETY: a stand-in holds no `Erased`.
unsafe impl<const N: usize> Canonical<kind::Type> for ChosenNth<N> {
    type Canon = Self;
}

impl<const N: usize> TyArg for ChosenNth<N> {
    const SLOT: SlotRepr = SlotRepr::Var;

    fn poly_ty(_: &Interner, vars: &PolyVars) -> PolyTy {
        vars.tys[N].clone()
    }

    fn held(interner: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        Self::slot(interner, vars)
    }
}

crate::within_every!(Nth<kind::Type, N>, const N: usize);
crate::cross_one_value!(Nth<kind::Type, N>, const N: usize);

/// The stand-ins appear inside types that ask their element to be
/// `Stored`, such as `Erased<Rt, T>`; being uninhabited, they never cross.
// SAFETY: a stand-in is uninhabited: `erase` is never reached and `materialize`
// crosses nothing.
unsafe impl<const N: usize, Rt> crate::OneValue<Rt> for Nth<kind::Type, N>
where
    Rt: crate::Runtime,
{
    fn erase(self, _: crate::Crossing<'_, Rt>) -> Rt::Value {
        match self {}
    }

    unsafe fn materialize(_: crate::Crossing<'_, Rt>, _: Rt::Value) -> Self {
        panic!("a value of a compile-time stand-in type was materialized")
    }
}

// SAFETY: the body is `stored_as_canonical!`'s, which names the payload as a
// `Self` through `Canonical`'s layers and reads nothing else; the capability is
// not kept.
unsafe impl<const N: usize, Rt> crate::Stored<Rt> for Nth<kind::Type, N>
where
    Rt: crate::Runtime,
{
    crate::stored_as_canonical!(Rt);
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

// SAFETY: a stand-in is uninhabited, so no value of it reaches a handler.
unsafe impl<'s, T> crate::Within<'s> for Spec<T> {}
crate::cross_one_value!(Spec<T>, T: Send + Sync + 'static);

// SAFETY: as `Nth<kind::Type, N>`'s.
unsafe impl<T, Rt> crate::OneValue<Rt> for Spec<T>
where
    T: Send + Sync + 'static,
    Rt: crate::Runtime,
{
    fn erase(self, _: crate::Crossing<'_, Rt>) -> Rt::Value {
        match self.1 {}
    }

    unsafe fn materialize(_: crate::Crossing<'_, Rt>, _: Rt::Value) -> Self {
        panic!("a value of a compile-time stand-in type was materialized")
    }
}

// SAFETY: as `Nth<kind::Type, N>`'s.
unsafe impl<T, Rt> crate::Stored<Rt> for Spec<T>
where
    T: Var<kind::Type> + 'static,
    Rt: crate::Runtime,
{
    crate::stored_as_canonical!(Rt);
}

impl<T, Rt> crate::Borrowable<Rt> for Spec<T>
where
    T: Var<kind::Type> + 'static,
    Rt: crate::Runtime,
{
    crate::whole_box_in_place!(Self, Rt);
}

// SAFETY: as `Nth<kind::Type, N>`: `Spec<T>` holds a `Bottom` and is uninhabited.
unsafe impl<T, Rt> crate::TransparentOver<Rt> for Spec<T>
where
    T: Send + Sync + 'static,
    Rt: crate::Runtime,
{
}

crate::within_every!(ChosenNth<N>, const N: usize);
crate::cross_one_value!(ChosenNth<N>, const N: usize);

// SAFETY: as `Nth<kind::Type, N>`'s.
unsafe impl<const N: usize, Rt> crate::OneValue<Rt> for ChosenNth<N>
where
    Rt: crate::Runtime,
{
    fn erase(self, _: crate::Crossing<'_, Rt>) -> Rt::Value {
        match self.0 {}
    }

    unsafe fn materialize(_: crate::Crossing<'_, Rt>, _: Rt::Value) -> Self {
        panic!("a value of a compile-time stand-in type was materialized")
    }
}

// SAFETY: as `Nth<kind::Type, N>`'s.
unsafe impl<const N: usize, Rt> crate::Stored<Rt> for ChosenNth<N>
where
    Rt: crate::Runtime,
{
    crate::stored_as_canonical!(Rt);
}

impl<const N: usize, Rt> crate::Borrowable<Rt> for ChosenNth<N>
where
    Rt: crate::Runtime,
{
    crate::whole_box_in_place!(Self, Rt);
}

// SAFETY: as `Nth<kind::Type, N>`: `ChosenNth<N>` holds a `Bottom` and is uninhabited.
unsafe impl<const N: usize, Rt> crate::TransparentOver<Rt> for ChosenNth<N> where Rt: crate::Runtime {}

macro_rules! impl_scalar_ty_arg {
    ($T:ty, $ty:expr) => {
        impl TyArg for $T {
            fn poly_ty(_: &Interner, _: &PolyVars) -> PolyTy {
                $ty
            }
        }

        impl Var<kind::Type> for $T {}

        // SAFETY: a scalar holds no `Erased`.
        unsafe impl Canonical<kind::Type> for $T {
            type Canon = Self;
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
impl_scalar_ty_arg!(char, PolyTy::Char);
impl_scalar_ty_arg!(String, PolyTy::String);
impl_scalar_ty_arg!(bool, PolyTy::Bool);
impl_scalar_ty_arg!((), PolyTy::Unit);

/// The language's `!`: an extern fn returning `Bottom` panics instead of
/// returning, so its call is typed `!` and nothing runs after it
/// (RFC-0038). It is also the uninhabited field of the compile-time
/// stand-ins. It is not Rust's `!` (`Never`, which `Owned` alone names):
/// a trait impl on that alias would conflict with every other crate's impl
/// of the same trait, and this type carries the language's impls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottom {}

impl_scalar_ty_arg!(Bottom, PolyTy::Never);

impl<T, const N: usize> Var<kind::Type> for [T; N] where T: Var<kind::Type> {}

// SAFETY: the element is its own canonical form's.
unsafe impl<T, const N: usize> Canonical<kind::Type> for [T; N]
where
    T: Var<kind::Type>,
{
    type Canon = [T::Canon; N];
}

impl<T, const N: usize> TyArg for [T; N]
where
    T: TyArg,
{
    const SLOT: SlotRepr = T::SLOT;

    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Array(Box::new(T::poly_ty(i, vars)), LenTerm::Known(N))
    }

    /// Its `N` elements in place, where `Arr` keeps its own in a buffer:
    /// another box of the same acvus type.
    fn held(i: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        TypeArg::Specialized(HeldTy::RustArray(Box::new(T::held(i, vars)), N))
    }
}

impl<T> Var<kind::Type> for Option<T> where T: Var<kind::Type> {}

// SAFETY: the payload is its own canonical form's.
unsafe impl<T> Canonical<kind::Type> for Option<T>
where
    T: Var<kind::Type>,
{
    type Canon = Option<T::Canon>;
}

impl<T> TyArg for Option<T>
where
    T: TyArg,
{
    const SLOT: SlotRepr = T::SLOT;

    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Option(Box::new(T::poly_ty(i, vars)))
    }

    fn held(i: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        TypeArg::Specialized(HeldTy::Option(Box::new(T::held(i, vars))))
    }
}

impl<T, E> Var<kind::Type> for Result<T, E>
where
    T: Var<kind::Type>,
    E: Var<kind::Type>,
{
}

// SAFETY: each arm is its own canonical form's.
unsafe impl<T, E> Canonical<kind::Type> for Result<T, E>
where
    T: Var<kind::Type>,
    E: Var<kind::Type>,
{
    type Canon = Result<T::Canon, E::Canon>;
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

    fn held(i: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        TypeArg::Specialized(HeldTy::Result(
            Box::new(T::held(i, vars)),
            Box::new(E::held(i, vars)),
        ))
    }
}

macro_rules! impl_tuple_ty_arg {
    ($($T:ident),+) => {
        impl<$($T),+> Var<kind::Type> for ($($T,)+)
        where
            $($T: Var<kind::Type>,)+
        {
        }

        // SAFETY: each part is its own canonical form's.
        unsafe impl<$($T),+> Canonical<kind::Type> for ($($T,)+)
        where
            $($T: Var<kind::Type>,)+
        {
            type Canon = ($($T::Canon,)+);
        }

        impl<$($T),+> TyArg for ($($T,)+)
        where
            $($T: TyArg,)+
        {
            const SLOT: SlotRepr = SlotRepr::Ground$(.join($T::SLOT))+;

            fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
                PolyTy::Tuple(vec![$($T::poly_ty(i, vars)),+])
            }

            fn held(i: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
                TypeArg::Specialized(HeldTy::Tuple(vec![$($T::held(i, vars)),+]))
            }
        }
    }
}

impl_tuple_ty_arg!(A);
impl_tuple_ty_arg!(A, B);
impl_tuple_ty_arg!(A, B, C);
impl_tuple_ty_arg!(A, B, C, D);
impl_tuple_ty_arg!(A, B, C, D, E);
impl_tuple_ty_arg!(A, B, C, D, E, F);
impl_tuple_ty_arg!(A, B, C, D, E, F, G);
impl_tuple_ty_arg!(A, B, C, D, E, F, G, H);

/// The bound of a type variable that ranges over a finite set of concrete
/// types: `A: Monomorphize<(i64, f64)>`. The declaration carries the set;
/// the handler is compiled once per member. The macro reads the set.
///
/// Sealed: `Member` is unnameable outside this crate, so the impls below
/// are the whole of it.
pub trait Monomorphize<Types>: mono_sealed::Member + Var<kind::Type> {}

mod mono_sealed {
    /// A type `mono_member!` names, or `Erased<R, T>`.
    pub trait Member {}
}

/// The bound of a type variable that each instance fills with a Rust type
/// of its own: `Ts: Var<kind::Type> + Chosen`, where a derived type's
/// payload projects through `Ts`, so no one box serves every `Ts`, the slot
/// cannot be uniform, and the box key keeps `Ts` as it is (RFC-0076).
/// `extern_signature!` and `#[derive(ExternType)]` read it by name. A
/// derived type's generics are Rust generics, so every type that fills a
/// type variable fills a `Chosen` one.
pub trait Chosen: Var<kind::Type> {}

impl<T> Chosen for T where T: Var<kind::Type> {}

/// One concrete type a `Monomorphize` parameter ranges over: the handler is
/// compiled at it, so it fills the parameter there. This list and the
/// `Erased` impl below are the reach of `Monomorphize`, and `Member` is
/// sealed — a member type not named here cannot be one, and the build says
/// so at the declaration.
macro_rules! mono_member {
    ($T:ty) => {
        impl mono_sealed::Member for $T {}
        impl<Types> Monomorphize<Types> for $T {}
    };
}

mono_member!(i64);
mono_member!(f64);
mono_member!(bool);
mono_member!(u8);
mono_member!(String);

/// A `Monomorphize` parameter carrying no bound the erased value fails also
/// compiles one handler for the runtime's own value, `Owned<R>`. The impl
/// holds for every `Erased<R, T>`, since nothing on `Erased` reads `T`
/// (`Canonical`).
impl<R, T> mono_sealed::Member for crate::Erased<R, T>
where
    R: crate::Runtime,
    T: 'static,
{
}

impl<R, T, Types> Monomorphize<Types> for crate::Erased<R, T>
where
    R: crate::Runtime,
    T: 'static,
{
}
