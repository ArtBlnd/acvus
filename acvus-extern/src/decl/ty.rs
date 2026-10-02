//! Rust types state their declared types (macros §5): the macros ask these
//! traits and never read a type's tokens for its meaning. A place's declared
//! type is its form's (crossing §2), one per Rust type written there.

use acvus_mir::ty::IntTy;
use acvus_utils::Interner;

use crate::boundary::entry::mode::{ByInstance, ByInstanceOf};
use crate::boundary::entry::{PairedMode, RequirementMode};
use crate::boundary::signed::Structural;
use crate::boundary::uniform::{Abstract, Boxed, Element, Index, Never, Uniform};
use crate::callback::{Callable, CallsBack, Func};
use crate::decl::{DeclEffect, DeclTy, Requirement};
use crate::instance::{Holds, Instance, InstanceOf, receiver};
use crate::register::GenericDeclared;
use crate::runtime::{Give, Key, MutKey, Open, Place, RefKey, Referent, Runtime};

/// A Rust type's declared type. Implemented by this crate for the language's
/// own types and the crossing's keys, and by the derive for an
/// extern-defined type.
pub trait DeclType {
    fn decl_ty(names: &Interner) -> DeclTy;
}

/// The declared type of the Rust type `P` a value comes in as, at any loan.
pub fn decl_in<'l, R: Runtime, P: Open<'l, R>>(names: &Interner) -> DeclTy {
    P::decl(names)
}

/// The declared type of the Rust type `P` a value goes out as, at any loan.
pub fn decl_out<'l, R: Runtime, P: Give<'l, R>>(names: &Interner) -> DeclTy {
    P::decl(names)
}

/// The index of a declaration's N-th type parameter. The parameter itself is
/// instantiated, where the declaration is built and where it is dispatched,
/// as the held value at this index, `Uniform<R, Nth<N>>`, whose declared
/// type is `DeclTy::Param(N)`.
pub struct Nth<const N: u32>;

/// The stand-in for the N-th effect variable.
pub struct EffectNth<const N: u32>;

/// The stand-in for the N-th identity variable.
pub struct IdentityNth<const N: u32>;

/// A type variable (compiler: extern fn §1): a held value, `Uniform<R, X>`,
/// the one representation a type variable's place takes (runtime contract
/// §1). It is the bound of a handler's type parameter, of an opaque type's,
/// and of a generic function's variables, and what an impl's or a
/// requirement's choice of one is. The handler holds, moves and lends a value
/// of it without knowing its type, and passes it on as the value it is.
///
/// An opaque type's type parameter is one as well, written so:
///
/// ```compile_fail
/// #[derive(acvus_extern::Opaque)]
/// struct Cell<T> {
///     item: T,
/// }
/// ```
///
/// A type variable's value holds the program's loans and never outlives the
/// call, so a handler puts it nowhere `'static` is required:
///
/// ```compile_fail,E0310
/// use acvus_extern::decl::TypeVar;
/// use acvus_extern::runtime::Runtime;
///
/// fn kept<T: 'static>(value: T) -> T {
///     value
/// }
///
/// #[acvus_extern::extern_fn]
/// fn keep<R: Runtime, T: TypeVar<R>>(x: T) -> T {
///     kept(x)
/// }
/// ```
#[diagnostic::on_unimplemented(
    message = "a type variable's place takes a held value, `Uniform<R, X>`, not `{Self}`",
    label = "not a held value",
    note = "a Rust value is its type's own representation, which never stands for a type variable (runtime contract §1): write `Uniform<R, X>` for the type `X`",
    note = "a slice is of held values: `&[Uniform<R, X>]`"
)]
pub trait TypeVar<R: Runtime>:
    DeclType
    + Referent<R>
    + Element<R>
    + Boxed<R, View = Self>
    + Structural<Canon = Uniform<R, Never>>
    + for<'l> Open<'l, R>
    + for<'l> Give<'l, R>
    + for<'l> Key<'l, R, Give = Self, Open = Self>
    + for<'l> Place<'l, R, Key = Self>
    + Send
    + Sync
{
}

impl<R: Runtime, X: Abstract + DeclType> TypeVar<R> for Uniform<R, X> where X: Index<R, View = Uniform<R, X>, Key = Uniform<R, X>> {}

impl<const N: u32> Abstract for Nth<N> {}

/// An effect variable (compiler: extern fn §2): the one bound a handler's
/// effect parameter carries, and its position among the declaration's. It
/// is no part of a layout, so it is its own canonical form.
pub trait EffectVar: Structural<Canon = Self> + 'static {
    const INDEX: u32;
}

/// An identity variable (compiler: types §3): the one bound a handler's
/// identity parameter carries, and its position among the declaration's. It
/// is no part of a layout, so it is its own canonical form.
pub trait IdentityVar: Structural<Canon = Self> + 'static {
    const INDEX: u32;
}

impl<const N: u32> EffectVar for EffectNth<N> {
    const INDEX: u32 = N;
}

impl<const N: u32> IdentityVar for IdentityNth<N> {
    const INDEX: u32 = N;
}

mod parts {
    use acvus_utils::Interner;

    use super::DeclTy;

    /// A tuple's parts, each a declared type, in order: a tuple's own, or a
    /// function type's parameters.
    pub trait DeclParts {
        fn decl_parts(names: &Interner) -> Vec<DeclTy>;
    }
}

pub use parts::DeclParts;

/// A `#[requires]` parameter's Rust type: a handle at a type, lent for the
/// call, `&InstanceOf<R, g, Ts>`, of a generic function taking its receiver
/// by `&` (impls §3). It picks the mode the entry opens the site's impl by,
/// and states the requirement: the generic function at `Ts`.
#[diagnostic::on_unimplemented(
    message = "`#[requires]` takes a handle at a type lent for the call, `&InstanceOf<R, g, Ts>`, not `{Self}`",
    label = "not a handle at a type",
    note = "a generic function that takes its receiver by `&` is required as `#[requires] name: &InstanceOf<R, g, Ts>`",
    note = "one that takes it by value or by `&mut` is required with its receiver as `#[instance] name: Instance<R, g, Ts>`"
)]
pub trait RequiresParam<R: Runtime> {
    type Mode: RequirementMode<R>;

    fn requirement(names: &Interner) -> Requirement;
}

impl<R, G, Ts> RequiresParam<R> for &InstanceOf<R, G, Ts>
where
    R: Runtime,
    G: GenericDeclared<R> + Holds<R, Ts>,
{
    type Mode = ByInstanceOf<G, Ts>;

    fn requirement(names: &Interner) -> Requirement {
        Requirement { function: G::path(names), pattern: G::pattern(names), fact: None }
    }
}

/// An `#[instance]` parameter's Rust type: a receiver and the impl its site
/// chose for it, `Instance<R, g, Ts>`, of a generic function taking its
/// receiver by value or by `&mut` (impls §3). It is an argument, the
/// receiver taken from the requirer's caller, and a requirement at once: it
/// picks the mode the entry opens both by, and states the receiver's declared
/// type and the requirement.
#[diagnostic::on_unimplemented(
    message = "`#[instance]` takes a receiver with its impl, `Instance<R, g, Ts>`, not `{Self}`",
    label = "not a receiver with its impl",
    note = "a generic function that takes its receiver by value or by `&mut` is required as `#[instance] name: Instance<R, g, Ts>`",
    note = "one that takes it by `&` is required as a handle at a type, `#[requires] name: &InstanceOf<R, g, Ts>`"
)]
pub trait InstanceParam<R: Runtime> {
    type Mode: PairedMode<R>;

    /// The receiver's declared type, the argument's.
    fn decl(names: &Interner) -> DeclTy;

    fn requirement(names: &Interner) -> Requirement;
}

impl<R, G, Ts> InstanceParam<R> for Instance<R, G, Ts>
where
    R: CallsBack,
    G: GenericDeclared<R> + Holds<R, Ts>,
    G::Takes: receiver::Receives,
    G::Held: DeclType,
{
    type Mode = ByInstance<G, Ts>;

    fn decl(names: &Interner) -> DeclTy {
        <G::Held as DeclType>::decl_ty(names)
    }

    fn requirement(names: &Interner) -> Requirement {
        Requirement { function: G::path(names), pattern: G::pattern(names), fact: None }
    }
}

/// The language's own types, each its declared type.
macro_rules! declared {
    ($([$($g:tt)*] $ty:ty => $decl:expr;)+) => {$(
        impl<$($g)*> DeclType for $ty {
            fn decl_ty(_: &Interner) -> DeclTy {
                $decl
            }
        }
    )+};
}

macro_rules! declared_primitives {
    ($($t:ty => $decl:expr),+ $(,)?) => {
        declared! { $([] $t => $decl;)+ }
    };
}

primitives!(declared_primitives);

declared! {
    [] String => DeclTy::String;
    [] str => DeclTy::Str;
}

impl<T: DeclType> DeclType for Option<T> {
    fn decl_ty(names: &Interner) -> DeclTy {
        DeclTy::Option(Box::new(T::decl_ty(names)))
    }
}

impl<T: DeclType, E: DeclType> DeclType for Result<T, E> {
    fn decl_ty(names: &Interner) -> DeclTy {
        DeclTy::Result(Box::new(T::decl_ty(names)), Box::new(E::decl_ty(names)))
    }
}

impl<const N: u32> DeclType for Nth<N> {
    fn decl_ty(_: &Interner) -> DeclTy {
        DeclTy::Param(N)
    }
}

/// The canonical index stands for no type: a held value at it is the
/// canonical form of one at a type variable, and is never declared.
impl DeclType for Never {
    fn decl_ty(_: &Interner) -> DeclTy {
        DeclTy::Never
    }
}

/// A value held at `X`: a type variable's value where `X` is a stand-in.
impl<R: Runtime, X: DeclType> DeclType for Uniform<R, X> {
    fn decl_ty(names: &Interner) -> DeclTy {
        X::decl_ty(names)
    }
}

/// Each tuple: its parts' declared types. No parts: no parameters.
impl DeclParts for () {
    fn decl_parts(_: &Interner) -> Vec<DeclTy> {
        Vec::new()
    }
}

macro_rules! decl_tuples {
    ($($n:literal ($($p:ident $v:ident $i:tt),+))+) => {$(
        impl<$($p: DeclType),+> DeclParts for ($($p,)+) {
            fn decl_parts(names: &Interner) -> Vec<DeclTy> {
                vec![$($p::decl_ty(names)),+]
            }
        }

        impl<$($p: DeclType),+> DeclType for ($($p,)+) {
            fn decl_ty(names: &Interner) -> DeclTy {
                DeclTy::Tuple(<Self as DeclParts>::decl_parts(names))
            }
        }
    )+};
}

tuples!(decl_tuples);

/// The run of elements a container lends; only under a reference.
impl<T: DeclType> DeclType for [T] {
    fn decl_ty(names: &Interner) -> DeclTy {
        DeclTy::Slice(Box::new(T::decl_ty(names)))
    }
}

impl<X: ?Sized + DeclType> DeclType for RefKey<X> {
    fn decl_ty(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: false, target: Box::new(X::decl_ty(names)) }
    }
}

impl<X: ?Sized + DeclType> DeclType for MutKey<X> {
    fn decl_ty(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: true, target: Box::new(X::decl_ty(names)) }
    }
}

/// A function type whose effect is one of the declaration's effect variables
/// (types §2). Its values are held as `Uniform<R, Callable<..>>`, taken by
/// value or lent as any held value is.
impl<A, U, E> DeclType for Callable<'_, A, U, E>
where
    A: DeclParts,
    U: DeclType,
    E: EffectVar,
{
    fn decl_ty(names: &Interner) -> DeclTy {
        DeclTy::Fn { params: A::decl_parts(names), result: Box::new(U::decl_ty(names)), effect: DeclEffect::Var(E::INDEX) }
    }
}

/// A function type as a part of another: the same declared type.
impl<A, U, E> DeclType for Func<A, U, E>
where
    A: DeclParts,
    U: DeclType,
    E: EffectVar,
{
    fn decl_ty(names: &Interner) -> DeclTy {
        DeclTy::Fn { params: A::decl_parts(names), result: Box::new(U::decl_ty(names)), effect: DeclEffect::Var(E::INDEX) }
    }
}

#[cfg(test)]
mod tests;
