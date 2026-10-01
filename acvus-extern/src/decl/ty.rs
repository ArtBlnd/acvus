//! Rust types state their declared types and their parameter modes (macros
//! §4): the macros ask these traits and never read a type's tokens for its
//! meaning.

use acvus_mir::ty::IntTy;
use acvus_utils::Interner;

use crate::boundary::entry::mode::{ByMut, ByRef, BySlice, BySliceMut, ByStr, ByUniform, ByValue};
use crate::boundary::uniform::Uniform;
use crate::callback::{Callable, Lent, LentMut};
use crate::decl::{DeclEffect, DeclTy};
use crate::runtime::Runtime;

/// A Rust type's declared type. Implemented by this crate for the language's
/// own types and the crossing's (`Uniform`, a stand-in, a reference, a
/// callback), and by the derive for an extern-defined type.
pub trait DeclType {
    fn decl_ty(names: &Interner) -> DeclTy;
}

/// The index of a declaration's N-th type parameter. The parameter itself is
/// instantiated, where the declaration is built and where it is dispatched,
/// as the uniform value at this index, `Uniform<R, Nth<N>>`, whose declared
/// type is `DeclTy::Param(N)`.
pub struct Nth<const N: u32>;

/// The stand-in for the N-th effect variable.
pub struct EffectNth<const N: u32>;

/// The stand-in for the N-th identity variable.
pub struct IdentityNth<const N: u32>;

/// The kinds a declaration's generic parameter may be (compiler: extern fn
/// §1-§2, types §3).
pub mod kind {
    pub struct Type;
    pub struct Effect;
    pub struct Identity;
}

/// A generic parameter of kind `K`: the one bound a handler's generic
/// parameter carries. Only the stand-ins are one: a type variable's is its
/// uniform value, so the handler holds, moves and lends a value of it without
/// knowing its type (runtime contract §1).
pub trait Var<K> {}

impl<R: Runtime, const N: u32> Var<kind::Type> for Uniform<R, Nth<N>> {}
impl<const N: u32> Var<kind::Effect> for EffectNth<N> {}
impl<const N: u32> Var<kind::Identity> for IdentityNth<N> {}

/// A handler parameter's Rust type picks the mode the entry opens it by
/// (macros §2): a value, a type variable's `Uniform`, `&T`, `&mut T`, a slice
/// of `Uniform`s, or `&str`. A function value is a `Uniform` at its
/// `Callable` type, taken or lent as any held value is.
pub trait ParamMode {
    /// The mode, as `crate::mode` names it.
    type Mode;
    /// The parameter's declared type.
    fn decl(names: &Interner) -> DeclTy;
}

mod parts {
    use acvus_utils::Interner;

    use super::DeclTy;

    /// A tuple's parts, each a declared type, in order: a tuple's own, or a
    /// callback's parameters.
    pub trait DeclParts {
        fn decl_parts(names: &Interner) -> Vec<DeclTy>;
    }
}

use parts::DeclParts;

/// The language's own types taken by value, each its declared type.
macro_rules! by_value {
    ($([$($g:tt)*] $ty:ty => $decl:expr;)+) => {$(
        impl<$($g)*> DeclType for $ty {
            fn decl_ty(_: &Interner) -> DeclTy {
                $decl
            }
        }

        impl<$($g)*> ParamMode for $ty {
            type Mode = ByValue<Self>;

            fn decl(names: &Interner) -> DeclTy {
                <Self as DeclType>::decl_ty(names)
            }
        }
    )+};
}

macro_rules! by_value_primitives {
    ($($t:ty => $decl:expr),+ $(,)?) => {
        by_value! { $([] $t => $decl;)+ }
    };
}

primitives!(by_value_primitives);

by_value! {
    [] String => DeclTy::String;
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

/// A part of an option, a result or a tuple a parameter is written with: a
/// value, taken with the whole by value, or a reference, the whole lent and
/// opened as its projection (crossing §2). `Of` is what the part is a part
/// of the lent value as.
pub trait Part {
    type Class;
    type Of: DeclType;
}

/// The classes a part is of.
pub mod part {
    /// A value.
    pub enum Value {}
    /// A shared reference.
    pub enum Shared {}
    /// An exclusive reference.
    pub enum Exclusive {}
}

/// How a parameter whose parts are of a class is passed: by value, or lent
/// whole and opened as its projection.
pub trait Passed<T> {
    type Mode;
    fn decl(names: &Interner) -> DeclTy;
}

impl<T: DeclType> Passed<T> for part::Value {
    type Mode = ByValue<T>;

    fn decl(names: &Interner) -> DeclTy {
        T::decl_ty(names)
    }
}

impl<T: DeclType> Passed<T> for part::Shared {
    type Mode = ByRef<T>;

    fn decl(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: false, target: Box::new(T::decl_ty(names)) }
    }
}

impl<T: DeclType> Passed<T> for part::Exclusive {
    type Mode = ByMut<T>;

    fn decl(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: true, target: Box::new(T::decl_ty(names)) }
    }
}

impl<P: DeclType> Part for &P {
    type Class = part::Shared;
    type Of = P;
}

impl<P: DeclType> Part for &mut P {
    type Class = part::Exclusive;
    type Of = P;
}

macro_rules! value_parts {
    ($([$($g:tt)*] $t:ty;)+) => {$(
        impl<$($g)*> Part for $t {
            type Class = part::Value;
            type Of = $t;
        }
    )+};
}

value_parts! {
    [] String;
    [R: Runtime, X: DeclType] Uniform<R, X>;
}

macro_rules! primitive_parts {
    ($($t:ty => $decl:expr),+ $(,)?) => {
        value_parts! { $([] $t;)+ }
    };
}

primitives!(primitive_parts);

impl<X: Part> Part for Option<X> {
    type Class = X::Class;
    type Of = Option<X::Of>;
}

impl<X: Part, E: Part<Class = X::Class>> Part for Result<X, E> {
    type Class = X::Class;
    type Of = Result<X::Of, E::Of>;
}

/// An option, a result or a tuple as a parameter: by value where its parts are
/// values, lent and projected where they are references.
impl<X: Part> ParamMode for Option<X>
where
    X::Class: Passed<Option<X::Of>>,
{
    type Mode = <X::Class as Passed<Option<X::Of>>>::Mode;

    fn decl(names: &Interner) -> DeclTy {
        <X::Class as Passed<Option<X::Of>>>::decl(names)
    }
}

impl<X: Part, E: Part<Class = X::Class>> ParamMode for Result<X, E>
where
    X::Class: Passed<Result<X::Of, E::Of>>,
{
    type Mode = <X::Class as Passed<Result<X::Of, E::Of>>>::Mode;

    fn decl(names: &Interner) -> DeclTy {
        <X::Class as Passed<Result<X::Of, E::Of>>>::decl(names)
    }
}

impl<const N: u32> DeclType for Nth<N> {
    fn decl_ty(_: &Interner) -> DeclTy {
        DeclTy::Param(N)
    }
}

/// A value at `X`, held: a type variable's value where `X` is a stand-in.
impl<R: Runtime, X: DeclType> DeclType for Uniform<R, X> {
    fn decl_ty(names: &Interner) -> DeclTy {
        X::decl_ty(names)
    }
}

impl<R: Runtime, X: DeclType> ParamMode for Uniform<R, X> {
    type Mode = ByUniform<X>;

    fn decl(names: &Interner) -> DeclTy {
        X::decl_ty(names)
    }
}

/// Each tuple: its parts' declared types, and a value of it taken by value.
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

        impl<Class, $($p: Part<Class = Class>),+> Part for ($($p,)+)
        where
            ($($p::Of,)+): DeclType,
        {
            type Class = Class;
            type Of = ($($p::Of,)+);
        }

        impl<Class, $($p: Part<Class = Class>),+> ParamMode for ($($p,)+)
        where
            ($($p::Of,)+): DeclType,
            Class: Passed<($($p::Of,)+)>,
        {
            type Mode = <Class as Passed<($($p::Of,)+)>>::Mode;

            fn decl(names: &Interner) -> DeclTy {
                <Class as Passed<($($p::Of,)+)>>::decl(names)
            }
        }
    )+};
}

tuples!(decl_tuples);

impl DeclType for str {
    fn decl_ty(_: &Interner) -> DeclTy {
        DeclTy::Str
    }
}

impl<R: Runtime, X: DeclType> DeclType for [Uniform<R, X>] {
    fn decl_ty(names: &Interner) -> DeclTy {
        DeclTy::Slice(Box::new(X::decl_ty(names)))
    }
}

/// A reference: a `Ref` to its target, as a parameter, a callback's lent
/// parameter, or a result that borrows what a parameter lent.
impl<T: ?Sized + DeclType> DeclType for &T {
    fn decl_ty(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: false, target: Box::new(T::decl_ty(names)) }
    }
}

impl<T: ?Sized + DeclType> DeclType for &mut T {
    fn decl_ty(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: true, target: Box::new(T::decl_ty(names)) }
    }
}

/// A lent parameter's mode, by what it lends.
macro_rules! lent {
    ($([$($g:tt)*] $ty:ty => $mode:ty;)+) => {$(
        impl<$($g)*> ParamMode for $ty {
            type Mode = $mode;

            fn decl(names: &Interner) -> DeclTy {
                <Self as DeclType>::decl_ty(names)
            }
        }
    )+};
}

lent! {
    ['a, T: DeclType] &'a T => ByRef<T>;
    ['a, T: DeclType] &'a mut T => ByMut<T>;
    ['a, R: Runtime, X: DeclType] &'a [Uniform<R, X>] => BySlice<X>;
    ['a, R: Runtime, X: DeclType] &'a mut [Uniform<R, X>] => BySliceMut<X>;
    ['a] &'a str => ByStr;
}

/// A callback's lent parameter is a reference to its target.
impl<T: ?Sized + DeclType> DeclType for Lent<T> {
    fn decl_ty(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: false, target: Box::new(T::decl_ty(names)) }
    }
}

impl<T: ?Sized + DeclType> DeclType for LentMut<T> {
    fn decl_ty(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: true, target: Box::new(T::decl_ty(names)) }
    }
}

/// A function type whose effect is the declaration's N-th effect variable
/// (types §2). Its values are held as `Uniform<R, Callable<..>>`, taken by
/// value or lent as any held value is.
impl<A, U, const N: u32> DeclType for Callable<'_, A, U, EffectNth<N>>
where
    A: DeclParts,
    U: DeclType,
{
    fn decl_ty(names: &Interner) -> DeclTy {
        DeclTy::Fn { params: A::decl_parts(names), result: Box::new(U::decl_ty(names)), effect: DeclEffect::Var(N) }
    }
}

#[cfg(test)]
mod tests;
