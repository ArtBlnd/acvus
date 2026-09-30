//! Rust types state their declared types and their parameter modes (macros
//! §4): the macros ask these traits and never read a type's tokens for its
//! meaning.

use acvus_mir::ty::IntTy;
use acvus_utils::Interner;

use crate::boundary::entry::mode::{ByCallback, ByMut, ByRef, BySlice, BySliceMut, ByStr, ByUniform, ByValue};
use crate::boundary::uniform::Uniform;
use crate::callback::{Callback, Lent, LentMut};
use crate::decl::{DeclEffect, DeclTy};
use crate::runtime::Runtime;

/// Names a declared type needs, interned.
pub struct DeclCx<'i> {
    pub names: &'i Interner,
}

/// A Rust type's declared type. Implemented by this crate for the language's
/// own types and the crossing's (`Uniform`, a stand-in, a reference, a
/// callback), and by the derive for an extern-defined type.
pub trait DeclType {
    fn decl_ty(cx: &mut DeclCx<'_>) -> DeclTy;
}

/// The stand-in for a declaration's N-th type parameter, at which a
/// declaration is both built and dispatched: its declared type is
/// `DeclTy::Param(N)`, and its value is `Uniform<R, Nth<N>>`.
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
/// parameter carries. Only the stand-ins are one.
pub trait Var<K> {}

impl<const N: u32> Var<kind::Type> for Nth<N> {}
impl<const N: u32> Var<kind::Effect> for EffectNth<N> {}
impl<const N: u32> Var<kind::Identity> for IdentityNth<N> {}

/// A handler parameter's Rust type picks the mode the entry opens it by
/// (macros §2): a value, a type variable's `Uniform`, `&T`, `&mut T`, a slice
/// of `Uniform`s, `&str`, or a `Callback`.
pub trait ParamMode {
    /// The mode, as `crate::mode` names it, at a runtime `R`.
    type Mode<R: Runtime>;
    /// What the parameter is in the declaration.
    fn decl(cx: &mut DeclCx<'_>) -> ParamDecl;
}

/// What a handler's parameter is in its declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamDecl {
    /// A declared parameter of this type, taking an argument position.
    Argument(DeclTy),
}

mod parts {
    use super::{DeclCx, DeclTy};

    /// A tuple's parts, each a declared type, in order: a tuple's own, or a
    /// callback's parameters.
    pub trait DeclParts {
        fn decl_parts(cx: &mut DeclCx<'_>) -> Vec<DeclTy>;
    }
}

use parts::DeclParts;

/// The language's own types taken by value, each its declared type.
macro_rules! by_value {
    ($([$($g:tt)*] $ty:ty => |$cx:pat_param| $decl:expr;)+) => {$(
        impl<$($g)*> DeclType for $ty {
            fn decl_ty($cx: &mut DeclCx<'_>) -> DeclTy {
                $decl
            }
        }

        impl<$($g)*> ParamMode for $ty {
            type Mode<Rt: Runtime> = ByValue<Self>;

            fn decl(cx: &mut DeclCx<'_>) -> ParamDecl {
                ParamDecl::Argument(<Self as DeclType>::decl_ty(cx))
            }
        }
    )+};
}

by_value! {
    [] i8 => |_| DeclTy::Int(IntTy::I8);
    [] i16 => |_| DeclTy::Int(IntTy::I16);
    [] i32 => |_| DeclTy::Int(IntTy::I32);
    [] i64 => |_| DeclTy::Int(IntTy::I64);
    [] u8 => |_| DeclTy::Int(IntTy::U8);
    [] u16 => |_| DeclTy::Int(IntTy::U16);
    [] u32 => |_| DeclTy::Int(IntTy::U32);
    [] u64 => |_| DeclTy::Int(IntTy::U64);
    [] f64 => |_| DeclTy::F64;
    [] bool => |_| DeclTy::Bool;
    [] char => |_| DeclTy::Char;
    [] () => |_| DeclTy::Unit;
    [] String => |_| DeclTy::String;
    [T: DeclType] Option<T> => |cx| DeclTy::Option(Box::new(T::decl_ty(cx)));
    [T: DeclType, E: DeclType] Result<T, E> => |cx| DeclTy::Result(Box::new(T::decl_ty(cx)), Box::new(E::decl_ty(cx)));
}

impl<const N: u32> DeclType for Nth<N> {
    fn decl_ty(_: &mut DeclCx<'_>) -> DeclTy {
        DeclTy::Param(N)
    }
}

/// A value at `X`, held: a type variable's value where `X` is a stand-in.
impl<R: Runtime, X: DeclType> DeclType for Uniform<R, X> {
    fn decl_ty(cx: &mut DeclCx<'_>) -> DeclTy {
        X::decl_ty(cx)
    }
}

impl<R: Runtime, X: DeclType> ParamMode for Uniform<R, X> {
    type Mode<Rt: Runtime> = ByUniform<X>;

    fn decl(cx: &mut DeclCx<'_>) -> ParamDecl {
        ParamDecl::Argument(X::decl_ty(cx))
    }
}

/// Tuples up to twelve parts, the arity the crossing's tuples stop at.
macro_rules! tuples {
    ($(($($p:ident),+))+) => {$(
        impl<$($p: DeclType),+> DeclParts for ($($p,)+) {
            fn decl_parts(cx: &mut DeclCx<'_>) -> Vec<DeclTy> {
                vec![$($p::decl_ty(cx)),+]
            }
        }

        by_value! {
            [$($p: DeclType),+] ($($p,)+) => |cx| DeclTy::Tuple(<Self as DeclParts>::decl_parts(cx));
        }
    )+};
}

impl DeclParts for () {
    fn decl_parts(_: &mut DeclCx<'_>) -> Vec<DeclTy> {
        Vec::new()
    }
}

tuples! {
    (A)
    (A, B)
    (A, B, C)
    (A, B, C, D)
    (A, B, C, D, E)
    (A, B, C, D, E, F)
    (A, B, C, D, E, F, G)
    (A, B, C, D, E, F, G, H)
    (A, B, C, D, E, F, G, H, I)
    (A, B, C, D, E, F, G, H, I, J)
    (A, B, C, D, E, F, G, H, I, J, K)
    (A, B, C, D, E, F, G, H, I, J, K, L)
}

impl DeclType for str {
    fn decl_ty(_: &mut DeclCx<'_>) -> DeclTy {
        DeclTy::Str
    }
}

impl<R: Runtime, X: DeclType> DeclType for [Uniform<R, X>] {
    fn decl_ty(cx: &mut DeclCx<'_>) -> DeclTy {
        DeclTy::Slice(Box::new(X::decl_ty(cx)))
    }
}

/// A reference: a `Ref` to its target, as a parameter, a callback's lent
/// parameter, or a result that borrows what a parameter lent.
impl<T: ?Sized + DeclType> DeclType for &T {
    fn decl_ty(cx: &mut DeclCx<'_>) -> DeclTy {
        DeclTy::Ref { mutable: false, target: Box::new(T::decl_ty(cx)) }
    }
}

impl<T: ?Sized + DeclType> DeclType for &mut T {
    fn decl_ty(cx: &mut DeclCx<'_>) -> DeclTy {
        DeclTy::Ref { mutable: true, target: Box::new(T::decl_ty(cx)) }
    }
}

/// A lent parameter's mode, by what it lends.
macro_rules! lent {
    ($([$($g:tt)*] $ty:ty => $mode:ty;)+) => {$(
        impl<$($g)*> ParamMode for $ty {
            type Mode<Rt: Runtime> = $mode;

            fn decl(cx: &mut DeclCx<'_>) -> ParamDecl {
                ParamDecl::Argument(<Self as DeclType>::decl_ty(cx))
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
    fn decl_ty(cx: &mut DeclCx<'_>) -> DeclTy {
        DeclTy::Ref { mutable: false, target: Box::new(T::decl_ty(cx)) }
    }
}

impl<T: ?Sized + DeclType> DeclType for LentMut<T> {
    fn decl_ty(cx: &mut DeclCx<'_>) -> DeclTy {
        DeclTy::Ref { mutable: true, target: Box::new(T::decl_ty(cx)) }
    }
}

/// A callback whose effect is the declaration's N-th effect variable.
impl<A, U, R, const N: u32> DeclType for Callback<'_, '_, A, U, EffectNth<N>, R>
where
    A: DeclParts,
    U: DeclType,
    R: Runtime,
{
    fn decl_ty(cx: &mut DeclCx<'_>) -> DeclTy {
        DeclTy::Fn { params: A::decl_parts(cx), result: Box::new(U::decl_ty(cx)), effect: DeclEffect::Var(N) }
    }
}

impl<A, U, R, const N: u32> ParamMode for Callback<'_, '_, A, U, EffectNth<N>, R>
where
    A: DeclParts,
    U: DeclType,
    R: Runtime,
{
    type Mode<Rt: Runtime> = ByCallback<A, U, EffectNth<N>>;

    fn decl(cx: &mut DeclCx<'_>) -> ParamDecl {
        ParamDecl::Argument(<Self as DeclType>::decl_ty(cx))
    }
}

#[cfg(test)]
mod tests;
