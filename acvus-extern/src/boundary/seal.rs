//! The seals (safety §1): one `unsafe` trait per meaning, at a hidden path.
//! Its implementations are the ones written here and by the macros; any other
//! is asserted with an `unsafe` token where it is written.

use crate::boundary::entry::mode::{ByInstance, ByInstanceOf};
use crate::boundary::loaned::{Mut, Ref};
use crate::boundary::uniform::Uniform;
use crate::runtime::{MutKey, RefKey, Runtime};
use crate::callback::Func;

/// The seal of a type's crossing: every trait that gives, takes, lends or
/// reads a value at a type (`Open`, `Give`, `Referent`, `Opaque`, `Primitive`,
/// `Element`, `Boxed`, a requirement's mode). A constructor is sealed where its parts
/// are, so no crossing of one is written over an unsealed part. A runtime trusts its own typed reads, so a crossing
/// written by hand could read a value at a type it is not.
///
/// # Safety
/// Only this crate implements it, for the language's types and its own
/// crossing's, and the derive, for the extern type it declares: each gives,
/// takes, lends and reads a value by its own kind's operations, at itself, and
/// its parts only by their own crossings.
///
/// A crossing written by hand, without the seal, does not compile:
///
/// ```compile_fail,E0277
/// use acvus_extern::Settled;
/// use acvus_extern::decl::{DeclTy, Interner};
/// use acvus_extern::runtime::{Form, Runtime};
///
/// struct HandWritten;
///
/// impl<'l, R: Runtime> Form<'l, R> for HandWritten {
///     fn decl(_: &Interner) -> DeclTy {
///         DeclTy::Unit
///     }
///     fn open(_: &'l R, _: R::Value, _: Settled<'l>) -> Self {
///         HandWritten
///     }
///     fn give(self, _: &'l R, _: Settled<'l>) -> R::Value {
///         unimplemented!()
///     }
/// }
/// ```
pub unsafe trait Crossing {}

macro_rules! crossing {
    ($([$($g:tt)*] $t:ty;)+) => {$(
        // SAFETY: this crate's crossing of the type.
        unsafe impl<$($g)*> Crossing for $t {}
    )+};
}

crossing! {
    [] String;
    [] str;
    [T: Crossing] Option<T>;
    [T: Crossing, E: Crossing] Result<T, E>;
    [R: Runtime, X] Uniform<R, X>;
    [T: Crossing] [T];
    ['a, X: ?Sized + Crossing] &'a X;
    ['a, X: ?Sized + Crossing] &'a mut X;
    [X: ?Sized + Crossing] RefKey<X>;
    [X: ?Sized + Crossing] MutKey<X>;
    ['l, R: Runtime, X: ?Sized + Crossing] Ref<'l, R, X>;
    ['l, R: Runtime, X: ?Sized + Crossing] Mut<'l, R, X>;
    [A, U, E] Func<A, U, E>;
    [G: Families, Ts] ByInstanceOf<G, Ts>;
    [G: Families, Ts] ByInstance<G, Ts>;
}

macro_rules! crossing_primitives {
    ($($t:ty => $decl:expr),+ $(,)?) => {
        crossing! { $([] $t;)+ }
    };
}

primitives!(crossing_primitives);

macro_rules! crossing_tuples {
    ($($n:literal ($($p:ident $v:ident $i:tt),+))+) => {
        crossing! { $([$($p: Crossing),+] ($($p,)+);)+ }
    };
}

tuples!(crossing_tuples);

/// The seal of a declaration's dispatch (`ExternFn`, `AwaitedFn`,
/// `Declared`).
///
/// # Safety
/// Only `#[extern_fn]` implements it, for the function it declares: the
/// dispatch opens each argument at the Rust type of the declaration's parameter at
/// that position, at the parameter's own type, and the declaration states that
/// parameter's type.
pub unsafe trait ExternFns {}

/// The seal of a generic function's family (`GenericDeclared`,
/// `GenericFunction`): what a requirer's call of an impl is made by. Its
/// signature at each choice is `FamilyAt`, an `unsafe` trait of its own.
///
/// # Safety
/// Only `#[generic_fn]` implements it, for the generic function it declares:
/// the family's path is that generic function's, and how it takes its
/// receiver is read off the one signature it declares.
pub unsafe trait Families {}
