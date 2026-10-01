//! The seals (safety §1): one `unsafe` trait per meaning, at a hidden path.
//! Its implementations are the ones written here and by the macros; any other
//! is asserted with an `unsafe` token where it is written.

use crate::boundary::entry::mode::{
    ByInstance, ByInstanceOf, ByLoaned, ByLoanedMut, ByMut, ByRef, BySlice, BySliceMut, ByStr, ByUniform, ByValue,
};
use crate::boundary::uniform::Uniform;
use crate::callback::{Lent, LentMut};
use crate::context::{Awaited, Returning};
use crate::runtime::Runtime;

/// The seal of a type's crossing: every trait that gives, takes, lends or
/// reads a value at a type (`Crosses`, `Opaque`, `Referent`, `Primitive`,
/// `LoanForm`, a mode, the argument tuples, a context a loan opens in). A
/// runtime trusts its own typed reads, so a crossing written by hand could
/// read a value at a type it is not.
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
/// use acvus_extern::callback::LoanForm;
/// use acvus_extern::runtime::{Mutability, Runtime};
///
/// struct HandWritten;
///
/// impl<R: Runtime> LoanForm<R> for HandWritten {
///     type At<'l> = ();
///
///     const LENT: Option<Mutability> = None;
///
///     fn lend<'l>(_: (), _: &'l R, _: Settled<'l>) -> R::Value {
///         unimplemented!()
///     }
///     fn read<'l>(_: R::Value, _: &'l R, _: Settled<'l>) {}
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
    [T] Option<T>;
    [T, E] Result<T, E>;
    [R: Runtime, X] Uniform<R, X>;
    [R: Runtime, X] [Uniform<R, X>];
    [R: Runtime, X] Vec<Uniform<R, X>>;
    [T: ?Sized] Lent<T>;
    [T: ?Sized] LentMut<T>;
    [T] ByValue<T>;
    [T] ByUniform<T>;
    [T] ByRef<T>;
    [T] ByMut<T>;
    [T] BySlice<T>;
    [T] BySliceMut<T>;
    [] ByStr;
    [T] ByLoaned<T>;
    [T] ByLoanedMut<T>;
    [F, I] ByInstance<F, I>;
    [F, Ts, L] ByInstanceOf<F, Ts, L>;
    [R: Runtime] Returning<'_, R>;
    [R: Runtime] Awaited<'_, R>;
}

macro_rules! crossing_primitives {
    ($($t:ty => $decl:expr),+ $(,)?) => {
        crossing! { $([] $t;)+ }
    };
}

primitives!(crossing_primitives);

macro_rules! crossing_tuples {
    ($($n:literal ($($p:ident $v:ident $i:tt),+))+) => {
        crossing! { $([$($p),+] ($($p,)+);)+ }
    };
}

tuples!(crossing_tuples);

/// The seal of a declaration's dispatch (`ExternFn`, `AwaitedFn`,
/// `Declared`).
///
/// # Safety
/// Only `#[extern_fn]` implements it, for the function it declares: the
/// dispatch opens each argument by the mode of the declaration's parameter at
/// that position, at the parameter's own type, and the declaration states that
/// parameter's type.
pub unsafe trait ExternFns {}
