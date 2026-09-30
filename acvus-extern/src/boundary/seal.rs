//! The seal of the crossing trait whose methods choose the type of a typed
//! read (safety §1). A runtime trusts its own typed reads, so a hand-written
//! loan form that read a result at a type other than the callback's would be
//! undefined behaviour reached from safe code. The seal is an `unsafe` trait
//! at a hidden path: its implementations are the ones named here, and any
//! other is asserted with an `unsafe` token where it is written.

/// # Safety
/// Only this crate implements it, for the loan forms it defines: each form
/// lends, moves and reads a value at the type its `At` names.
///
/// A `LoanForm` written by hand, without the seal, does not compile:
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
///     fn give_moved<'a>(_: (), _: &'a R, _: Settled<'a>) -> R::Value {
///         unimplemented!()
///     }
///     fn read<'l>(_: R::Value, _: &'l R, _: Settled<'l>) {}
/// }
/// ```
pub unsafe trait LoanForms {}

use crate::boundary::uniform::Uniform;
use crate::callback::{Lent, LentMut};
use crate::runtime::Runtime;

// SAFETY: a crossing type is its own form: it is given and taken at itself.
unsafe impl<T> LoanForms for T where T: Crossings {}
// SAFETY: `Lent<T>` lends, moves and reads a reference to a `T`.
unsafe impl<T> LoanForms for Lent<T> where T: ?Sized {}
// SAFETY: as `Lent`'s, exclusively.
unsafe impl<T> LoanForms for LentMut<T> where T: ?Sized {}

/// The seal of `Crosses` and `Opaque`: a crossing makes and takes values with
/// the runtime's operations, which read at the type they are asked for, so a
/// crossing written by hand could read a value at a type it is not.
///
/// # Safety
/// Only this crate implements it, for the language's types, and the derive,
/// for the extern type it declares: each crossing gives and takes a value by
/// its own kind's operations, at itself, and its parts only by their own
/// crossings.
pub unsafe trait Crossings {}

macro_rules! crossings {
    ($([$($g:tt)*] $t:ty;)+) => {$(
        // SAFETY: this crate's crossing of the type, in `crate::crossing`.
        unsafe impl<$($g)*> Crossings for $t {}
    )+};
}

crossings! {
    [] i8; [] i16; [] i32; [] i64; [] u8; [] u16; [] u32; [] u64; [] f64; [] bool; [] char; [] ();
    [] String;
    [T] Option<T>;
    [T, E] Result<T, E>;
    [R: Runtime, X] Uniform<R, X>;
    [A] (A,);
    [A, B] (A, B);
    [A, B, C] (A, B, C);
    [A, B, C, D] (A, B, C, D);
    [A, B, C, D, E] (A, B, C, D, E);
    [A, B, C, D, E, F] (A, B, C, D, E, F);
    [A, B, C, D, E, F, G] (A, B, C, D, E, F, G);
    [A, B, C, D, E, F, G, H] (A, B, C, D, E, F, G, H);
    [A, B, C, D, E, F, G, H, I] (A, B, C, D, E, F, G, H, I);
    [A, B, C, D, E, F, G, H, I, J] (A, B, C, D, E, F, G, H, I, J);
    [A, B, C, D, E, F, G, H, I, J, K] (A, B, C, D, E, F, G, H, I, J, K);
    [A, B, C, D, E, F, G, H, I, J, K, L] (A, B, C, D, E, F, G, H, I, J, K, L);
}

/// # Safety
/// Only `#[extern_fn]` implements it, for the function it declares: the
/// dispatch opens each argument by the mode of the declaration's parameter at
/// that position, at the parameter's own type.
pub unsafe trait ExternFns {}
