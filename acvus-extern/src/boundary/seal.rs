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
/// impl LoanForm for HandWritten {
///     type At<'l> = ();
///
///     const LENT: Option<Mutability> = None;
///
///     fn lend<'l, R: Runtime>(_: (), _: &'l R, _: Settled<'l>) -> R::Value {
///         unimplemented!()
///     }
///     fn give_moved<'a, R: Runtime>(_: (), _: &'a R, _: Settled<'a>) -> R::Value {
///         unimplemented!()
///     }
///     fn read<'l, R: Runtime>(_: R::Value, _: &'l R, _: Settled<'l>) {}
/// }
/// ```
pub unsafe trait LoanForms {}

use crate::callback::{Lent, LentMut};
use crate::runtime::Crosses;

// SAFETY: a crossing type is its own form: it is given and taken at itself.
unsafe impl<T> LoanForms for T where T: Crosses {}
// SAFETY: `Lent<T>` lends, moves and reads a reference to a `T`.
unsafe impl<T> LoanForms for Lent<T> where T: ?Sized {}
// SAFETY: as `Lent`'s, exclusively.
unsafe impl<T> LoanForms for LentMut<T> where T: ?Sized {}
