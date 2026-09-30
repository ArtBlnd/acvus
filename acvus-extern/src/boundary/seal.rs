//! The seals of the crossing traits whose methods choose the type of a typed
//! read (safety §1). A runtime trusts its own typed reads, so a hand-written
//! implementation that read a part at a type other than the shape's, or a
//! loan form that read a result at a type other than the callback's, would be
//! undefined behaviour reached from safe code. Each seal is an `unsafe` trait
//! at a hidden path: its implementations are the ones named here and the ones
//! `shape!` emits, and any other is asserted with an `unsafe` token where it
//! is written.

/// # Safety
/// Only `shape!` implements it, for the shape it declares, and the same
/// expansion implements `Projected` for that shape: `AGGREGATE` is the
/// shape's aggregate, and each part is read, lent or projected at the type
/// the shape gives it, in the order of its name's text (compiler: types §2).
///
/// A `Projected` written by hand, without the seal, does not compile:
///
/// ```compile_fail,E0277
/// use acvus_extern::runtime::Runtime;
/// use acvus_extern::shape::{Aggregate, LentParts, LentPartsMut, OwnedParts, Projected};
///
/// struct HandWritten;
///
/// impl<R: Runtime> Projected<R> for HandWritten {
///     type Ref<'a> = ();
///     type Mut<'a> = ();
///     type Owned = ();
///
///     const AGGREGATE: Aggregate = Aggregate::Tuple;
///
///     fn project<'c>(_: LentParts<'c, R>) {}
///     fn project_mut<'c>(_: LentPartsMut<'c, R>) {}
///     fn owned(_: OwnedParts<R>) {}
/// }
/// ```
pub unsafe trait ShapeViews {}

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

use crate::callback::{Lent, LentMut, Opt, Res, Tup};
use crate::runtime::Crosses;

// SAFETY: a crossing type is its own form: it is given and taken at itself.
unsafe impl<T> LoanForms for T where T: Crosses {}
// SAFETY: `Lent<T>` lends, moves and reads a reference to a `T`.
unsafe impl<T> LoanForms for Lent<T> where T: ?Sized {}
// SAFETY: as `Lent`'s, exclusively.
unsafe impl<T> LoanForms for LentMut<T> where T: ?Sized {}
// SAFETY: `Opt<F>` is `Option`'s variants, its payload at `F`.
unsafe impl<F> LoanForms for Opt<F> where F: LoanForms {}
// SAFETY: `Res<F, G>` is `Result`'s variants, its payloads at `F` and `G`.
unsafe impl<F, G> LoanForms for Res<F, G>
where
    F: LoanForms,
    G: LoanForms,
{
}

/// Tuples of forms up to twelve, the arity the crossing's tuples stop at.
macro_rules! tuple_forms {
    ($(($($f:ident),*))+) => {$(
        // SAFETY: a tuple of forms is a tuple of its parts, each at its form.
        unsafe impl<$($f),*> LoanForms for Tup<($($f,)*)> where $($f: LoanForms),* {}
    )+};
}

tuple_forms! {
    ()
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
