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

/// # Safety
/// Only this crate implements it, for the built-in types, and the derive, for
/// an extern-defined type it signs, with `Crosses` beside it: an opaque type
/// is given, taken and lent at itself, and a transparent structured type is
/// composed, taken apart and projected at each part's own type.
///
/// A `Crosses` written by hand, without the seal, does not compile:
///
/// ```compile_fail,E0277
/// use acvus_extern::Settled;
/// use acvus_extern::runtime::{Crosses, Runtime};
/// use acvus_extern::{LayoutFact, Structural, StructuralFacts};
///
/// struct HandWritten;
///
/// unsafe impl Structural for HandWritten {
///     type Canon = HandWritten;
///     const FACTS: StructuralFacts = StructuralFacts { layout: LayoutFact::OwnCanonical, generics: &[] };
/// }
///
/// impl Crosses for HandWritten {
///     type Lent<'c, R: Runtime> = () where Self: 'c;
///     type LentMut<'c, R: Runtime> = () where Self: 'c;
///
///     fn give<R: Runtime>(self, _: &R, _: Settled<'_>) -> R::Value {
///         unimplemented!()
///     }
///     fn take<R: Runtime>(_: R::Value, _: &R, _: Settled<'_>) -> Self {
///         unimplemented!()
///     }
///     fn lent<'c, R: Runtime>(_: &'c R, _: &'c R::Value, _: Settled<'c>) {}
///     fn lent_mut<'c, R: Runtime>(_: &'c R, _: &'c mut R::Value, _: Settled<'c>) {}
/// }
/// ```
pub unsafe trait Crossing {}

use crate::boundary::uniform::Uniform;
use crate::callback::{Lent, LentMut, Opt, Res, Tup};
use crate::runtime::{Crosses, Runtime};

/// The built-in types this crate gives `Crosses`.
macro_rules! built_in_crossings {
    ($($t:ty),+ $(,)?) => {$(
        // SAFETY: `crosses` implements `Crosses` for it, opaque.
        unsafe impl Crossing for $t {}
    )+};
}

built_in_crossings!(i8, i16, i32, i64, u8, u16, u32, u64, f64, bool, char, String, ());

// SAFETY: `crosses` implements `Crosses` for it, opaque.
unsafe impl<T> Crossing for Vec<T> {}
// SAFETY: `crosses` implements `Crosses` for it, opaque: the value it holds
// is given as the Rust value it is.
unsafe impl<R: Runtime, X> Crossing for Uniform<R, X> {}
// SAFETY: `crosses` implements `Crosses` for it, transparent: composed and
// projected at its payload's own type.
unsafe impl<T> Crossing for Option<T> {}
// SAFETY: as for `Option`, at each payload's own type.
unsafe impl<T, E> Crossing for Result<T, E> {}

/// Tuples of parts up to twelve.
macro_rules! tuple_crossings {
    ($(($($p:ident),+))+) => {$(
        // SAFETY: `crosses` implements `Crosses` for it, transparent:
        // composed and projected at each part's own type, in order.
        unsafe impl<$($p),+> Crossing for ($($p,)+) {}
    )+};
}

tuple_crossings! {
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

/// The crossing of an opaque type: given and taken whole at itself, and lent
/// as a guard. The `Crosses` items of the crate's opaque types, and those the
/// derive emits, are made of these; each needs the entry's `Settled`, so none
/// is reached outside a call.
pub mod opaque {
    use crate::boundary::entry::Settled;
    use crate::guard::{Guard, GuardMut};
    use crate::runtime::{Crosses, Runtime};

    /// An opaque part lent shared in a projection.
    pub type Lent<'c, R, T> = Guard<'c, <R as Runtime>::Shared<'c, T>>;
    /// An opaque part lent exclusively in a projection.
    pub type LentMut<'c, R, T> = GuardMut<'c, <R as Runtime>::Exclusive<'c, T>>;

    pub fn give<T: Crosses, R: Runtime>(value: T, rt: &R, at: Settled<'_>) -> R::Value {
        rt.give(value, at)
    }

    pub fn take<T: Crosses, R: Runtime>(v: R::Value, rt: &R, at: Settled<'_>) -> T {
        rt.take(v, at)
    }

    pub fn lent<'c, T: Crosses + 'c, R: Runtime>(rt: &'c R, place: &'c R::Value, at: Settled<'c>) -> Lent<'c, R, T> {
        Guard::of_the_entry(rt.shared(place, at))
    }

    pub fn lent_mut<'c, T: Crosses + 'c, R: Runtime>(
        rt: &'c R,
        place: &'c mut R::Value,
        at: Settled<'c>,
    ) -> LentMut<'c, R, T> {
        GuardMut::of_the_entry(rt.exclusive(place, at))
    }
}

// SAFETY: this crate's `Projected` for `Option` reads its tag by the order of
// the variants' names, `None` 0 and `Some` 1, and its payload at `T`.
unsafe impl<T> ShapeViews for Option<T> {}
// SAFETY: as for `Option`: `Err` 0 and `Ok` 1, the payloads at `T` and `E`.
unsafe impl<T, E> ShapeViews for Result<T, E> {}

/// Tuples of parts up to twelve.
macro_rules! tuple_views {
    ($(($($p:ident),+))+) => {$(
        // SAFETY: this crate's `Projected` for a tuple reads each part at its
        // own type, in order.
        unsafe impl<$($p),+> ShapeViews for ($($p,)+) {}
    )+};
}

tuple_views! {
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
