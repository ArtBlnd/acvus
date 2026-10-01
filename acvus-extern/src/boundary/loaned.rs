//! A retained loan, a carrier: the runtime's reference value an argument lent,
//! taken as it is and kept in a value that outlives the call it was lent in,
//! opened only within a call (K12, limitations §3). The checker holds the loan
//! while that value is live, as the declaration's flows say; the value holds
//! no Rust reference.

use std::marker::PhantomData;

use crate::boundary::entry::Settled;
use crate::boundary::signed::{GenericParam, LayoutFact, Structural, StructuralFacts};
use crate::context::Opens;
use crate::runtime::{Crosses, Runtime};

/// A shared loan on storage of type `T`, retained as a reference value. `'a`
/// is the checker's loan, read by the declaration's flows; no Rust reference
/// lives in it.
pub struct Loaned<'a, T, R: Runtime> {
    reference: R::Value,
    _loan: PhantomData<(&'a (), fn() -> T)>,
}

/// An exclusive loan, retained as a reference value.
pub struct LoanedMut<'a, T, R: Runtime> {
    reference: R::Value,
    _loan: PhantomData<(&'a mut (), fn() -> T)>,
}

impl<T, R: Runtime> Loaned<'_, T, R> {
    /// The reference value the argument passed, taken as it is.
    pub(crate) fn of_the_entry(reference: R::Value) -> Self {
        Loaned { reference, _loan: PhantomData }
    }
}

impl<T, R: Runtime> LoanedMut<'_, T, R> {
    pub(crate) fn of_the_entry(reference: R::Value) -> Self {
        LoanedMut { reference, _loan: PhantomData }
    }
}

impl<'a, T: Crosses<R> + 'a, R: Runtime> Loaned<'a, T, R> {
    /// Opens the loan as `T`'s view, for no longer than the call the context
    /// is of: the runtime is borrowed for the view through the context. A
    /// context at `'a`, `'a` lying in its call, gives the view for as long as
    /// the loan; another, for less. A shared loan's views may be held
    /// together. A view at the loan's lifetime is had only through a context
    /// at it:
    ///
    /// ```compile_fail,E0621
    /// use acvus_extern::Loaned;
    /// use acvus_extern::context::Returning;
    /// use acvus_extern::runtime::Runtime;
    ///
    /// fn text<'a, R: Runtime>(cx: &Returning<'_, R>, text: &Loaned<'a, String, R>) -> &'a String {
    ///     text.open(cx)
    /// }
    /// ```
    pub fn open(&self, cx: &impl Opens<'a, R>) -> T::Open<'a> {
        T::open_ref(cx.runtime(), self.reference, Settled::of_the_entry())
    }
}

impl<T: Crosses<R>, R: Runtime> LoanedMut<'_, T, R> {
    /// Opens the loan as `T`'s exclusive view for as long as `self` is
    /// borrowed, within the call the context is of.
    pub fn open<'s>(&'s mut self, cx: &impl Opens<'s, R>) -> T::OpenMut<'s>
    where
        T: 's,
    {
        T::open_ref_mut(cx.runtime(), self.reference, Settled::of_the_entry())
    }
}

// SAFETY: a loan is one `R::Value` and phantoms at every `'a` and `T`, so it and
// its canonical form, the same loan at `'static` over `T`'s canonical form,
// have one size, alignment, field offset and release (none: the value is
// `Copy`). It reaches `T` only as a phantom, and nothing in it is
// interior-mutable. `FACTS` states its lifetime and its type parameter, as the
// checker's flows read them; the runtime `R` is not declared to the checker.
unsafe impl<'a, T: Structural, R: Runtime> Structural for Loaned<'a, T, R> {
    type Canon = Loaned<'static, T::Canon, R>;

    const FACTS: StructuralFacts = StructuralFacts {
        layout: LayoutFact::of_parts(&[T::FACTS.layout]),
        generics: &[GenericParam::Lifetime, GenericParam::Type],
    };
}

// SAFETY: as `Loaned`'s.
unsafe impl<'a, T: Structural, R: Runtime> Structural for LoanedMut<'a, T, R> {
    type Canon = LoanedMut<'static, T::Canon, R>;

    const FACTS: StructuralFacts = StructuralFacts {
        layout: LayoutFact::of_parts(&[T::FACTS.layout]),
        generics: &[GenericParam::Lifetime, GenericParam::Type],
    };
}
