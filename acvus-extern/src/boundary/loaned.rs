//! A reference value (runtime contract §2, References): a value's place the
//! runtime named, kept as the runtime's value, `Ref` shared and `Mut`
//! exclusive. It is what a reference is wherever it is not a view within a
//! call: what goes out of a call, and a loan a value keeps past the call it
//! was lent in (K12, limitations §1). The runtime makes one only from a box's
//! place, so its loan `'l` is the runtime's borrow of that box; and it is
//! opened only within a call (crossing §1).

use std::marker::PhantomData;
use std::ops::Range;

use acvus_utils::Interner;

use crate::boundary::entry::Settled;
use crate::boundary::seal::Crossing;
use crate::boundary::signed::{GenericParam, LayoutFact, Structural, StructuralFacts};
use crate::boundary::uniform::Boxed;
use crate::decl::{DeclTy, DeclType};
use crate::executor::Executor;
use crate::runtime::{Give, MutKey, Open, Place, RefKey, RefKind, Referent, Runtime, ViewKind};

/// A shared reference value at the loan `'l`, naming a place of `X`.
pub struct Ref<'l, R: Runtime, X: ?Sized> {
    reference: R::Value,
    _loan: PhantomData<(&'l (), fn() -> *const X)>,
}

/// An exclusive reference value at the loan `'l`: invariant in `X`, as
/// `&'l mut X` is, so nothing of a shorter loan is written through it.
pub struct Mut<'l, R: Runtime, X: ?Sized> {
    reference: R::Value,
    _loan: PhantomData<(&'l mut (), fn(*const X) -> *const X)>,
}

impl<R: Runtime, X: ?Sized> Clone for Ref<'_, R, X> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime, X: ?Sized> Copy for Ref<'_, R, X> {}

impl<'l, R: Runtime, X: ?Sized> Ref<'l, R, X> {
    /// A reference to what a box holds, made by the runtime from the box's
    /// place, for as long as the box is borrowed.
    pub fn of<B: Boxed<R, View = X>>(cx: &Executor<'l, R>, held: &'l B) -> Self {
        Ref { reference: <R::Ref as RefKind<R>>::of(cx.runtime(), held.place(), Settled::of_the_entry()), _loan: PhantomData }
    }

    /// The place, viewed as `X` for as long as the loan, within the call the
    /// executor is of. A view at the loan's lifetime is had only through an
    /// executor at it:
    ///
    /// ```compile_fail,E0621
    /// use acvus_extern::Ref;
    /// use acvus_extern::executor::Executor;
    /// use acvus_extern::runtime::Runtime;
    ///
    /// fn text<'a, R: Runtime>(cx: &Executor<'_, R>, text: &Ref<'a, R, String>) -> &'a String {
    ///     text.get(cx)
    /// }
    /// ```
    pub fn get(&self, cx: &Executor<'l, R>) -> &'l X
    where
        X: Referent<R> + 'l,
    {
        X::view(cx.runtime(), self.reference, Settled::of_the_entry())
    }
}

impl<'l, R: Runtime, T> Ref<'l, R, [T]> {
    /// `range` of this run, at the same loan.
    pub fn narrow(self, cx: &Executor<'l, R>, range: Range<usize>) -> Self {
        let reference =
            <R::View as ViewKind<R>>::narrow(cx.runtime(), self.reference, range.start, range.end, Settled::of_the_entry());
        Ref { reference, _loan: PhantomData }
    }
}

impl<'l, R: Runtime> Ref<'l, R, str> {
    /// `range` of the text a boxed `String` holds, for as long as the box is
    /// borrowed. Out of the text's bounds or off a character's boundary, it
    /// panics, as slicing does.
    pub fn text<B: Boxed<R, View = String>>(cx: &Executor<'l, R>, held: &'l B, range: Range<usize>) -> Self {
        let reference =
            <R::View as ViewKind<R>>::text(cx.runtime(), held.place(), range.start, range.end, Settled::of_the_entry());
        Ref { reference, _loan: PhantomData }
    }

    /// `range` of this text, at the same loan.
    pub fn narrow(self, cx: &Executor<'l, R>, range: Range<usize>) -> Self {
        let reference =
            <R::View as ViewKind<R>>::narrow(cx.runtime(), self.reference, range.start, range.end, Settled::of_the_entry());
        Ref { reference, _loan: PhantomData }
    }
}

impl<'l, R: Runtime, X: ?Sized> Mut<'l, R, X> {
    /// An exclusive reference to what a box holds, made by the runtime from
    /// the box's place, for as long as the box is borrowed.
    pub fn of<B: Boxed<R, View = X>>(cx: &Executor<'l, R>, held: &'l mut B) -> Self {
        Mut { reference: <R::Ref as RefKind<R>>::of_mut(cx.runtime(), held.place_mut(), Settled::of_the_entry()), _loan: PhantomData }
    }

    /// The place, viewed as `X` exclusively for as long as `self` is
    /// borrowed, within the call the executor is of.
    pub fn get<'s>(&'s mut self, cx: &Executor<'s, R>) -> &'s mut X
    where
        X: Referent<R> + 's,
    {
        X::view_mut(cx.runtime(), self.reference, Settled::of_the_entry())
    }

    /// A shorter loan of the same place, while `self` is borrowed.
    pub fn reborrow<'s>(&'s mut self) -> Mut<'s, R, X> {
        Mut { reference: self.reference, _loan: PhantomData }
    }
}

impl<'l, R: Runtime, T> Mut<'l, R, [T]> {
    /// `range` of this run, at the same loan.
    pub fn narrow(self, cx: &Executor<'l, R>, range: Range<usize>) -> Self {
        let reference =
            <R::View as ViewKind<R>>::narrow(cx.runtime(), self.reference, range.start, range.end, Settled::of_the_entry());
        Mut { reference, _loan: PhantomData }
    }
}

impl<'l, R: Runtime> Mut<'l, R, str> {
    /// `range` of the text a boxed `String` holds, exclusively for as long as
    /// the box is borrowed. Out of the text's bounds or off a character's
    /// boundary, it panics, as slicing does.
    pub fn text<B: Boxed<R, View = String>>(cx: &Executor<'l, R>, held: &'l mut B, range: Range<usize>) -> Self {
        let reference = <R::View as ViewKind<R>>::text_mut(
            cx.runtime(),
            held.place_mut(),
            range.start,
            range.end,
            Settled::of_the_entry(),
        );
        Mut { reference, _loan: PhantomData }
    }

    /// `range` of this text, at the same loan.
    pub fn narrow(self, cx: &Executor<'l, R>, range: Range<usize>) -> Self {
        let reference =
            <R::View as ViewKind<R>>::narrow(cx.runtime(), self.reference, range.start, range.end, Settled::of_the_entry());
        Mut { reference, _loan: PhantomData }
    }
}

impl<'l, R: Runtime, X: ?Sized + DeclType + Crossing> Open<'l, R> for Ref<'l, R, X> {
    fn decl(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: false, target: Box::new(X::decl_ty(names)) }
    }

    fn open(_rt: &'l R, v: R::Value, _at: Settled<'l>) -> Self {
        Ref { reference: v, _loan: PhantomData }
    }
}

impl<'l, R: Runtime, X: ?Sized + DeclType + Crossing> Give<'l, R> for Ref<'l, R, X> {
    fn decl(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: false, target: Box::new(X::decl_ty(names)) }
    }

    fn give(self, _rt: &'l R, _at: Settled<'l>) -> R::Value {
        self.reference
    }
}

impl<'l, R: Runtime, X: ?Sized + DeclType + Crossing> Open<'l, R> for Mut<'l, R, X> {
    fn decl(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: true, target: Box::new(X::decl_ty(names)) }
    }

    fn open(_rt: &'l R, v: R::Value, _at: Settled<'l>) -> Self {
        Mut { reference: v, _loan: PhantomData }
    }
}

impl<'l, R: Runtime, X: ?Sized + DeclType + Crossing> Give<'l, R> for Mut<'l, R, X> {
    fn decl(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: true, target: Box::new(X::decl_ty(names)) }
    }

    fn give(self, _rt: &'l R, _at: Settled<'l>) -> R::Value {
        self.reference
    }
}

impl<'l, R: Runtime, X: ?Sized + Place<'l, R>> Place<'l, R> for Ref<'l, R, X> {
    type Key = RefKey<X::Key>;
}

impl<'l, R: Runtime, X: ?Sized + Place<'l, R>> Place<'l, R> for Mut<'l, R, X> {
    type Key = MutKey<X::Key>;
}

// SAFETY: a reference value is one `R::Value` and phantoms at every `'l` and
// `X`, so it and its canonical form, the same loan at `'static` over `X`'s
// canonical form, have one size, alignment, field offset and release (none:
// the value is `Copy`); its lifetime makes it another type than its
// canonical form. It reaches `X` only as a phantom, and nothing in it is
// interior-mutable. `FACTS` states its lifetime and its type parameter, as
// the checker's flows read them; the runtime `R` is not declared to the
// checker.
unsafe impl<'l, X: Structural, R: Runtime> Structural for Ref<'l, R, X> {
    type Canon = Ref<'static, R, X::Canon>;

    const FACTS: StructuralFacts = StructuralFacts {
        layout: LayoutFact::CanonicalOf,
        generics: &[GenericParam::Lifetime, GenericParam::Type],
    };
}

// SAFETY: as `Ref`'s.
unsafe impl<'l, X: Structural, R: Runtime> Structural for Mut<'l, R, X> {
    type Canon = Mut<'static, R, X::Canon>;

    const FACTS: StructuralFacts = StructuralFacts {
        layout: LayoutFact::CanonicalOf,
        generics: &[GenericParam::Lifetime, GenericParam::Type],
    };
}
