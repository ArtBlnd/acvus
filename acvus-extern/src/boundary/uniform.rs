//! A type variable's value: the runtime's uniform value indexed by the
//! variable, owned, and released once by its drop (G7).

use std::marker::PhantomData;

use crate::boundary::entry::Settled;
use crate::boundary::seal::Crossing;
use crate::boundary::signed::{GenericParam, LayoutFact, Structural, StructuralFacts};
use crate::executor::Executor;
use crate::decl::DeclType;
use crate::runtime::Runtime;

/// The permission to release one value, made only by an owning holder's drop.
pub struct Releasing {
    _private: (),
}

/// A runtime value's release. The value is `Copy`; only its owning holder
/// releases it, through a `Releasing` it alone can make.
pub trait Release: Copy {
    fn release(self, releasing: Releasing);
}

/// The uniform index fixed: every `Uniform<R, X>` has `Uniform<R, Never>` as
/// its canonical form.
pub enum Never {}

/// A box as an element of a run: a slice of boxes is the run of their
/// places.
pub trait Element<R: Runtime>: Crossing + Sized {
    #[doc(hidden)]
    fn slice(places: &[R::Value]) -> &[Self];
    #[doc(hidden)]
    fn slice_mut(places: &mut [R::Value]) -> &mut [Self];
}

impl<R: Runtime, X> Element<R> for Uniform<R, X> {
    fn slice(places: &[R::Value]) -> &[Self] {
        // SAFETY: `Uniform` is `repr(transparent)` over the runtime value, so
        // a slice of values is a slice of boxes, of one length.
        unsafe { &*(std::ptr::from_ref(places) as *const [Self]) }
    }

    fn slice_mut(places: &mut [R::Value]) -> &mut [Self] {
        // SAFETY: as `slice`, exclusively.
        unsafe { &mut *(std::ptr::from_mut(places) as *mut [Self]) }
    }
}

/// The uniform value at a type variable `X`. No trait implemented on it has an
/// item that depends on `X` (runtime contract §1).
#[repr(transparent)]
pub struct Uniform<R: Runtime, X> {
    value: R::Value,
    _index: PhantomData<fn() -> X>,
}

impl<R: Runtime, X> Uniform<R, X> {
    /// The value the crossing hands over, owned from here on.
    pub(crate) fn of_the_crossing(value: R::Value) -> Self {
        Uniform { value, _index: PhantomData }
    }

    /// The value handed on to its next owner, unreleased.
    pub(crate) fn into_value(self) -> R::Value {
        let value = self.value;
        std::mem::forget(self);
        value
    }

    /// The value, still owned here, handed to the runtime for an operation
    /// that reads it at `X`.
    pub(crate) fn value(&self) -> R::Value {
        self.value
    }

    /// The value in `place` seen as a `Uniform` at `X`, borrowed.
    pub(crate) fn at_place(place: &R::Value) -> &Self {
        // SAFETY: `Uniform` is `repr(transparent)` over the runtime value.
        unsafe { &*std::ptr::from_ref(place).cast::<Self>() }
    }

    pub(crate) fn at_place_mut(place: &mut R::Value) -> &mut Self {
        // SAFETY: as `at_place`, exclusively.
        unsafe { &mut *std::ptr::from_mut(place).cast::<Self>() }
    }
}

/// What a box holds in place, made, taken and viewed through its kind: a
/// primitive, text, an extern-defined value. Only through the runtime is a
/// box made, opened or taken.
pub trait InPlace<R: Runtime>: Crossing + DeclType + Sized {
    #[doc(hidden)]
    fn give(self, rt: &R, at: Settled<'_>) -> R::Value;
    #[doc(hidden)]
    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self;
    #[doc(hidden)]
    fn shared<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> &'a Self
    where
        Self: 'a;
    #[doc(hidden)]
    fn exclusive<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> &'a mut Self
    where
        Self: 'a;
}

/// A box's index whose values have no Rust view of their own: a type
/// variable's stand-in, the canonical index, a function type, an impl. A box
/// at one is viewed as the box itself.
pub trait Abstract {}

impl Abstract for Never {}

/// A box's index, and what a box at it is: viewed as (`View`), and named in a
/// function type's signature as (`Key`).
pub trait Index<R: Runtime> {
    type View: ?Sized;
    type Key;
}

/// A box, whatever its index: what a reference value is made from, and what
/// it is viewed as through one.
pub trait Boxed<R: Runtime>: Crossing {
    type View: ?Sized;

    #[doc(hidden)]
    fn place(&self) -> &R::Value;
    #[doc(hidden)]
    fn place_mut(&mut self) -> &mut R::Value;
}

impl<R: Runtime, X: Index<R>> Boxed<R> for Uniform<R, X> {
    type View = X::View;

    fn place(&self) -> &R::Value {
        &self.value
    }

    fn place_mut(&mut self) -> &mut R::Value {
        &mut self.value
    }
}

impl<R: Runtime, X> Uniform<R, X> {
    /// `value` boxed at its own type, within the call `cx` belongs to: its
    /// kind makes the runtime's value of it, and the box keeps `X`, so it is
    /// read and given only at `X` and never erased (runtime contract §1). A
    /// type variable's index has no kind of its own:
    ///
    /// ```compile_fail,E0277
    /// use acvus_extern::Uniform;
    /// use acvus_extern::executor::Executor;
    /// use acvus_extern::decl::Nth;
    /// use acvus_extern::runtime::Runtime;
    ///
    /// fn erase<R: Runtime>(cx: &Executor<'_, R>) -> Uniform<R, Nth<0>> {
    ///     Uniform::new(cx, Nth::<0>)
    /// }
    /// ```
    pub fn new<'c>(cx: &Executor<'c, R>, value: X) -> Self
    where
        X: InPlace<R>,
    {
        Uniform::of_the_crossing(value.give(cx.runtime(), Settled::of_the_entry()))
    }

    /// The value itself. A `Uniform` only dropped is released and never
    /// taken.
    pub fn into_inner<'c>(self, cx: &Executor<'c, R>) -> X
    where
        X: InPlace<R>,
    {
        X::take(cx.runtime(), self.into_value(), Settled::of_the_entry())
    }

    /// The value in place, for as long as the box is borrowed.
    pub fn as_ref<'s>(&'s self, cx: &Executor<'s, R>) -> &'s X
    where
        X: InPlace<R>,
    {
        X::shared(cx.runtime(), &self.value, Settled::of_the_entry())
    }

    /// The value in place, exclusively, for as long as the box is borrowed.
    pub fn as_mut<'s>(&'s mut self, cx: &Executor<'s, R>) -> &'s mut X
    where
        X: InPlace<R>,
    {
        X::exclusive(cx.runtime(), &mut self.value, Settled::of_the_entry())
    }

}

/// A primitive: a `Copy` value the runtime holds in its own representation
/// and views in place.
pub trait Primitive: Copy + Send + Sync + Crossing + 'static {}

macro_rules! primitive {
    ($($t:ty => $decl:expr),+ $(,)?) => {$(
        impl Primitive for $t {}
    )+};
}

primitives!(primitive);

impl<R: Runtime, X> Drop for Uniform<R, X> {
    fn drop(&mut self) {
        self.value.release(Releasing { _private: () });
    }
}

// SAFETY: `Uniform` is `repr(transparent)` over the runtime value, and its
// index is a `PhantomData<fn() -> X>` of size 0 and alignment 1 for every `X`,
// so `Uniform<R, X>` and `Uniform<R, Never>` have one size, alignment and
// field offset. Their release is the one `Drop` above, which cannot depend on
// `X` (E0367). `X` is held only as that phantom, no projection through it
// exists, and nothing over it is interior-mutable. `FACTS` states the index as
// the one type parameter the checker's flows read; the runtime `R` is not
// declared to the checker.
unsafe impl<R, X> Structural for Uniform<R, X>
where
    R: Runtime,
{
    type Canon = Uniform<R, Never>;

    const FACTS: StructuralFacts = StructuralFacts {
        layout: LayoutFact::CanonicalOf,
        generics: &[GenericParam::Type],
    };
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{Never, Uniform};
    use crate::boundary::signed::{LayoutFact, Structural};
    use crate::boundary::stored::LayoutWitness;
    use crate::boundary::stub::{Counted, Stub};

    /// The index of a type variable in the tests.
    struct X;

    fn owned<I>(releases: &'static AtomicUsize, id: u32) -> Uniform<Stub, I> {
        Uniform { value: Counted { releases, id }, _index: PhantomData }
    }

    #[test]
    fn a_dropped_uniform_releases_its_value_once() {
        static RELEASES: AtomicUsize = AtomicUsize::new(0);

        drop(owned::<X>(&RELEASES, 0));

        assert_eq!(RELEASES.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_uniform_cast_to_its_canonical_form_is_still_released_once() {
        static RELEASES: AtomicUsize = AtomicUsize::new(0);

        let canon = LayoutWitness::<Option<Uniform<Stub, X>>, _>::canonical().cast(Some(owned(&RELEASES, 0)));
        assert_eq!(RELEASES.load(Ordering::SeqCst), 0);
        drop(canon);

        assert_eq!(RELEASES.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_canonical_uniform_reads_back_at_its_index_in_place() {
        static RELEASES: AtomicUsize = AtomicUsize::new(0);
        let mut canon: (Uniform<Stub, Never>, i64) = (owned(&RELEASES, 7), 1);

        let at_x = LayoutWitness::<(Uniform<Stub, X>, i64), _>::canonical().cast_back_mut(&mut canon);
        at_x.1 = 2;
        assert_eq!(at_x.0.value.id, 7);
        drop(canon);

        assert_eq!(RELEASES.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_constructor_over_a_uniform_is_not_its_own_canonical_form() {
        assert_eq!(<Uniform<Stub, X>>::FACTS.layout, LayoutFact::CanonicalOf);
        assert_eq!(<(i64, Option<Uniform<Stub, X>>)>::FACTS.layout, LayoutFact::CanonicalOf);
        assert_eq!(<(i64, Option<char>)>::FACTS.layout, LayoutFact::OwnCanonical);
        assert_eq!(<Result<(i64, u8), String>>::FACTS.generics.len(), 2);
    }
}
