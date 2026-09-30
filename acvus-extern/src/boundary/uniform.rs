//! A type variable's value: the runtime's uniform value indexed by the
//! variable, owned, and released once by its drop (G7).

use std::marker::PhantomData;
use std::mem::ManuallyDrop;

use crate::boundary::entry::Settled;
use crate::boundary::signed::{GenericParam, LayoutFact, Structural, StructuralFacts, opaque_crossing};
use crate::context::Opens;
use crate::runtime::{Crosses, Runtime};
use crate::shape::{OwnedParts, Projected};

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

/// The uniform value at a type variable `X`. No trait implemented on it has an
/// item that depends on `X` (runtime contract §1).
#[repr(transparent)]
pub struct Uniform<R: Runtime, X> {
    value: ManuallyDrop<R::Value>,
    _index: PhantomData<fn() -> X>,
}

impl<R: Runtime, X> Uniform<R, X> {
    /// The value the crossing hands over, owned from here on.
    pub(crate) fn of_the_crossing(value: R::Value) -> Self {
        Uniform { value: ManuallyDrop::new(value), _index: PhantomData }
    }

    /// The value handed on to its next owner, unreleased.
    pub(crate) fn into_value(self) -> R::Value {
        *ManuallyDrop::new(self).value
    }
}

impl<R: Runtime, T: Crosses> Uniform<R, T> {
    /// Lends the value in place within the call `cx` belongs to.
    pub fn open<'c>(&'c self, cx: &'c impl Opens<'c, R>) -> R::Shared<'c, T> {
        cx.runtime().shared(&self.value, Settled::of_the_entry())
    }

    pub fn open_mut<'c>(&'c mut self, cx: &'c impl Opens<'c, R>) -> R::Exclusive<'c, T> {
        cx.runtime().exclusive(&mut self.value, Settled::of_the_entry())
    }

    /// The value itself. A `Uniform` only dropped is released and never
    /// taken.
    pub fn into_inner<'c>(self, cx: &'c impl Opens<'c, R>) -> T {
        self.take(cx.runtime(), Settled::of_the_entry())
    }
}

impl<R: Runtime, T: Structural> Uniform<R, T> {
    /// The value moved out at its type: the one place a value is taken, the
    /// runtime's `take` for an opaque type and the owned view for a
    /// transparent one. The take consumes the value's storage, so the
    /// holder's release is skipped.
    pub(crate) fn take(self, runtime: &R, at: Settled<'_>) -> T {
        T::take(self.into_value(), runtime, at)
    }
}

impl<R: Runtime, S: Projected<R>> Uniform<R, S> {
    /// The shape's owned view: the aggregate taken apart once, each part then
    /// held by its own `Uniform`.
    pub fn into_view<'c>(self, cx: &'c impl Opens<'c, R>) -> S::Owned {
        self.view(cx.runtime(), Settled::of_the_entry())
    }

    pub(crate) fn view(self, runtime: &R, at: Settled<'_>) -> S::Owned {
        let aggregate = self.into_value();
        let (tag, parts) = runtime.take_apart(aggregate, at);
        S::owned(OwnedParts::of_the_body(tag, parts))
    }
}

impl<R: Runtime, X> Drop for Uniform<R, X> {
    fn drop(&mut self) {
        (*self.value).release(Releasing { _private: () });
    }
}

// SAFETY: `Uniform` is `repr(transparent)` over the runtime value, and its
// index is a `PhantomData<fn() -> X>` of size 0 and alignment 1 for every `X`,
// so `Uniform<R, X>` and `Uniform<R, Never>` have one size, alignment and
// field offset. Their release is the one `Drop` above, which cannot depend on
// `X` (E0367). `X` is held only as that phantom, no projection through it
// exists, and nothing over it is interior-mutable. `FACTS` states the index as
// the one type parameter the checker's flows read; the runtime `R` is not
// declared to the checker. A `Uniform` held inside a Rust value crosses
// opaque, at itself.
unsafe impl<R, X> Structural for Uniform<R, X>
where
    R: Runtime,
{
    type Canon = Uniform<R, Never>;

    const FACTS: StructuralFacts = StructuralFacts {
        layout: LayoutFact::CanonicalOf,
        generics: &[GenericParam::Type],
    };

    opaque_crossing!();
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;
    use std::mem::ManuallyDrop;
    use std::ops::{Deref, DerefMut};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use acvus_utils::Interner;

    use super::{Never, Release, Releasing, Uniform};
    use crate::boundary::entry::Settled;
    use crate::boundary::signed::{LayoutFact, SignedS1, SignedS2, SignedS3, Structural};
    use crate::boundary::stored::LayoutWitness;
    use crate::runtime::{Crosses, RefTarget, Runtime};

    /// A value that counts its releases and names no storage.
    #[derive(Clone, Copy)]
    struct Counted {
        releases: &'static AtomicUsize,
        id: u32,
    }

    impl Release for Counted {
        fn release(self, _releasing: Releasing) {
            self.releases.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A runtime whose only live operation is release; the tests read no value
    /// through it.
    struct Counting;

    // SAFETY: `Counting` lends no storage, makes no reference value and keeps
    // nothing: every operation that would is unreachable below.
    unsafe impl SignedS1 for Counting {}
    // SAFETY: as for S1.
    unsafe impl SignedS2 for Counting {}
    // SAFETY: as for S1.
    unsafe impl SignedS3 for Counting {}

    struct NoBody;

    struct NoGuard<T: ?Sized>(PhantomData<fn() -> Box<T>>);

    impl<T: ?Sized> Deref for NoGuard<T> {
        type Target = T;
        fn deref(&self) -> &T {
            unreachable!("the release tests lend no storage")
        }
    }

    impl<T: ?Sized> DerefMut for NoGuard<T> {
        fn deref_mut(&mut self) -> &mut T {
            unreachable!("the release tests lend no storage")
        }
    }

    impl Runtime for Counting {
        type Value = Counted;
        type Shared<'a, T: Crosses + 'a> = NoGuard<T>;
        type Exclusive<'a, T: Crosses + 'a> = NoGuard<T>;
        type SharedSlice<'a, T: Crosses + 'a> = NoGuard<[T]>;
        type ExclusiveSlice<'a, T: Crosses + 'a> = NoGuard<[T]>;
        type SharedStr<'a> = NoGuard<str>;

        fn take<T: Crosses>(&self, _: Counted, _: Settled<'_>) -> T {
            unreachable!("the release tests read no value")
        }
        fn give<T: Crosses>(&self, _: T, _: Settled<'_>) -> Counted {
            unreachable!("the release tests make no value")
        }
        fn shared<'a, T: Crosses + 'a>(&'a self, _: &'a Counted, _: Settled<'a>) -> NoGuard<T> {
            unreachable!("the release tests lend no storage")
        }
        fn exclusive<'a, T: Crosses + 'a>(&'a self, _: &'a mut Counted, _: Settled<'a>) -> NoGuard<T> {
            unreachable!("the release tests lend no storage")
        }
        fn shared_slice<'a, T: Crosses + 'a>(&'a self, _: &'a Counted, _: Settled<'a>) -> NoGuard<[T]> {
            unreachable!("the release tests lend no storage")
        }
        fn exclusive_slice<'a, T: Crosses + 'a>(
            &'a self,
            _: &'a mut Counted,
            _: Settled<'a>,
        ) -> NoGuard<[T]> {
            unreachable!("the release tests lend no storage")
        }
        fn shared_str<'a>(&'a self, _: &'a Counted, _: Settled<'a>) -> NoGuard<str> {
            unreachable!("the release tests lend no storage")
        }
        fn take_apart(&self, _: Counted, _: Settled<'_>) -> (Option<u32>, impl ExactSizeIterator<Item = Counted>) {
            unreachable!("the release tests take no aggregate apart");
            #[allow(unreachable_code)]
            (None, std::iter::empty())
        }
        fn make_tuple(&self, _: impl ExactSizeIterator<Item = Counted>, _: Settled<'_>) -> Counted {
            unreachable!("the release tests make no tuple")
        }
        fn make_object(&self, _: impl ExactSizeIterator<Item = Counted>, _: Settled<'_>) -> Counted {
            unreachable!("the release tests make no object")
        }
        fn make_variant(&self, _: u32, _: Option<Counted>, _: Settled<'_>) -> Counted {
            unreachable!("the release tests make no variant")
        }
        fn lent_part<'a>(&'a self, _: &'a Counted, _: u32, _: Settled<'a>) -> &'a Counted {
            unreachable!("the release tests lend no storage")
        }
        fn lent_parts_mut<'a>(
            &'a self,
            _: &'a mut Counted,
            _: Settled<'a>,
        ) -> impl ExactSizeIterator<Item = &'a mut Counted> {
            unreachable!("the release tests lend no storage");
            #[allow(unreachable_code)]
            std::iter::empty()
        }
        fn lent_tag(&self, _: &Counted, _: Settled<'_>) -> u32 {
            unreachable!("the release tests lend no storage")
        }
        fn reference<'a, P: ?Sized>(&'a self, _: RefTarget<'a, P>, _: Settled<'a>) -> Counted {
            unreachable!("the release tests make no reference")
        }
        fn target<'l, P: ?Sized>(&'l self, _: Counted, _: Settled<'l>) -> &'l P {
            unreachable!("the release tests make no reference")
        }
        fn target_mut<'l, P: ?Sized>(&'l self, _: Counted, _: Settled<'l>) -> &'l mut P {
            unreachable!("the release tests make no reference")
        }
        fn interner(&self) -> &Interner {
            unreachable!("the release tests intern no name")
        }
        async fn sleep(&self, _: Duration) {
            unreachable!("the release tests do not wait")
        }
    }

    /// The index of a type variable in the tests.
    struct X;

    fn owned<I>(releases: &'static AtomicUsize, id: u32) -> Uniform<Counting, I> {
        Uniform {
            value: ManuallyDrop::new(Counted { releases, id }),
            _index: PhantomData,
        }
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

        let canon = LayoutWitness::<Option<Uniform<Counting, X>>, _>::canonical()
            .cast(Some(owned(&RELEASES, 0)));
        assert_eq!(RELEASES.load(Ordering::SeqCst), 0);
        drop(canon);

        assert_eq!(RELEASES.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_canonical_uniform_reads_back_at_its_index_in_place() {
        static RELEASES: AtomicUsize = AtomicUsize::new(0);
        let mut canon: (Uniform<Counting, Never>, i64) = (owned(&RELEASES, 7), 1);

        let at_x =
            LayoutWitness::<(Uniform<Counting, X>, i64), _>::canonical().cast_back_mut(&mut canon);
        at_x.1 = 2;
        assert_eq!(at_x.0.value.id, 7);
        drop(canon);

        assert_eq!(RELEASES.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_constructor_over_a_uniform_is_not_its_own_canonical_form() {
        assert_eq!(
            <Uniform<Counting, X>>::FACTS.layout,
            LayoutFact::CanonicalOf
        );
        assert_eq!(
            <(i64, Option<Uniform<Counting, X>>)>::FACTS.layout,
            LayoutFact::CanonicalOf
        );
        assert_eq!(
            <(i64, Option<char>)>::FACTS.layout,
            LayoutFact::OwnCanonical
        );
        assert_eq!(<Result<(i64, u8), String>>::FACTS.generics.len(), 2);
    }
}
