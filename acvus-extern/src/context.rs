//! A body's context (macros §1, impls §3): which calls the body may make, and
//! how it finishes. A returning body calls callbacks and impls returning; an
//! awaited body calls them either way.

use std::ops::{Deref, DerefMut};

use crate::boundary::entry::{Settled, SiteShape};
use crate::boundary::uniform::Uniform;
use crate::callback::{RustFn, RustFns};
use crate::done::Done;
use crate::guard::{Guard, GuardMut};
use crate::runtime::{Crosses, RefTarget, Runtime};

pub struct Returning<'c, R: Runtime> {
    runtime: &'c R,
    site: &'c SiteShape<R>,
}

pub struct Awaited<'c, R: Runtime> {
    runtime: &'c R,
    site: &'c SiteShape<R>,
}

mod sealed {
    pub trait Sealed {}
}

/// A context a retained loan may be opened in: either body's.
pub trait Opens<'c, R: Runtime>: sealed::Sealed {
    /// The runtime the call runs at.
    #[doc(hidden)]
    fn runtime(&self) -> &'c R;
}

impl<R: Runtime> sealed::Sealed for Returning<'_, R> {}
impl<R: Runtime> sealed::Sealed for Awaited<'_, R> {}

impl<'c, R: Runtime> Opens<'c, R> for Returning<'c, R> {
    fn runtime(&self) -> &'c R {
        self.runtime
    }
}

impl<'c, R: Runtime> Opens<'c, R> for Awaited<'c, R> {
    fn runtime(&self) -> &'c R {
        self.runtime
    }
}

impl<'c, R: Runtime> Returning<'c, R> {
    pub(crate) fn of_the_entry(runtime: &'c R, site: &'c SiteShape<R>) -> Self {
        Returning { runtime, site }
    }

    /// Finishes the call with its result, crossed at the site's settled result
    /// type.
    pub fn finish<T: Crosses>(self, value: T) -> Done<R> {
        Done::of(self.runtime.give(value, Settled::of_the_entry()))
    }

    /// Finishes the call with a result that borrows a part of what the call
    /// lent: `part` picks it from the guard's target, and the reference value
    /// is made while the guard still lends it (S2 names the target, or its
    /// written-back copy, after the guard drops).
    pub fn finish_lent<G, P>(self, guard: Guard<'c, G>, part: impl for<'s> FnOnce(&'s G::Target) -> &'s P) -> Done<R>
    where
        G: Deref,
        P: ?Sized,
    {
        let value = self.runtime.reference(RefTarget::Shared(part(&*guard)), Settled::of_the_entry());
        drop(guard);
        Done::of(value)
    }

    pub fn finish_lent_mut<G, P>(
        self,
        guard: GuardMut<'c, G>,
        part: impl for<'s> FnOnce(&'s mut G::Target) -> &'s mut P,
    ) -> Done<R>
    where
        G: DerefMut,
        P: ?Sized,
    {
        let mut guard = guard;
        let value = self.runtime.reference(RefTarget::Exclusive(part(&mut *guard)), Settled::of_the_entry());
        drop(guard);
        Done::of(value)
    }
}

impl<'c, R: RustFns> Returning<'c, R> {
    /// Finishes the call with a Rust function, as a script function value of
    /// the function type it was made at.
    pub fn finish_rust_fn(self, f: RustFn<R>) -> Done<R> {
        Done::of(self.runtime.rust_fn(f))
    }
}

impl<'c, R: Runtime> Returning<'c, R> {
    /// Finishes the call with a borrowed result of a transparent structured
    /// type, which crosses as its projection (crossing §2): `build` pushes a
    /// reference per leaf and a value per owned part, and composes them with
    /// the aggregates they form, until one value is left.
    pub fn finish_projection(self, build: impl FnOnce(&mut ResultParts<'c, R>)) -> Done<R> {
        let mut parts = ResultParts { runtime: self.runtime, pieces: Vec::new() };
        build(&mut parts);
        assert_eq!(parts.pieces.len(), 1, "a projection result composes to one value");
        let result = parts.pieces.pop().expect("one piece is left");
        Done::of(result.into_value())
    }
}

/// The pieces of a projection result, composed from the last pushed. Each is
/// held by a `Uniform`, so a result the body abandons is released.
pub struct ResultParts<'c, R: Runtime> {
    runtime: &'c R,
    pieces: Vec<Uniform<R, ()>>,
}

impl<'c, R: Runtime> ResultParts<'c, R> {
    fn push(&mut self, value: R::Value) {
        self.pieces.push(Uniform::of_the_crossing(value));
    }

    /// The last `n` pieces, in the order they were pushed.
    fn last(&mut self, n: u32) -> impl ExactSizeIterator<Item = R::Value> + use<'_, 'c, R> {
        let n = n as usize;
        let from = self
            .pieces
            .len()
            .checked_sub(n)
            .expect("an aggregate is composed from pieces already pushed");
        self.pieces.drain(from..).map(Uniform::into_value)
    }

    /// A leaf borrowing a part of what the call lent: the reference value is
    /// made while the guard lends the part.
    pub fn lent<G, P>(&mut self, guard: Guard<'c, G>, part: impl for<'s> FnOnce(&'s G::Target) -> &'s P)
    where
        G: Deref,
        P: ?Sized,
    {
        let value = self.runtime.reference(RefTarget::Shared(part(&*guard)), Settled::of_the_entry());
        drop(guard);
        self.push(value);
    }

    /// As `lent`, writable, from an exclusive guard.
    pub fn lent_mut<G, P>(&mut self, guard: GuardMut<'c, G>, part: impl for<'s> FnOnce(&'s mut G::Target) -> &'s mut P)
    where
        G: DerefMut,
        P: ?Sized,
    {
        let mut guard = guard;
        let value = self.runtime.reference(RefTarget::Exclusive(part(&mut *guard)), Settled::of_the_entry());
        drop(guard);
        self.push(value);
    }

    /// A part given by value, at its settled type.
    pub fn value<T: Crosses>(&mut self, value: T) {
        let value = self.runtime.give(value, Settled::of_the_entry());
        self.push(value);
    }

    /// A tuple of the last `n` pieces.
    pub fn tuple(&mut self, n: u32) {
        let runtime = self.runtime;
        let tuple = runtime.make_tuple(self.last(n), Settled::of_the_entry());
        self.push(tuple);
    }

    /// An object of the last `n` pieces, its fields in the type's order.
    pub fn object(&mut self, n: u32) {
        let runtime = self.runtime;
        let object = runtime.make_object(self.last(n), Settled::of_the_entry());
        self.push(object);
    }

    /// A variant of tag `tag`, its payload the last piece where it has one.
    pub fn variant(&mut self, tag: u32, has_payload: bool) {
        let payload = has_payload.then(|| self.last(1).next().expect("one piece"));
        let variant = self.runtime.make_variant(tag, payload, Settled::of_the_entry());
        self.push(variant);
    }
}

impl<'c, R: Runtime> Awaited<'c, R> {
    pub(crate) fn of_the_entry(runtime: &'c R, site: &'c SiteShape<R>) -> Self {
        Awaited { runtime, site }
    }

    pub fn finish<T: Crosses>(self, value: T) -> Done<R> {
        Done::of(self.runtime.give(value, Settled::of_the_entry()))
    }

    pub fn sleep(&self, d: std::time::Duration) -> impl Future<Output = ()> + Send {
        self.runtime.sleep(d)
    }
}
