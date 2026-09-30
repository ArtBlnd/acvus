//! A body's context (macros §1, impls §3): which calls the body may make, and
//! how it finishes. A returning body calls callbacks and impls returning; an
//! awaited body calls them either way.

use std::ops::{Deref, DerefMut};

use crate::boundary::entry::{Settled, SiteShape};
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
    pub fn finish<T: Crosses<R>>(self, value: T) -> Done<R> {
        Done::of(value.give(self.runtime, Settled::of_the_entry()))
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

impl<'c, R: Runtime> Awaited<'c, R> {
    pub(crate) fn of_the_entry(runtime: &'c R, site: &'c SiteShape<R>) -> Self {
        Awaited { runtime, site }
    }

    pub fn finish<T: Crosses<R>>(self, value: T) -> Done<R> {
        Done::of(value.give(self.runtime, Settled::of_the_entry()))
    }

    pub fn sleep(&self, d: std::time::Duration) -> impl Future<Output = ()> + Send {
        self.runtime.sleep(d)
    }
}
