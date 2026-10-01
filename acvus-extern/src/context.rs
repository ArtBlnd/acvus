//! A body's context (macros §1, impls §3): which calls the body may make, and
//! how it finishes. A returning body calls callbacks and impls returning; an
//! awaited body calls them either way.

use std::ops::{Deref, DerefMut};

use crate::boundary::entry::{Settled, SiteShape};
use crate::done::Done;
use crate::guard::{Guard, GuardMut};
use crate::borrowed::{Made, Sink};
use crate::boundary::seal::Crossing;
use crate::runtime::{Crosses, Runtime};

pub struct Returning<'c, R: Runtime> {
    runtime: &'c R,
    site: &'c SiteShape<R>,
}

pub struct Awaited<'c, R: Runtime> {
    runtime: &'c R,
    site: &'c SiteShape<R>,
}

/// A context a retained loan may be opened in: either body's.
pub trait Opens<'c, R: Runtime>: Crossing {
    /// The runtime the call runs at.
    #[doc(hidden)]
    fn runtime(&self) -> &'c R;
}


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

    /// Finishes the call with a result borrowing parts of what the call lent
    /// through `guard`: `part`, given the context, reads the guard's target
    /// and makes the result through the `Sink`, each reference a reference
    /// value made while the guard still lends it (`borrowed`).
    pub fn finish_borrowed<G>(
        self,
        guard: Guard<'c, G>,
        part: impl for<'s> FnOnce(&'s Self, &'s G::Target, Sink<'s, R>) -> Made<'s, R>,
    ) -> Done<R>
    where
        G: Deref,
    {
        let sink = Sink::of_the_finish(self.runtime, Settled::of_the_entry());
        let value = part(&self, &*guard, sink).into_value();
        drop(guard);
        Done::of(value)
    }

    /// As `finish_borrowed`, through an exclusive guard.
    pub fn finish_borrowed_mut<G>(
        self,
        guard: GuardMut<'c, G>,
        part: impl for<'s> FnOnce(&'s Self, &'s mut G::Target, Sink<'s, R>) -> Made<'s, R>,
    ) -> Done<R>
    where
        G: DerefMut,
    {
        let mut guard = guard;
        let sink = Sink::of_the_finish(self.runtime, Settled::of_the_entry());
        let value = part(&self, &mut *guard, sink).into_value();
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
