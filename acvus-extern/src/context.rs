//! A body's context (macros §1, impls §3): which calls the body may make, and
//! how it finishes. A returning body calls callbacks and impls returning; an
//! awaited body calls them either way.

use crate::boundary::entry::Settled;
use crate::done::Done;
use crate::borrowed::{Borrows, Sink};
use crate::boundary::seal::Crossing;
use crate::runtime::{Crosses, Runtime};

pub struct Returning<'c, R: Runtime> {
    runtime: &'c R,
}

pub struct Awaited<'c, R: Runtime> {
    runtime: &'c R,
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
    pub(crate) fn of_the_entry(runtime: &'c R) -> Self {
        Returning { runtime }
    }

    /// Finishes the call with its result, crossed at the site's settled result
    /// type.
    pub fn finish<T: Crosses<R>>(self, value: T) -> Done<R> {
        Done::of(value.give(self.runtime, Settled::of_the_entry()))
    }

    /// Finishes the call with a result that borrows: each reference in it a
    /// reference value made from it, at a lifetime within this call (runtime
    /// contract §2, References). Which loan each holds, an argument's lent in
    /// place or one a carrier holds, the declaration's lifetimes say.
    pub fn finish_borrowed<'s>(self, result: impl Borrows<'s, R>) -> Done<R>
    where
        'c: 's,
    {
        let sink = Sink::of_the_finish(self.runtime, Settled::of_the_entry());
        Done::of(result.made(&sink).into_value())
    }
}

impl<'c, R: Runtime> Awaited<'c, R> {
    pub(crate) fn of_the_entry(runtime: &'c R) -> Self {
        Awaited { runtime }
    }

    pub fn finish<T: Crosses<R>>(self, value: T) -> Done<R> {
        Done::of(value.give(self.runtime, Settled::of_the_entry()))
    }

    pub fn sleep(&self, d: std::time::Duration) -> impl Future<Output = ()> + Send {
        self.runtime.sleep(d)
    }
}
