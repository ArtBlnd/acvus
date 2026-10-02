//! A body's executor (macros §1, impls §3): the runtime its call runs at,
//! given to the returning body and to the awaited body alike. Through it a
//! body opens what it holds, calls callbacks and impls, synchronously or
//! awaited, and finishes.

use crate::boundary::entry::Settled;
use crate::done::Done;
use crate::runtime::{Give, Runtime};

pub struct Executor<'c, R: Runtime> {
    runtime: &'c R,
}

impl<'c, R: Runtime> Executor<'c, R> {
    pub(crate) fn of_the_entry(runtime: &'c R) -> Self {
        Executor { runtime }
    }

    /// The runtime the call runs at.
    pub(crate) fn runtime(&self) -> &'c R {
        self.runtime
    }

    /// Finishes the call with its result at the site's settled result type: a
    /// runtime value, at the loan `'s` it borrows for (crossing §4).
    pub fn finish<'s, P: Give<'s, R>>(self, result: P) -> Done<R>
    where
        'c: 's,
    {
        Done::of(result.give(self.runtime, Settled::of_the_entry()))
    }

    pub fn sleep(&self, d: std::time::Duration) -> impl Future<Output = ()> + Send {
        self.runtime.sleep(d)
    }
}
