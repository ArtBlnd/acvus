//! A body's executor (macros §1, impls §3): the runtime its call runs at,
//! given to the returning body and to the awaited body alike. Through it a
//! body opens what it holds, calls callbacks and impls, synchronously or
//! awaited, and finishes.

use crate::boundary::entry::Settled;
use crate::borrowed::{Borrows, Sink};
use crate::done::Done;
use crate::runtime::{Crosses, Runtime};

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

    /// Finishes the call with its result, crossed at the site's settled result
    /// type.
    pub fn finish<T: Crosses<R>>(self, value: T) -> Done<R> {
        Done::of(value.give(self.runtime, Settled::of_the_entry()))
    }

    /// Finishes the call with a result that borrows: each reference in it a
    /// reference value made from it, at a lifetime within this call (runtime
    /// contract §2, References). Which loan each holds, an argument's lent in
    /// place or one a carrier holds, the declaration's lifetimes say. Only a
    /// returning body's result borrows (macros §1).
    pub fn finish_borrowed<'s>(self, result: impl Borrows<'s, R>) -> Done<R>
    where
        'c: 's,
    {
        let sink = Sink::of_the_finish(self.runtime, Settled::of_the_entry());
        Done::of(result.made(&sink).into_value())
    }

    pub fn sleep(&self, d: std::time::Duration) -> impl Future<Output = ()> + Send {
        self.runtime.sleep(d)
    }
}
