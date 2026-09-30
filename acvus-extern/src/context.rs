//! A body's context (macros §1, impls §3): which calls the body may make, and
//! how it finishes. A returning body calls callbacks and impls returning; an
//! awaited body calls them either way.

use crate::boundary::entry::SiteShape;
use crate::done::Done;
use crate::runtime::{Crosses, Runtime};

pub struct Returning<'c, R: Runtime> {
    runtime: &'c R,
    site: &'c SiteShape,
}

pub struct Awaited<'c, R: Runtime> {
    runtime: &'c R,
    site: &'c SiteShape,
}

mod sealed {
    pub trait Sealed {}
}

/// A context a retained loan may be opened in: either body's.
pub trait Opens<'c, R: Runtime>: sealed::Sealed {}

impl<R: Runtime> sealed::Sealed for Returning<'_, R> {}
impl<R: Runtime> sealed::Sealed for Awaited<'_, R> {}
impl<'c, R: Runtime> Opens<'c, R> for Returning<'c, R> {}
impl<'c, R: Runtime> Opens<'c, R> for Awaited<'c, R> {}

impl<'c, R: Runtime> Returning<'c, R> {
    /// Finishes the call with its result, crossed at the site's settled result
    /// type.
    pub fn finish<T: Crosses>(self, value: T) -> Done<R> {
        todo!("give the value at the settled result type")
    }

    /// Finishes the call with a result that borrows what the call lent.
    pub fn finish_lent<T: Crosses + 'c>(self, lent: R::Shared<'c, T>) -> Done<R> {
        todo!("the reference value made from the guard")
    }

    pub fn finish_lent_mut<T: Crosses + 'c>(self, lent: R::Exclusive<'c, T>) -> Done<R> {
        todo!("the reference value made from the guard")
    }
}

impl<'c, R: Runtime> Awaited<'c, R> {
    pub fn finish<T: Crosses>(self, value: T) -> Done<R> {
        todo!("give the value at the settled result type")
    }

    pub fn sleep(&self, d: std::time::Duration) -> impl Future<Output = ()> + Send {
        self.runtime.sleep(d)
    }
}
