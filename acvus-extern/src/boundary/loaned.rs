//! A retained loan: the runtime's reference value inside a value that outlives
//! the call it was lent in, opened only within a later call (K12).

use std::marker::PhantomData;

use crate::boundary::entry::Settled;
use crate::context::Opens;
use crate::runtime::{Crosses, Runtime};

/// A shared loan on storage of type `T`, retained as a reference value. `'a`
/// is the checker's loan, read by the declaration's flows; no Rust reference
/// lives in it.
pub struct Loaned<'a, T, R: Runtime> {
    reference: R::Value,
    _loan: PhantomData<(&'a (), fn() -> T)>,
}

/// An exclusive loan, retained as a reference value.
pub struct LoanedMut<'a, T, R: Runtime> {
    reference: R::Value,
    _loan: PhantomData<(&'a mut (), fn() -> T)>,
}

impl<T: Crosses, R: Runtime> Loaned<'_, T, R> {
    /// Opens the loan within the call `cx` belongs to.
    pub fn open<'c>(&'c self, cx: &'c impl Opens<'c, R>) -> R::Shared<'c, T> {
        cx.runtime().shared(self.reference, Settled::of_the_entry())
    }
}

impl<T: Crosses, R: Runtime> LoanedMut<'_, T, R> {
    pub fn open<'c>(&'c mut self, cx: &'c impl Opens<'c, R>) -> R::Exclusive<'c, T> {
        cx.runtime().exclusive(self.reference, Settled::of_the_entry())
    }
}
