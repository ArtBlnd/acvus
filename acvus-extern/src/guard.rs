//! A guard a call opened, as the handler holds it (crossing §2, §4). Only the
//! entry makes one, from the runtime's guard under the call's `Settled`, so a
//! part borrowed from it is a part of storage this call lent: nothing else has
//! the type a borrowed result is finished from (K16).

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

/// A shared guard of the call `'c` over the runtime's guard `G`.
pub struct Guard<'c, G> {
    inner: G,
    _call: PhantomData<fn(&'c ()) -> &'c ()>,
}

/// An exclusive guard of the call `'c` over the runtime's guard `G`.
pub struct GuardMut<'c, G> {
    inner: G,
    _call: PhantomData<fn(&'c ()) -> &'c ()>,
}

impl<G> Guard<'_, G> {
    pub(crate) fn of_the_entry(inner: G) -> Self {
        Guard { inner, _call: PhantomData }
    }

    pub(crate) fn into_inner(self) -> G {
        self.inner
    }
}

impl<G> GuardMut<'_, G> {
    pub(crate) fn of_the_entry(inner: G) -> Self {
        GuardMut { inner, _call: PhantomData }
    }

    pub(crate) fn into_inner(self) -> G {
        self.inner
    }
}

impl<G: Deref> Deref for Guard<'_, G> {
    type Target = G::Target;

    fn deref(&self) -> &G::Target {
        &self.inner
    }
}

impl<G: Deref> Deref for GuardMut<'_, G> {
    type Target = G::Target;

    fn deref(&self) -> &G::Target {
        &self.inner
    }
}

impl<G: DerefMut> DerefMut for GuardMut<'_, G> {
    fn deref_mut(&mut self) -> &mut G::Target {
        &mut self.inner
    }
}
