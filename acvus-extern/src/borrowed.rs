//! A result that borrows (crossing §4): a value of a transparent structure
//! (`Option`, `Result`, a tuple) whose every leaf is a reference to a part of
//! what the call lent. It is made under the guard's borrow `'s`, while the
//! guard still lends: the body turns each reference into a reference value
//! through the `Sink` the finish hands it, and gives back what it `Made`.
//! Neither can be had outside that borrow, so a result is finished only from
//! the guard (K16). No part's type is required to outlive anything but the
//! borrow its reference is at: a type variable's value is no `'static` type
//! (limitations §3).

use std::marker::PhantomData;

use crate::boundary::entry::Settled;
use crate::runtime::{OptionKind, RefKind, RefTarget, Referent, ResultKind, Runtime, TupleKind};

/// A result's value made under the guard's borrow `'s`.
pub struct Made<'s, R: Runtime> {
    value: R::Value,
    _borrow: PhantomData<fn(&'s ()) -> &'s ()>,
}

impl<R: Runtime> Made<'_, R> {
    pub(crate) fn into_value(self) -> R::Value {
        self.value
    }
}

/// What makes a borrowed result's leaves under the guard's borrow `'s`: a
/// reference value from each part. A structure over them is made by
/// `Borrows`.
pub struct Sink<'s, R: Runtime> {
    runtime: &'s R,
    at: Settled<'s>,
}

impl<'s, R: Runtime> Sink<'s, R> {
    pub(crate) fn of_the_finish(runtime: &'s R, at: Settled<'s>) -> Self {
        Sink { runtime, at }
    }

    fn made(&self, value: R::Value) -> Made<'s, R> {
        Made { value, _borrow: PhantomData }
    }

    /// A shared reference to `part`.
    pub fn lent<P: ?Sized + Referent>(&self, part: &'s P) -> Made<'s, R> {
        self.made(<R::Ref as RefKind<R>>::reference(self.runtime, RefTarget::Shared(part), self.at))
    }

    /// An exclusive reference to `part`.
    pub fn lent_mut<P: ?Sized + Referent>(&self, part: &'s mut P) -> Made<'s, R> {
        self.made(<R::Ref as RefKind<R>>::reference(self.runtime, RefTarget::Exclusive(part), self.at))
    }
}

/// A borrowed result at the guard's borrow `'s`, read off its type: a
/// reference is lent, and a structure of borrowed results is made by its kind
/// over its parts.
pub trait Borrows<'s, R: Runtime> {
    fn made(self, sink: &Sink<'s, R>) -> Made<'s, R>;
}

impl<'s, R: Runtime, P: ?Sized + Referent + 's> Borrows<'s, R> for &'s P {
    fn made(self, sink: &Sink<'s, R>) -> Made<'s, R> {
        sink.lent(self)
    }
}

impl<'s, R: Runtime, P: ?Sized + Referent + 's> Borrows<'s, R> for &'s mut P {
    fn made(self, sink: &Sink<'s, R>) -> Made<'s, R> {
        sink.lent_mut(self)
    }
}

impl<'s, R: Runtime, B: Borrows<'s, R>> Borrows<'s, R> for Option<B> {
    fn made(self, sink: &Sink<'s, R>) -> Made<'s, R> {
        let part = self.map(|part| part.made(sink).into_value());
        sink.made(<R::Option as OptionKind<R>>::give(sink.runtime, part, sink.at))
    }
}

impl<'s, R: Runtime, B: Borrows<'s, R>, C: Borrows<'s, R>> Borrows<'s, R> for Result<B, C> {
    fn made(self, sink: &Sink<'s, R>) -> Made<'s, R> {
        let part = self.map(|part| part.made(sink).into_value()).map_err(|part| part.made(sink).into_value());
        sink.made(<R::Result as ResultKind<R>>::give(sink.runtime, part, sink.at))
    }
}

/// A tuple of borrowed results, made by its kind over its parts.
macro_rules! tuple_borrows {
    ($($n:literal ($($b:ident $v:ident $i:tt),+))+) => {$(
        impl<'s, R: Runtime, $($b: Borrows<'s, R>),+> Borrows<'s, R> for ($($b,)+) {
            fn made(self, sink: &Sink<'s, R>) -> Made<'s, R> {
                let ($($v,)+) = self;
                let parts = [$($v.made(sink).into_value()),+];
                sink.made(<R::Tuple as TupleKind<R>>::give::<$n>(sink.runtime, parts, sink.at))
            }
        }
    )+};
}

tuples!(tuple_borrows);
