//! A script closure an extern calls (crossing §3-§4): a returning call lends
//! references and its result borrows only what it lent; an awaited call takes
//! its arguments by move and lends none of the handler's storage (K3).

use std::marker::PhantomData;

use crate::boundary::entry::Settled;
use crate::context::{Awaited, Returning};
use crate::runtime::{Crosses, Runtime};

/// A callback of function type `Fn(A) -> U` at effect `E`; `'a` is the loans
/// its captures hold. Only the entry makes one.
pub struct Callback<'a, A, U, E, R: Runtime> {
    closure: R::Value,
    _at: PhantomData<(&'a (), fn(A) -> U, E)>,
}

mod sealed {
    pub trait Sealed {}
}

/// A callback's parameters, a tuple: lent as the handler passes them to a
/// returning call, or moved into an awaited one.
pub trait CallArgs<R: Runtime>: sealed::Sealed {
    type Lent<'l>;
    type Moved;
}

impl<'a, A: CallArgs<R>, U: Crosses, E, R: CallsBack> Callback<'a, A, U, E, R> {
    pub fn call<'l>(&self, cx: &Returning<'_, R>, args: A::Lent<'l>) -> U {
        todo!("call_now at a Settled the callback proves")
    }

    pub fn call_await(&self, cx: &Awaited<'_, R>, args: A::Moved) -> impl Future<Output = U> + Send {
        async { todo!("call_later at a Settled the callback proves") }
    }
}

/// Calling a script closure, returning or awaited (runtime contract §2).
pub trait CallsBack: Runtime {
    fn call_now<A: CallArgs<Self>, U: Crosses>(&self, closure: Self::Value, args: A::Lent<'_>, at: Settled<'_>) -> U;
    fn call_later<A: CallArgs<Self>, U: Crosses>(
        &self,
        closure: Self::Value,
        args: A::Moved,
        at: Settled<'_>,
    ) -> impl Future<Output = U> + Send;
}
