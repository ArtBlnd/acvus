//! A script closure an extern calls (crossing §3-§4): a returning call lends
//! references and its result borrows only what it lent; an awaited call takes
//! its arguments by move and lends none of the handler's storage (K3).

use std::marker::PhantomData;

use crate::boundary::entry::{Entry, Settled};
use crate::done::Done;
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

/// Calling a script closure, returning or awaited (runtime contract §2). The
/// arguments are typed at the runtime the call runs at, `R`: the runtime
/// itself, or the adapter over it (K15), whose values are the runtime's.
pub trait CallsBack: Runtime {
    fn call_now<R, A, U>(&self, closure: Self::Value, args: A::Lent<'_>, at: Settled<'_>) -> U
    where
        R: Runtime<Value = Self::Value>,
        A: CallArgs<R>,
        U: Crosses;
    fn call_later<R, A, U>(&self, closure: Self::Value, args: A::Moved, at: Settled<'_>) -> impl Future<Output = U> + Send
    where
        R: Runtime<Value = Self::Value>,
        A: CallArgs<R>,
        U: Crosses;
}

/// A Rust function an extern returns as a script function value (runtime
/// contract §2, Rust functions): each call of it is a call through an entry,
/// crossing its arguments at that call's settled types. Only the boundary makes
/// one, from a handler's glue.
pub struct RustFn<R: Runtime> {
    call: fn(Entry<'_, R>) -> Done<R>,
}

/// Making a script function value from a Rust function. `R` is the runtime the
/// glue runs at: the runtime itself, or the adapter over it (K15).
pub trait RustFns: Runtime {
    fn rust_fn<R>(&self, f: RustFn<R>) -> Self::Value
    where
        R: Runtime<Value = Self::Value>;
}
