//! The one door of one call (crossing §1): it reads every argument at the
//! site's settled type, opens it by its mode, and hands the body its
//! arguments and its context inside a closure (G1, G9, G10).

use std::marker::PhantomData;

use crate::context::{Awaited, Returning};
use crate::done::Done;
use crate::runtime::{Mutability, Runtime};

/// Proof that a read or a result is the crossing's, at the type the checker
/// settled for the site of the call `'c`. Only the entry, and what it opens,
/// make one. `'c` is invariant, so it brands the call: a guard opened under
/// it, and a part borrowed from that guard, belong to this call alone.
#[derive(Clone, Copy)]
pub struct Settled<'c> {
    _call: PhantomData<fn(&'c ()) -> &'c ()>,
}

impl Settled<'_> {
    pub(crate) fn of_the_entry() -> Self {
        Settled { _call: PhantomData }
    }
}

/// How each argument of one call site is passed, as the checker settled it.
pub struct SiteShape {
    lent: Box<[Option<Mutability>]>,
}

pub struct Entry<'c, R: Runtime> {
    runtime: &'c R,
    args: &'c [R::Value],
    site: &'c SiteShape,
}

impl<'c, R: Runtime> Entry<'c, R> {
    /// # Safety
    /// Each of `args` is a value of the type the checker settled for its
    /// position at the call site `site` describes (safety §3). The runtime
    /// that calls the entry owes it; the adapter over that runtime owes it too,
    /// since it passes the runtime's values unchanged.
    pub unsafe fn new(runtime: &'c R, args: &'c [R::Value], site: &'c SiteShape) -> Self {
        Entry { runtime, args, site }
    }

    pub fn open<A: Args<R>>(self, body: impl FnOnce(Returning<'c, R>, A::Opened<'c>) -> Done<R>) -> Done<R> {
        todo!("open A at Settled, run the body")
    }

    pub fn open_awaited<A: Args<R>>(
        self,
        body: impl AsyncFnOnce(Awaited<'c, R>, A::Opened<'c>) -> Done<R>,
    ) -> impl Future<Output = Done<R>> + 'c {
        async { todo!("open A at Settled, run the body") }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A parameter's mode, chosen by its Rust type: how the entry opens it.
pub trait Mode<R: Runtime>: sealed::Sealed {
    type Opened<'c>;
}

/// The parameters of a handler, a tuple of modes, opened together.
pub trait Args<R: Runtime>: sealed::Sealed {
    type Opened<'c>;
}

/// The parameter modes (macros §2, crossing §2).
pub mod mode {
    use std::marker::PhantomData;

    /// Taken by value.
    pub struct ByValue<T>(PhantomData<fn() -> T>);
    /// Lent `&T`, `T` not a transparent structured type.
    pub struct ByRef<T>(PhantomData<fn() -> T>);
    /// Lent `&mut T`.
    pub struct ByMut<T>(PhantomData<fn() -> T>);
    /// A program's array lent as `&[T]` or `&mut [T]`.
    pub struct BySlice<T>(PhantomData<fn() -> T>);
    pub struct BySliceMut<T>(PhantomData<fn() -> T>);
    /// A `String` lent as `&str`.
    pub struct ByStr;
    /// A transparent structured value lent as its projection.
    pub struct ByProjection<P>(PhantomData<fn() -> P>);
    pub struct ByProjectionMut<P>(PhantomData<fn() -> P>);
    /// A script closure.
    pub struct ByCallback<A, U, E>(PhantomData<fn() -> (A, U, E)>);
    /// A requirement owning its receiver, and one standing at a type.
    pub struct ByInstance<F, I>(PhantomData<fn() -> (F, I)>);
    pub struct ByInstanceOf<F, Ts, L>(PhantomData<fn() -> (F, Ts, L)>);
}
