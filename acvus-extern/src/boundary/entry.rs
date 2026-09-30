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

/// One call site as the checker settled it: how each argument position is
/// passed, and the impl values chosen for the handler's requirements, in the
/// handler's order (impls §3). The runtime builds it once, at preparation.
pub struct SiteShape<R: Runtime> {
    lent: Box<[Option<Mutability>]>,
    impls: Box<[R::Value]>,
}

impl<R: Runtime> SiteShape<R> {
    pub fn new(lent: Box<[Option<Mutability>]>, impls: Box<[R::Value]>) -> Self {
        SiteShape { lent, impls }
    }
}

/// One call's door. Its arguments are the run of the call's argument
/// positions, a type (`Args::Run`), so the entry indexes it without a compare
/// (performance: facts in types).
pub struct Entry<'c, R: Runtime> {
    runtime: &'c R,
    args: &'c [R::Value],
    site: &'c SiteShape<R>,
}

impl<'c, R: Runtime> Entry<'c, R> {
    /// The entry of one call. It is safe: the checker's decision is trusted
    /// only by the runtime's own typed reads (safety §3), and only the runtime
    /// holds values to pass here, since the host sees none (crossing §6).
    pub fn new(runtime: &'c R, args: &'c [R::Value], site: &'c SiteShape<R>) -> Self {
        Entry { runtime, args, site }
    }

    /// Opens the arguments `A`, one per position, and the requirements `Q`,
    /// from the site's impls, and runs the returning body.
    pub fn open<A, Q>(self, body: impl FnOnce(Returning<'c, R>, A::Opened<'c>, Q::Opened<'c>) -> Done<R>) -> Done<R>
    where
        A: Args<R> + 'c,
        Q: Requires<R> + 'c,
    {
        todo!("A::Run from args, each opened at Settled by its mode; Q from the site's impls; then the body")
    }

    pub fn open_awaited<A, Q>(
        self,
        body: impl AsyncFnOnce(Awaited<'c, R>, A::Opened<'c>, Q::Opened<'c>) -> Done<R>,
    ) -> impl Future<Output = Done<R>> + Send + 'c
    where
        A: Args<R> + 'c,
        Q: Requires<R> + 'c,
    {
        async { todo!("as open, the body awaited") }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A parameter's mode, chosen by its Rust type (macros §4): how the entry
/// opens its argument.
pub trait Mode<R: Runtime>: sealed::Sealed {
    type Opened<'c>
    where
        Self: 'c,
        R: 'c;
}

/// A requirement's mode: how the entry opens the impl the site chose.
pub trait RequirementMode<R: Runtime>: sealed::Sealed {
    type Opened<'c>
    where
        Self: 'c,
        R: 'c;
}

/// The argument parameters of a handler, a tuple of modes, one position each.
/// `Run` is the call's argument run at that arity, `[R::Value; N]`.
pub trait Args<R: Runtime>: sealed::Sealed {
    type Run;
    type Opened<'c>
    where
        Self: 'c,
        R: 'c;
}

/// The requirement parameters of a handler, a tuple of requirement modes.
pub trait Requires<R: Runtime>: sealed::Sealed {
    type Opened<'c>
    where
        Self: 'c,
        R: 'c;
}

/// The parameter modes (macros §2, crossing §2).
pub mod mode {
    use std::marker::PhantomData;

    /// Taken by value where the handler needs the value itself: opened as a
    /// `Uniform` and taken at once.
    pub struct ByValue<T>(PhantomData<fn() -> T>);
    /// Held by value where the handler writes `Uniform<R, T>`: read in place,
    /// released if dropped, taken only by `into_inner`.
    pub struct ByUniform<T>(PhantomData<fn() -> T>);
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
    /// A script closure, its parameters and result loan forms.
    pub struct ByCallback<A, U, E>(PhantomData<fn() -> (A, U, E)>);
    /// A requirement owning its receiver, and one standing at a type.
    pub struct ByInstance<F, I>(PhantomData<fn() -> (F, I)>);
    pub struct ByInstanceOf<F, Ts, L>(PhantomData<fn() -> (F, Ts, L)>);
}

use mode::*;

use crate::boundary::instance::{Instance, InstanceOf};
use crate::boundary::uniform::Uniform;
use crate::callback::Callback;
use crate::guard::{Guard, GuardMut};
use crate::impls::receiver::{Reads, Steps};
use crate::impls::{Family, Instances};
use crate::runtime::Crosses;
use crate::shape::Projected;

macro_rules! modes {
    ($([$($g:tt)*] $m:ty => $opened:ty;)+) => {$(
        impl<$($g)*> sealed::Sealed for $m {}

        impl<R: Runtime, $($g)*> Mode<R> for $m {
            type Opened<'c> = $opened
            where
                Self: 'c,
                R: 'c;
        }
    )+};
}

modes! {
    [T: Crosses] ByValue<T> => T;
    [T] ByUniform<T> => Uniform<R, T>;
    [T: Crosses] ByRef<T> => Guard<'c, R::Shared<'c, T>>;
    [T: Crosses] ByMut<T> => GuardMut<'c, R::Exclusive<'c, T>>;
    [T: Crosses] BySlice<T> => Guard<'c, R::SharedSlice<'c, T>>;
    [T: Crosses] BySliceMut<T> => GuardMut<'c, R::ExclusiveSlice<'c, T>>;
    [] ByStr => Guard<'c, R::SharedStr<'c>>;
}

impl<P> sealed::Sealed for ByProjection<P> {}
impl<P> sealed::Sealed for ByProjectionMut<P> {}
impl<A, U, E> sealed::Sealed for ByCallback<A, U, E> {}
impl<F, I> sealed::Sealed for ByInstance<F, I> {}
impl<F, Ts, L> sealed::Sealed for ByInstanceOf<F, Ts, L> {}

impl<R: Runtime, P: Projected<R>> Mode<R> for ByProjection<P> {
    type Opened<'c> = P::Ref<'c>
    where
        Self: 'c,
        R: 'c;
}

impl<R: Runtime, P: Projected<R>> Mode<R> for ByProjectionMut<P> {
    type Opened<'c> = P::Mut<'c>
    where
        Self: 'c,
        R: 'c;
}

impl<R: Runtime, A, U, E> Mode<R> for ByCallback<A, U, E> {
    type Opened<'c> = Callback<'c, 'c, A, U, E, R>
    where
        Self: 'c,
        R: 'c;
}

impl<R: Instances, F: Family<Receiver = Steps>, I> RequirementMode<R> for ByInstance<F, I> {
    type Opened<'c> = Instance<R, F, I>
    where
        Self: 'c,
        R: 'c;
}

impl<R: Instances, F: Family<Receiver = Reads>, Ts, L> RequirementMode<R> for ByInstanceOf<F, Ts, L> {
    type Opened<'c> = InstanceOf<R, F, Ts, L>
    where
        Self: 'c,
        R: 'c;
}

/// Tuples of modes up to twelve, the arity the crossing's tuples stop at.
macro_rules! tuples {
    ($($n:literal ($($m:ident),*))+) => {$(
        impl<$($m),*> sealed::Sealed for ($($m,)*) {}

        impl<R: Runtime, $($m: Mode<R>),*> Args<R> for ($($m,)*) {
            type Run = [R::Value; $n];
            type Opened<'c> = ($($m::Opened<'c>,)*)
            where
                Self: 'c,
                R: 'c;
        }

        impl<R: Runtime, $($m: RequirementMode<R>),*> Requires<R> for ($($m,)*) {
            type Opened<'c> = ($($m::Opened<'c>,)*)
            where
                Self: 'c,
                R: 'c;
        }
    )+};
}

tuples! {
    0 ()
    1 (A)
    2 (A, B)
    3 (A, B, C)
    4 (A, B, C, D)
    5 (A, B, C, D, E)
    6 (A, B, C, D, E, F)
    7 (A, B, C, D, E, F, G)
    8 (A, B, C, D, E, F, G, H)
    9 (A, B, C, D, E, F, G, H, I)
    10 (A, B, C, D, E, F, G, H, I, J)
    11 (A, B, C, D, E, F, G, H, I, J, K)
    12 (A, B, C, D, E, F, G, H, I, J, K, L)
}
