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
    args: &'c mut [R::Value],
    site: &'c SiteShape<R>,
}

impl<'c, R: Runtime> Entry<'c, R> {
    /// The entry of one call. It is safe: the checker's decision is trusted
    /// only by the runtime's own typed reads (safety §3), and only the runtime
    /// holds values to pass here, since the host sees none (crossing §6).
    pub fn new(runtime: &'c R, args: &'c mut [R::Value], site: &'c SiteShape<R>) -> Self {
        Entry { runtime, args, site }
    }

    /// Opens the arguments `A`, one per position, and the requirements `Q`,
    /// from the site's impls, and runs the returning body.
    pub fn open<A, Q>(self, body: impl FnOnce(Returning<'c, R>, A::Opened<'c>, Q::Opened<'c>) -> Done<R>) -> Done<R>
    where
        A: Args<R> + 'c,
        Q: Requires<R> + 'c,
    {
        let at = Settled::of_the_entry();
        let args = A::open(self.runtime, self.args, &self.site.lent, at);
        let requirements = Q::open(self.runtime, &self.site.impls, at);
        body(Returning::of_the_entry(self.runtime, self.site), args, requirements)
    }

    /// Opens the arguments and the requirements as `open` does, and runs the
    /// awaited body. The arguments are opened before the body's future is
    /// made, so the future holds them and nothing else of the entry.
    pub fn open_awaited<A, Q, Fut>(
        self,
        body: impl FnOnce(Awaited<'c, R>, A::Opened<'c>, Q::Opened<'c>) -> Fut,
    ) -> impl Future<Output = Done<R>> + Send + 'c
    where
        A: Args<R> + 'c,
        Q: Requires<R> + 'c,
        Fut: Future<Output = Done<R>> + Send + 'c,
    {
        let at = Settled::of_the_entry();
        let args = A::open(self.runtime, self.args, &self.site.lent, at);
        let requirements = Q::open(self.runtime, &self.site.impls, at);
        body(Awaited::of_the_entry(self.runtime, self.site), args, requirements)
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

    /// How the checker passes the argument this mode opens.
    #[doc(hidden)]
    const LENT: Option<Mutability>;

    #[doc(hidden)]
    fn open<'c>(runtime: &'c R, v: &'c mut R::Value, at: Settled<'c>) -> Self::Opened<'c>
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

    #[doc(hidden)]
    fn open<'c>(runtime: &'c R, impl_value: R::Value, at: Settled<'c>) -> Self::Opened<'c>
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

    /// Each argument opened by its position's mode.
    #[doc(hidden)]
    fn open<'c>(
        runtime: &'c R,
        args: &'c mut [R::Value],
        lent: &[Option<Mutability>],
        at: Settled<'c>,
    ) -> Self::Opened<'c>
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

    /// Each impl the site chose opened by its requirement's mode.
    #[doc(hidden)]
    fn open<'c>(runtime: &'c R, impls: &'c [R::Value], at: Settled<'c>) -> Self::Opened<'c>
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
    /// Lent `&T`.
    pub struct ByRef<T>(PhantomData<fn() -> T>);
    /// Lent `&mut T`.
    pub struct ByMut<T>(PhantomData<fn() -> T>);
    /// A program's array lent as `&[Uniform<R, T>]` or `&mut [Uniform<R, T>]`,
    /// its argument a reference to the array or to a run of it, read through
    /// `target`.
    pub struct BySlice<T>(PhantomData<fn() -> T>);
    pub struct BySliceMut<T>(PhantomData<fn() -> T>);
    /// Text lent as `&str`, its argument a reference to a `String` or to a
    /// range of one, read through `target`.
    pub struct ByStr;
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

macro_rules! modes {
    ($([$($g:tt)*] $m:ty => $opened:ty, $lent:expr, |$runtime:ident, $v:ident, $at:ident| $open:expr;)+) => {$(
        impl<$($g)*> sealed::Sealed for $m {}

        impl<R: Runtime, $($g)*> Mode<R> for $m {
            type Opened<'c> = $opened
            where
                Self: 'c,
                R: 'c;

            const LENT: Option<Mutability> = $lent;

            fn open<'c>($runtime: &'c R, $v: &'c mut R::Value, $at: Settled<'c>) -> Self::Opened<'c>
            where
                Self: 'c,
                R: 'c,
            {
                $open
            }
        }
    )+};
}

const SHARED: Option<Mutability> = Some(Mutability::Shared);
const EXCLUSIVE: Option<Mutability> = Some(Mutability::Exclusive);

modes! {
    [T: Crosses] ByValue<T> => T, None,
        |runtime, v, at| Uniform::<R, T>::of_the_crossing(*v).take(runtime, at);
    [T] ByUniform<T> => Uniform<R, T>, None,
        |_runtime, v, _at| Uniform::of_the_crossing(*v);
    [T: Crosses] ByRef<T> => Guard<'c, R::Shared<'c, T>>, SHARED,
        |runtime, v, at| Guard::of_the_entry(runtime.shared(v, at));
    [T: Crosses] ByMut<T> => GuardMut<'c, R::Exclusive<'c, T>>, EXCLUSIVE,
        |runtime, v, at| GuardMut::of_the_entry(runtime.exclusive(v, at));
    [T] BySlice<T> => Guard<'c, &'c [Uniform<R, T>]>, SHARED,
        |runtime, v, at| Guard::of_the_entry(runtime.target(*v, at));
    [T] BySliceMut<T> => GuardMut<'c, &'c mut [Uniform<R, T>]>, EXCLUSIVE,
        |runtime, v, at| GuardMut::of_the_entry(runtime.target_mut(*v, at));
    [] ByStr => Guard<'c, &'c str>, SHARED,
        |runtime, v, at| Guard::of_the_entry(runtime.target(*v, at));
    [A, U, E] ByCallback<A, U, E> => Callback<'c, 'c, A, U, E, R>, None,
        |_runtime, v, _at| Callback::of_the_entry(*v);
}

impl<F, I> sealed::Sealed for ByInstance<F, I> {}
impl<F, Ts, L> sealed::Sealed for ByInstanceOf<F, Ts, L> {}

impl<R: Instances, F: Family<Receiver = Steps>, I> RequirementMode<R> for ByInstance<F, I> {
    type Opened<'c> = Instance<R, F, I>
    where
        Self: 'c,
        R: 'c;

    fn open<'c>(_runtime: &'c R, _impl_value: R::Value, _at: Settled<'c>) -> Instance<R, F, I>
    where
        Self: 'c,
        R: 'c,
    {
        todo!("the impl cut: an Instance pairs the impl with the receiver it was chosen for")
    }
}

impl<R: Instances, F: Family<Receiver = Reads>, Ts, L> RequirementMode<R> for ByInstanceOf<F, Ts, L> {
    type Opened<'c> = InstanceOf<R, F, Ts, L>
    where
        Self: 'c,
        R: 'c;

    fn open<'c>(_runtime: &'c R, impl_value: R::Value, _at: Settled<'c>) -> InstanceOf<R, F, Ts, L>
    where
        Self: 'c,
        R: 'c,
    {
        InstanceOf::of_the_entry(impl_value)
    }
}

/// Tuples of modes up to twelve, the arity the crossing's tuples stop at.
macro_rules! tuples {
    ($($n:literal ($($m:ident $v:ident),*))+) => {$(
        impl<$($m),*> sealed::Sealed for ($($m,)*) {}

        impl<R: Runtime, $($m: Mode<R>),*> Args<R> for ($($m,)*) {
            type Run = [R::Value; $n];
            type Opened<'c> = ($($m::Opened<'c>,)*)
            where
                Self: 'c,
                R: 'c;

            #[allow(unused_variables, clippy::unused_unit)]
            fn open<'c>(
                runtime: &'c R,
                args: &'c mut [R::Value],
                lent: &[Option<Mutability>],
                at: Settled<'c>,
            ) -> Self::Opened<'c>
            where
                Self: 'c,
                R: 'c,
            {
                let run: &'c mut Self::Run = args.try_into().expect("a call passes one value per parameter");
                let modes: [Option<Mutability>; $n] = [$($m::LENT),*];
                debug_assert_eq!(lent, modes, "the site passes each argument as its parameter's mode opens it");
                let [$($v),*] = run;
                ($($m::open(runtime, $v, at),)*)
            }
        }

        impl<R: Runtime, $($m: RequirementMode<R>),*> Requires<R> for ($($m,)*) {
            type Opened<'c> = ($($m::Opened<'c>,)*)
            where
                Self: 'c,
                R: 'c;

            #[allow(unused_variables, clippy::unused_unit)]
            fn open<'c>(runtime: &'c R, impls: &'c [R::Value], at: Settled<'c>) -> Self::Opened<'c>
            where
                Self: 'c,
                R: 'c,
            {
                let run: &[R::Value; $n] = impls.try_into().expect("a site chooses one impl per requirement");
                let [$($v),*] = *run;
                ($($m::open(runtime, $v, at),)*)
            }
        }
    )+};
}

tuples! {
    0 ()
    1 (A a)
    2 (A a, B b)
    3 (A a, B b, C c)
    4 (A a, B b, C c, D d)
    5 (A a, B b, C c, D d, E e)
    6 (A a, B b, C c, D d, E e, F f)
    7 (A a, B b, C c, D d, E e, F f, G g)
    8 (A a, B b, C c, D d, E e, F f, G g, H h)
    9 (A a, B b, C c, D d, E e, F f, G g, H h, I i)
    10 (A a, B b, C c, D d, E e, F f, G g, H h, I i, J j)
    11 (A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k)
    12 (A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k, L l)
}
