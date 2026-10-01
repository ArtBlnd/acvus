//! The one door of one call (crossing §1): it reads every argument at the
//! site's settled type, opens it by its mode, and hands the body its
//! arguments and its context inside a closure (G1, G9, G10).

use std::marker::PhantomData;

use crate::context::{Awaited, Returning};
use crate::done::Done;
use crate::runtime::Runtime;

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

/// One call site as the checker settled it: the impl values chosen for the
/// handler's `M` requirements, in the handler's order (impls §3), a value of
/// static size. The runtime builds it once, at preparation.
pub struct SiteShape<R: Runtime, const M: usize> {
    impls: [R::Value; M],
}

impl<R: Runtime, const M: usize> SiteShape<R, M> {
    pub fn new(impls: [R::Value; M]) -> Self {
        SiteShape { impls }
    }
}

/// One call's door. Its arguments are the run of the call's `N` argument
/// positions, a value of static size, so the entry opens it without a
/// compare and a call inlined into its caller keeps it in registers.
pub struct Entry<'c, R: Runtime, const N: usize, const M: usize> {
    runtime: &'c R,
    args: [R::Value; N],
    site: &'c SiteShape<R, M>,
}

impl<'c, R: Runtime, const N: usize, const M: usize> Entry<'c, R, N, M> {
    /// The entry of one call. It is safe: the checker's decision is trusted
    /// only by the runtime's own typed reads (safety §3), and only the runtime
    /// holds values to pass here, since the host sees none (crossing §6).
    pub fn new(runtime: &'c R, args: [R::Value; N], site: &'c SiteShape<R, M>) -> Self {
        Entry { runtime, args, site }
    }

    /// Opens the arguments `A`, one per position, and the requirements `Q`,
    /// from the site's impls, and runs the returning body.
    pub fn open<A, Q>(self, body: impl FnOnce(Returning<'c, R>, A::Opened<'c>, Q::Opened<'c>) -> Done<R>) -> Done<R>
    where
        A: Args<R, N> + 'c,
        Q: Requires<R, M> + 'c,
    {
        let at = Settled::of_the_entry();
        let args = A::open(self.runtime, self.args, at);
        let requirements = Q::open(self.runtime, self.site.impls, at);
        body(Returning::of_the_entry(self.runtime), args, requirements)
    }

    /// Opens the arguments and the requirements as `open` does, and runs the
    /// awaited body. The arguments are opened before the body's future is
    /// made, so the future holds them and nothing else of the entry.
    pub fn open_awaited<A, Q, Fut>(
        self,
        body: impl FnOnce(Awaited<'c, R>, A::Opened<'c>, Q::Opened<'c>) -> Fut,
    ) -> impl Future<Output = Done<R>> + Send + 'c
    where
        A: Args<R, N> + 'c,
        Q: Requires<R, M> + 'c,
        Fut: Future<Output = Done<R>> + Send + 'c,
    {
        let at = Settled::of_the_entry();
        let args = A::open(self.runtime, self.args, at);
        let requirements = Q::open(self.runtime, self.site.impls, at);
        body(Awaited::of_the_entry(self.runtime), args, requirements)
    }
}

/// A parameter's mode, chosen by its Rust type (macros §4): how the entry
/// opens its argument.
pub trait Mode<R: Runtime>: Crossing {
    type Opened<'c>
    where
        Self: 'c,
        R: 'c;

    #[doc(hidden)]
    fn open<'c>(runtime: &'c R, v: R::Value, at: Settled<'c>) -> Self::Opened<'c>
    where
        Self: 'c,
        R: 'c;

    /// The Rust argument the handler's parameter takes.
    type Arg<'c>
    where
        Self: 'c,
        R: 'c;

    /// The handler's argument made from what the entry opened: a value moved
    /// on, or the projection a guard holds moved out of it.
    fn arg<'c>(opened: Self::Opened<'c>) -> Self::Arg<'c>
    where
        Self: 'c,
        R: 'c;
}

/// A requirement's mode: how the entry opens the impl the site chose.
pub trait RequirementMode<R: Runtime>: Crossing {
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

/// The argument parameters of a handler, a tuple of `N` modes, one position
/// each, opened from the call's argument run `[R::Value; N]`.
pub trait Args<R: Runtime, const N: usize>: Crossing {
    type Opened<'c>
    where
        Self: 'c,
        R: 'c;

    /// Each argument opened by its position's mode.
    #[doc(hidden)]
    fn open<'c>(runtime: &'c R, args: [R::Value; N], at: Settled<'c>) -> Self::Opened<'c>
    where
        Self: 'c,
        R: 'c;
}

/// The requirement parameters of a handler, a tuple of requirement modes.
pub trait Requires<R: Runtime, const M: usize>: Crossing {
    type Opened<'c>
    where
        Self: 'c,
        R: 'c;

    /// Each impl the site chose opened by its requirement's mode.
    #[doc(hidden)]
    fn open<'c>(runtime: &'c R, impls: [R::Value; M], at: Settled<'c>) -> Self::Opened<'c>
    where
        Self: 'c,
        R: 'c;
}

/// The parameter modes (macros §2, crossing §2).
pub mod mode {
    use std::marker::PhantomData;

    /// Taken by value where the handler needs the value itself: taken at once
    /// by its crossing.
    pub struct ByValue<T>(PhantomData<fn() -> T>);
    /// Held by value where the handler writes `Uniform<R, T>`: read in place,
    /// released if dropped, taken only by `into_inner`.
    pub struct ByUniform<T>(PhantomData<fn() -> T>);
    /// Lent `&T`, opened as `T`'s view of what the reference names.
    pub struct ByRef<T>(PhantomData<fn() -> T>);
    /// Lent `&mut T`, opened as `T`'s exclusive view.
    pub struct ByMut<T>(PhantomData<fn() -> T>);
    /// A program's array lent as `&[Uniform<R, T>]` or `&mut [Uniform<R, T>]`,
    /// its argument a reference to the array or to a run of it, read through
    /// `target`.
    pub struct BySlice<T>(PhantomData<fn() -> T>);
    pub struct BySliceMut<T>(PhantomData<fn() -> T>);
    /// Text lent as `&str`, its argument a reference to a `String` or to a
    /// range of one, read through `target`.
    pub struct ByStr;
    /// A reference kept as it is, a carrier: `Loaned<'a, T, R>` for `&T`,
    /// `LoanedMut<'a, T, R>` for `&mut T`, for a result to hold the loan
    /// (limitations §3).
    pub struct ByLoaned<T>(PhantomData<fn() -> T>);
    pub struct ByLoanedMut<T>(PhantomData<fn() -> T>);
    /// A requirement owning its receiver, and one standing at a type.
    pub struct ByInstance<F, I>(PhantomData<fn() -> (F, I)>);
    pub struct ByInstanceOf<F, Ts, L>(PhantomData<fn() -> (F, Ts, L)>);
}

use mode::*;

use crate::boundary::instance::{Instance, InstanceOf};
use crate::boundary::loaned::{Loaned, LoanedMut};
use crate::boundary::uniform::Uniform;
use crate::guard::{Guard, GuardMut};
use crate::impls::receiver::{Reads, Steps};
use crate::impls::{Family, Instances};
use crate::boundary::seal::Crossing;
use crate::runtime::{Crosses, RefKind};

/// How a mode's opened value becomes the handler's argument: moved on, or the
/// projection a guard holds moved out of it.
macro_rules! pass {
    (moved, $opened:ident) => {
        $opened
    };
    (viewed, $opened:ident) => {
        $opened.into_inner()
    };
}

macro_rules! modes {
    ($r:ident; $([$($g:tt)*] $m:ty => $opened:ty, $arg:ty, $passed:ident, |$runtime:ident, $v:ident, $at:ident| $open:expr;)+) => {$(
        impl<$r: Runtime, $($g)*> Mode<$r> for $m {
            type Opened<'c> = $opened
            where
                Self: 'c,
                $r: 'c;

            fn open<'c>($runtime: &'c $r, $v: $r::Value, $at: Settled<'c>) -> Self::Opened<'c>
            where
                Self: 'c,
                $r: 'c,
            {
                $open
            }

            type Arg<'c>
                = $arg
            where
                Self: 'c,
                $r: 'c;

            fn arg<'c>(opened: Self::Opened<'c>) -> Self::Arg<'c>
            where
                Self: 'c,
                $r: 'c,
            {
                pass!($passed, opened)
            }
        }
    )+};
}

modes! {
    R;
    [T: Crosses<R>] ByValue<T> => T, T, moved,
        |runtime, v, at| T::take(runtime, v, at);
    [T: Crosses<R>] ByRef<T> => Guard<'c, T::Open<'c>>, T::Open<'c>, viewed,
        |runtime, v, at| Guard::of_the_entry(T::open_ref(runtime, v, at));
    [T: Crosses<R>] ByMut<T> => GuardMut<'c, T::OpenMut<'c>>, T::OpenMut<'c>, viewed,
        |runtime, v, at| GuardMut::of_the_entry(T::open_ref_mut(runtime, v, at));
    [T] ByUniform<T> => Uniform<R, T>, Uniform<R, T>, moved,
        |_runtime, v, _at| Uniform::of_the_crossing(v);
    [T] BySlice<T> => Guard<'c, &'c [Uniform<R, T>]>, &'c [Uniform<R, T>], viewed,
        |runtime, v, at| Guard::of_the_entry(<R::Ref as RefKind<R>>::target(runtime, v, at));
    [T] BySliceMut<T> => GuardMut<'c, &'c mut [Uniform<R, T>]>, &'c mut [Uniform<R, T>], viewed,
        |runtime, v, at| GuardMut::of_the_entry(<R::Ref as RefKind<R>>::target_mut(runtime, v, at));
    [] ByStr => Guard<'c, &'c str>, &'c str, viewed,
        |runtime, v, at| Guard::of_the_entry(<R::Ref as RefKind<R>>::target(runtime, v, at));
    [T] ByLoaned<T> => Loaned<'c, T, R>, Loaned<'c, T, R>, moved,
        |_runtime, v, _at| Loaned::of_the_entry(v);
    [T] ByLoanedMut<T> => LoanedMut<'c, T, R>, LoanedMut<'c, T, R>, moved,
        |_runtime, v, _at| LoanedMut::of_the_entry(v);
}

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

/// The argument and requirement parameters of each arity: none, and each
/// tuple of modes.
macro_rules! mode_tuples {
    ($($n:literal ($($m:ident $v:ident $i:tt),*))+) => {$(
        impl<R: Runtime, $($m: Mode<R>),*> Args<R, $n> for ($($m,)*) {
            type Opened<'c> = ($($m::Opened<'c>,)*)
            where
                Self: 'c,
                R: 'c;

            #[allow(unused_variables, clippy::unused_unit)]
            fn open<'c>(runtime: &'c R, args: [R::Value; $n], at: Settled<'c>) -> Self::Opened<'c>
            where
                Self: 'c,
                R: 'c,
            {
                let [$($v),*] = args;
                ($($m::open(runtime, $v, at),)*)
            }
        }

        impl<R: Runtime, $($m: RequirementMode<R>),*> Requires<R, $n> for ($($m,)*) {
            type Opened<'c> = ($($m::Opened<'c>,)*)
            where
                Self: 'c,
                R: 'c;

            #[allow(unused_variables, clippy::unused_unit)]
            fn open<'c>(runtime: &'c R, impls: [R::Value; $n], at: Settled<'c>) -> Self::Opened<'c>
            where
                Self: 'c,
                R: 'c,
            {
                let [$($v),*] = impls;
                ($($m::open(runtime, $v, at),)*)
            }
        }
    )+};
}

mode_tuples! { 0 () }
tuples!(mode_tuples);
