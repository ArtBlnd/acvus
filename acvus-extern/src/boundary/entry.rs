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

/// A parameter's mode, chosen by its Rust type (macros §4): how the entry
/// opens its argument.
pub trait Mode<R: Runtime>: Crossing {
    type Opened<'c>
    where
        Self: 'c,
        R: 'c;

    /// How the checker passes the argument this mode opens.
    #[doc(hidden)]
    const LENT: Option<Mutability>;

    /// Whether the handler may call through the argument, which only an
    /// awaited body may (macros §1): a callback.
    #[doc(hidden)]
    const CALLS: bool;

    #[doc(hidden)]
    fn open<'c>(runtime: &'c R, v: &'c R::Value, at: Settled<'c>) -> Self::Opened<'c>
    where
        Self: 'c,
        R: 'c;

    /// The Rust argument the handler's parameter takes, borrowed for `'h`.
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

/// The argument parameters of a handler, a tuple of modes, one position each.
/// `Run` is the call's argument run at that arity, `[R::Value; N]`.
pub trait Args<R: Runtime>: Crossing {
    type Run;

    /// Whether any argument is one the handler may call through, which
    /// only an awaited body may (macros §1).
    const CALLS: bool;
    type Opened<'c>
    where
        Self: 'c,
        R: 'c;

    /// Each argument opened by its position's mode.
    #[doc(hidden)]
    fn open<'c>(
        runtime: &'c R,
        args: &'c [R::Value],
        lent: &[Option<Mutability>],
        at: Settled<'c>,
    ) -> Self::Opened<'c>
    where
        Self: 'c,
        R: 'c;
}

/// The requirement parameters of a handler, a tuple of requirement modes.
pub trait Requires<R: Runtime>: Crossing {
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
    ($r:ident; $([$($g:tt)*] $m:ty => $opened:ty, $lent:expr, $calls:expr, $arg:ty, $passed:ident, |$runtime:ident, $v:ident, $at:ident| $open:expr;)+) => {$(
        impl<$r: Runtime, $($g)*> Mode<$r> for $m {
            type Opened<'c> = $opened
            where
                Self: 'c,
                $r: 'c;

            const LENT: Option<Mutability> = $lent;

            const CALLS: bool = $calls;

            fn open<'c>($runtime: &'c $r, $v: &'c $r::Value, $at: Settled<'c>) -> Self::Opened<'c>
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

const SHARED: Option<Mutability> = Some(Mutability::Shared);
const EXCLUSIVE: Option<Mutability> = Some(Mutability::Exclusive);

modes! {
    R;
    [T: Crosses<R>] ByValue<T> => T, None, false, T, moved,
        |runtime, v, at| T::take(runtime, *v, at);
    [T: Crosses<R>] ByRef<T> => Guard<'c, T::Open<'c>>, SHARED, false, T::Open<'c>, viewed,
        |runtime, v, at| Guard::of_the_entry(T::open_ref(runtime, *v, at));
    [T: Crosses<R>] ByMut<T> => GuardMut<'c, T::OpenMut<'c>>, EXCLUSIVE, false, T::OpenMut<'c>, viewed,
        |runtime, v, at| GuardMut::of_the_entry(T::open_ref_mut(runtime, *v, at));
    [T] ByUniform<T> => Uniform<R, T>, None, false, Uniform<R, T>, moved,
        |_runtime, v, _at| Uniform::of_the_crossing(*v);
    [T] BySlice<T> => Guard<'c, &'c [Uniform<R, T>]>, SHARED, false, &'c [Uniform<R, T>], viewed,
        |runtime, v, at| Guard::of_the_entry(<R::Ref as RefKind<R>>::target(runtime, *v, at));
    [T] BySliceMut<T> => GuardMut<'c, &'c mut [Uniform<R, T>]>, EXCLUSIVE, false, &'c mut [Uniform<R, T>], viewed,
        |runtime, v, at| GuardMut::of_the_entry(<R::Ref as RefKind<R>>::target_mut(runtime, *v, at));
    [] ByStr => Guard<'c, &'c str>, SHARED, false, &'c str, viewed,
        |runtime, v, at| Guard::of_the_entry(<R::Ref as RefKind<R>>::target(runtime, *v, at));
    [A, U, E] ByCallback<A, U, E> => Callback<'c, 'c, A, U, E, R>, None, true, Callback<'c, 'c, A, U, E, R>, moved,
        |_runtime, v, _at| Callback::of_the_entry(*v);
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
        impl<R: Runtime, $($m: Mode<R>),*> Args<R> for ($($m,)*) {
            type Run = [R::Value; $n];
            const CALLS: bool = false $(|| $m::CALLS)*;
            type Opened<'c> = ($($m::Opened<'c>,)*)
            where
                Self: 'c,
                R: 'c;

            #[allow(unused_variables, clippy::unused_unit)]
            fn open<'c>(
                runtime: &'c R,
                args: &'c [R::Value],
                lent: &[Option<Mutability>],
                at: Settled<'c>,
            ) -> Self::Opened<'c>
            where
                Self: 'c,
                R: 'c,
            {
                let run: &'c Self::Run = args.try_into().expect("a call passes one value per parameter");
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

mode_tuples! { 0 () }
tuples!(mode_tuples);
