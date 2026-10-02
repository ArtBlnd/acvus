//! The one door of one call (crossing §1): it hands the body the call's
//! arguments and the impls its site chose, each a value to open once by its
//! parameter's mode at the site's settled type, and its executor, inside a
//! closure (G1, G9, G10).

use std::marker::PhantomData;

use crate::executor::Executor;
use crate::done::Done;
use crate::runtime::Runtime;

/// Proof that a read or a result is the crossing's, at the type the checker
/// settled for the site of the call: only this crate makes one, within a
/// call. It bounds no lifetime: what a read lends is bounded by the runtime's
/// borrow, which the call's executor gives.
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

    /// Runs the returning body with the call's arguments and the impls its
    /// site chose, each to be opened once by its parameter's mode.
    pub fn open(
        self,
        body: impl FnOnce(Executor<'c, R>, [Argument<'c, R>; N], [Impl<'c, R>; M]) -> Done<R>,
    ) -> Done<R> {
        let (args, impls) = self.given();
        body(Executor::of_the_entry(self.runtime), args, impls)
    }

    /// Runs the awaited body as `open` runs the returning one. The body opens
    /// its arguments before it makes its future, so the future holds them
    /// and nothing else of the entry.
    pub fn open_awaited<Fut>(
        self,
        body: impl FnOnce(Executor<'c, R>, [Argument<'c, R>; N], [Impl<'c, R>; M]) -> Fut,
    ) -> impl Future<Output = Done<R>> + Send + 'c
    where
        Fut: Future<Output = Done<R>> + Send + 'c,
    {
        let (args, impls) = self.given();
        body(Executor::of_the_entry(self.runtime), args, impls)
    }

    fn given(&self) -> ([Argument<'c, R>; N], [Impl<'c, R>; M]) {
        let (runtime, at) = (self.runtime, Settled::of_the_entry());
        let args = self.args.map(|value| Argument { runtime, value, at });
        let impls = self.site.impls.map(|value| Impl { runtime, value, at });
        (args, impls)
    }
}

/// One argument of a call, opened once by its parameter's mode: only the
/// entry makes one, and opening it consumes it.
pub struct Argument<'c, R: Runtime> {
    runtime: &'c R,
    value: R::Value,
    at: Settled<'c>,
}

impl<'c, R: Runtime> Argument<'c, R> {
    /// The argument opened by the mode `P` its parameter's Rust type picks.
    pub fn open<P: Mode<R> + 'c>(self) -> P::Opened<'c> {
        P::open(self.runtime, self.value, self.at)
    }
}

/// One impl a call's site chose for a requirement (impls §3), opened once by
/// its requirement's mode: only the entry makes one, and opening it
/// consumes it.
pub struct Impl<'c, R: Runtime> {
    runtime: &'c R,
    value: R::Value,
    at: Settled<'c>,
}

impl<'c, R: Runtime> Impl<'c, R> {
    /// The impl opened by the mode `Q` its requirement's Rust type picks.
    pub fn open<Q: RequirementMode<R> + 'c>(self) -> Q::Opened<'c> {
        Q::open(self.runtime, self.value, self.at)
    }
}

/// A parameter's mode, chosen by its Rust type (macros §4): how the entry
/// opens its argument as the Rust argument the handler's parameter takes,
/// within the call `'c`.
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
}

use mode::*;

use crate::boundary::loaned::{Loaned, LoanedMut};
use crate::boundary::uniform::Uniform;
use crate::boundary::seal::Crossing;
use crate::runtime::{Crosses, RefKind};

macro_rules! modes {
    ($r:ident; $([$($g:tt)*] $m:ty => $opened:ty, |$runtime:ident, $v:ident, $at:ident| $open:expr;)+) => {$(
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
        }
    )+};
}

modes! {
    R;
    [T: Crosses<R>] ByValue<T> => T,
        |runtime, v, at| T::take(runtime, v, at);
    [T: Crosses<R>] ByRef<T> => T::Open<'c>,
        |runtime, v, at| T::open_ref(runtime, v, at);
    [T: Crosses<R>] ByMut<T> => T::OpenMut<'c>,
        |runtime, v, at| T::open_ref_mut(runtime, v, at);
    [T] ByUniform<T> => Uniform<R, T>,
        |_runtime, v, _at| Uniform::of_the_crossing(v);
    [T] BySlice<T> => &'c [Uniform<R, T>],
        |runtime, v, at| <R::Ref as RefKind<R>>::target(runtime, v, at);
    [T] BySliceMut<T> => &'c mut [Uniform<R, T>],
        |runtime, v, at| <R::Ref as RefKind<R>>::target_mut(runtime, v, at);
    [] ByStr => &'c str,
        |runtime, v, at| <R::Ref as RefKind<R>>::target(runtime, v, at);
    [T] ByLoaned<T> => Loaned<'c, T, R>,
        |_runtime, v, _at| Loaned::of_the_entry(v);
    [T] ByLoanedMut<T> => LoanedMut<'c, T, R>,
        |_runtime, v, _at| LoanedMut::of_the_entry(v);
}
