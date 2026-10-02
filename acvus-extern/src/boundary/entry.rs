//! The one door of one call (crossing §1): it hands the body the call's
//! arguments and the impls its site chose, each a value to open once by its
//! parameter's Rust type or its requirement's mode at the site's settled
//! type, and its executor, inside a
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
    /// site chose, each to be opened once at its parameter's Rust type or
    /// by its requirement's mode.
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
        let impls = self.site.impls.each_ref().map(|place| Impl { runtime, place, at });
        (args, impls)
    }
}

/// One argument of a call, opened once at its parameter's Rust type: only the
/// entry makes one, and opening it consumes it.
pub struct Argument<'c, R: Runtime> {
    runtime: &'c R,
    value: R::Value,
    at: Settled<'c>,
}

impl<'c, R: Runtime> Argument<'c, R> {
    /// The argument opened as the Rust type `P` its parameter writes, at the
    /// call's loan (crossing §2).
    pub fn open<P: Open<'c, R>>(self) -> P {
        P::open(self.runtime, self.value, self.at)
    }

    /// The argument opened with the impl its site chose for it, by the
    /// mode `B` of a parameter that takes both (impls §3).
    pub fn open_with<B: PairedMode<R> + 'c>(self, r#impl: Impl<'c, R>) -> B::Opened<'c> {
        B::open(self.runtime, self.value, r#impl.place, self.at)
    }
}

/// One impl a call's site chose for a requirement (impls §3), opened once by
/// its requirement's mode: only the entry makes one, and opening it
/// consumes it. The impl's value is the site's, lent for the call.
pub struct Impl<'c, R: Runtime> {
    runtime: &'c R,
    place: &'c R::Value,
    at: Settled<'c>,
}

impl<'c, R: Runtime> Impl<'c, R> {
    /// The impl opened by the mode `Q` its requirement's Rust type picks.
    pub fn open<Q: RequirementMode<R> + 'c>(self) -> Q::Opened<'c> {
        Q::open(self.runtime, self.place, self.at)
    }
}

/// A requirement's mode: how the entry opens the impl the site chose.
pub trait RequirementMode<R: Runtime>: Crossing {
    type Opened<'c>
    where
        Self: 'c,
        R: 'c;

    /// Opens the impl the site chose, its value the site's, lent for the
    /// call.
    #[doc(hidden)]
    fn open<'c>(runtime: &'c R, r#impl: &'c R::Value, at: Settled<'c>) -> Self::Opened<'c>
    where
        Self: 'c,
        R: 'c;
}

/// The mode of a parameter that takes an argument and the impl its site
/// chose for it, as one value (impls §3).
pub trait PairedMode<R: Runtime>: Crossing {
    type Opened<'c>
    where
        Self: 'c,
        R: 'c;

    /// Opens the argument with the impl the site chose for it, its value
    /// the site's, lent for the call.
    #[doc(hidden)]
    fn open<'c>(runtime: &'c R, argument: R::Value, r#impl: &'c R::Value, at: Settled<'c>) -> Self::Opened<'c>
    where
        Self: 'c,
        R: 'c;
}

/// The requirements' modes.
pub mod mode {
    use std::marker::PhantomData;

    /// A requirement held as a handle at a type, `&InstanceOf<R, G, Ts>`,
    /// the site's impl value lent for the call (impls §3).
    pub struct ByInstanceOf<G, Ts>(PhantomData<fn() -> (G, Ts)>);
    /// A receiver and the impl the site chose for it, owned as one value,
    /// `Instance<R, G, Ts>`: the argument taken, and the impl given its own
    /// value (impls §3).
    pub struct ByInstance<G, Ts>(PhantomData<fn() -> (G, Ts)>);
}

use mode::*;

use crate::boundary::seal::{Crossing, Families};
use crate::boundary::uniform::Uniform;
use crate::callback::CallsBack;
use crate::instance::{Holds, Instance, InstanceOf};
use crate::runtime::Open;

impl<R: Runtime, G: Families, Ts> RequirementMode<R> for ByInstanceOf<G, Ts> {
    type Opened<'c>
        = &'c InstanceOf<R, G, Ts>
    where
        Self: 'c,
        R: 'c;

    fn open<'c>(_runtime: &'c R, r#impl: &'c R::Value, _at: Settled<'c>) -> Self::Opened<'c>
    where
        Self: 'c,
        R: 'c,
    {
        InstanceOf::at_place(r#impl)
    }
}

impl<R: CallsBack, G, Ts> PairedMode<R> for ByInstance<G, Ts>
where
    G: Holds<R, Ts>,
{
    type Opened<'c>
        = Instance<R, G, Ts>
    where
        Self: 'c,
        R: 'c;

    fn open<'c>(runtime: &'c R, argument: R::Value, r#impl: &'c R::Value, at: Settled<'c>) -> Self::Opened<'c>
    where
        Self: 'c,
        R: 'c,
    {
        Instance::of_the_entry(Uniform::of_the_crossing(argument), Uniform::of_the_crossing(runtime.own(r#impl, at)))
    }
}
