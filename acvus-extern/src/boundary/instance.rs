//! A requirement's value (impls §3): the impl a call's site chose for a
//! requirement, a function value, held in one of two forms. A handle at the
//! type, `&InstanceOf`, is lent for the call and applied to any value of the
//! type, whatever way the generic function takes its receiver. An instance,
//! `Instance`, owns a receiver and the impl together: only the entry makes
//! one, from the argument and the impl the site chose; it calls that receiver
//! only, and may be moved and kept.
//!
//! Either is called as a function value is (runtime contract §2): returning
//! or awaited, given its arguments as runtime values and opening its result
//! (crossing §2-§4).

use std::marker::PhantomData;

use crate::boundary::entry::Settled;
use crate::boundary::seal::Families;
use crate::boundary::signed::{LayoutFact, Structural, StructuralFacts};
use crate::boundary::uniform::{Abstract, Boxed, Index, Uniform};
use crate::callback::{CallsBack, Keys};
use crate::decl::{DeclTy, Interner};
use crate::executor::Executor;
use crate::runtime::{Key, Open, Runtime};

/// How a generic function takes its receiver, its first parameter.
pub mod receiver {
    /// No receiver: the generic function takes no parameter.
    pub enum None {}
    /// By `&`: an instance lends its receiver shared to each call.
    pub enum Shared {}
    /// By `&mut`: an instance lends its receiver exclusively to each call.
    pub enum Mut {}
    /// By value: an instance gives its receiver to its one call.
    pub enum Value {}

    /// A way of taking a receiver: an instance has one to own.
    pub trait Receives {}

    impl Receives for Shared {}
    impl Receives for Mut {}
    impl Receives for Value {}

    /// A way of taking a receiver by lending it: an instance lends its own.
    pub trait Lent: Receives {
        #[doc(hidden)]
        fn lend<'a, R: crate::runtime::Runtime>(
            rt: &'a R,
            place: &'a mut R::Value,
            at: crate::boundary::entry::Settled<'a>,
        ) -> R::Value;
    }

    impl Lent for Shared {
        fn lend<'a, R: crate::runtime::Runtime>(
            rt: &'a R,
            place: &'a mut R::Value,
            at: crate::boundary::entry::Settled<'a>,
        ) -> R::Value {
            <R::Ref as crate::runtime::RefKind<R>>::of(rt, place, at)
        }
    }

    impl Lent for Mut {
        fn lend<'a, R: crate::runtime::Runtime>(
            rt: &'a R,
            place: &'a mut R::Value,
            at: crate::boundary::entry::Settled<'a>,
        ) -> R::Value {
            <R::Ref as crate::runtime::RefKind<R>>::of_mut(rt, place, at)
        }
    }
}

/// A generic function's family: how it takes its receiver. Only
/// `#[generic_fn]` writes one, for the generic function it declares:
///
/// ```compile_fail,E0277
/// use acvus_extern::instance::{GenericFunction, receiver};
///
/// struct HandWritten;
///
/// impl GenericFunction for HandWritten {
///     type Takes = receiver::Shared;
/// }
/// ```
pub trait GenericFunction: Families {
    type Takes;
}

/// A generic function's signature at the choice `Ts` of its variables, its
/// parts named as keys (crossing §2), read off the Rust signature at the
/// loan `'p`. `#[generic_fn]` implements it once, over every choice whose
/// type variables are held values.
///
/// # Safety
/// The implementer states that this is the generic function's signature at
/// `Ts`, read off its one declared signature: a requirer's call is made by
/// these keys, and an impl of the generic function at `Ts` opens a run made
/// by them. A choice of a Rust value for a type variable is refused, since a
/// Rust value is its type's own representation and never a variable's
/// (runtime contract §1):
///
/// ```compile_fail,E0599
/// use acvus_extern::executor::Executor;
/// use acvus_extern::instance::InstanceOf;
/// use acvus_extern::callback::CallsBack;
///
/// #[acvus_extern::generic_fn]
/// fn same<R: acvus_extern::runtime::Runtime, T: acvus_extern::decl::TypeVar<R>>(a: &T, b: &T) -> bool;
///
/// fn eq<R: CallsBack>(cx: &Executor<'_, R>, eq: &InstanceOf<R, same, (i64,)>) -> bool {
///     eq.call(cx, (&1, &2))
/// }
/// ```
pub unsafe trait FamilyAt<'p, R: Runtime, Ts>: GenericFunction {
    /// Every parameter's key, the receiver's first.
    type Params: Keys<'p, R>;
    type Result: Key<'p, R>;
    /// The parameters after the receiver.
    type Rest: Keys<'p, R>;
}

/// What a requirement of a generic function at the choice `Ts` states and
/// holds, at no loan: its pattern, and the receiver an instance owns.
///
/// # Safety
/// As `FamilyAt`'s, of the same signature.
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no signature at `{Ts}`",
    label = "not a choice of its variables",
    note = "a type variable is chosen as a held value, `Uniform<R, X>`, or a variable of the requirer's own; an effect or identity variable as one of the requirer's"
)]
pub unsafe trait Holds<R: Runtime, Ts>: GenericFunction {
    /// The receiver's own type, an instance's argument; `()` where the
    /// generic function takes no parameter.
    type Held: ?Sized;

    /// The type variables' declared types at `Ts`, a requirement's pattern.
    fn pattern(names: &Interner) -> Vec<DeclTy>;
}

/// A handle at a type: the impl value of the generic function `G` at `Ts`
/// the call's site chose, lent for the call (impls §3). It is the site's
/// value seen in place, so it is neither owned, copied nor made but by the
/// entry, and it borrows the call.
#[repr(transparent)]
pub struct InstanceOf<R: Runtime, G, Ts> {
    r#impl: R::Value,
    _family: PhantomData<fn() -> (G, Ts)>,
}

impl<R: Runtime, G, Ts> InstanceOf<R, G, Ts> {
    /// The site's impl value in `place` seen as a handle, borrowed as the
    /// place is.
    pub(crate) fn at_place(place: &R::Value) -> &Self {
        // SAFETY: `InstanceOf` is `repr(transparent)` over the runtime value,
        // its other field zero-sized.
        unsafe { &*std::ptr::from_ref(place).cast::<Self>() }
    }
}

impl<R: CallsBack, G, Ts> InstanceOf<R, G, Ts> {
    /// A returning call, its arguments given as runtime values for `'l` and
    /// its result opened, borrowing only what it lent (crossing §4).
    pub fn call<'l>(
        &'l self,
        cx: &'l Executor<'_, R>,
        args: <<G as FamilyAt<'l, R, Ts>>::Params as Keys<'l, R>>::Give,
    ) -> <<G as FamilyAt<'l, R, Ts>>::Result as Key<'l, R>>::Open
    where
        G: FamilyAt<'l, R, Ts>,
    {
        let (runtime, at) = (cx.runtime(), Settled::of_the_entry());
        let result = runtime.call_now(self.r#impl, G::Params::give_run(args, runtime, at), at);
        <<G::Result as Key<'l, R>>::Open as Open<'l, R>>::open(runtime, result, at)
    }

    /// An awaited call, its arguments given for the call future `'a`, which
    /// holds the call's run (crossing §3, §5).
    pub fn call_await<'a>(
        &'a self,
        cx: &'a Executor<'_, R>,
        args: <<G as FamilyAt<'a, R, Ts>>::Params as Keys<'a, R>>::Give,
    ) -> impl Future<Output = <<G as FamilyAt<'a, R, Ts>>::Result as Key<'a, R>>::Open> + Send + 'a
    where
        G: FamilyAt<'a, R, Ts>,
        <<G as FamilyAt<'a, R, Ts>>::Result as Key<'a, R>>::Open: Send,
    {
        let (runtime, at) = (cx.runtime(), Settled::of_the_entry());
        let result = runtime.call_later(self.r#impl, G::Params::give_run(args, runtime, at), at);
        async move { <<G::Result as Key<'a, R>>::Open as Open<'a, R>>::open(runtime, result.await, at) }
    }
}

/// The index an impl's value is held at: the generic function `G` at `Ts`.
pub struct Chosen<G, Ts> {
    _never: PhantomData<fn() -> (G, Ts)>,
}

/// The index an instance's receiver is held at: the receiver of `G` at `Ts`.
pub struct Receiver<G, Ts> {
    _never: PhantomData<fn() -> (G, Ts)>,
}

impl<G, Ts> Abstract for Chosen<G, Ts> {}
impl<G, Ts> Abstract for Receiver<G, Ts> {}

impl<R: Runtime, G, Ts> Index<R> for Chosen<G, Ts> {
    type View = Uniform<R, Chosen<G, Ts>>;
    type Key = Uniform<R, Chosen<G, Ts>>;
}

impl<R: Runtime, G, Ts> Index<R> for Receiver<G, Ts> {
    type View = Uniform<R, Receiver<G, Ts>>;
    type Key = Uniform<R, Receiver<G, Ts>>;
}

/// A receiver and the impl of the generic function `G` at `Ts` the site
/// chose for it, both owned as boxes in one value: only the entry makes it,
/// it calls that receiver only, and it may be moved and kept, in a value that
/// holds it, for later calls.
pub struct Instance<R: Runtime, G, Ts> {
    receiver: Uniform<R, Receiver<G, Ts>>,
    r#impl: Uniform<R, Chosen<G, Ts>>,
}

impl<R: Runtime, G, Ts> Instance<R, G, Ts> {
    pub(crate) fn of_the_entry(receiver: Uniform<R, Receiver<G, Ts>>, r#impl: Uniform<R, Chosen<G, Ts>>) -> Self {
        Instance { receiver, r#impl }
    }

    /// The run of a call: the receiver as the generic function takes it, then
    /// the rest.
    fn run<'l>(receiver: R::Value, rest: impl Iterator<Item = R::Value>) -> impl Iterator<Item = R::Value> {
        std::iter::once(receiver).chain(rest)
    }
}

impl<R: CallsBack, G, Ts> Instance<R, G, Ts>
where
    G: GenericFunction,
    G::Takes: receiver::Lent,
{
    /// A returning call, the receiver lent as the generic function takes it,
    /// `&` or `&mut`, with the rest for `'l`.
    pub fn call<'l>(
        &'l mut self,
        cx: &'l Executor<'_, R>,
        rest: <<G as FamilyAt<'l, R, Ts>>::Rest as Keys<'l, R>>::Give,
    ) -> <<G as FamilyAt<'l, R, Ts>>::Result as Key<'l, R>>::Open
    where
        G: FamilyAt<'l, R, Ts>,
    {
        let (runtime, at) = (cx.runtime(), Settled::of_the_entry());
        let receiver = <G::Takes as receiver::Lent>::lend(runtime, self.receiver.place_mut(), at);
        let run = Self::run(receiver, G::Rest::give_run(rest, runtime, at));
        let result = runtime.call_now(self.r#impl.value(), run, at);
        <<G::Result as Key<'l, R>>::Open as Open<'l, R>>::open(runtime, result, at)
    }

    /// An awaited call, the receiver lent as the generic function takes it,
    /// with the rest for the call future `'a`, which holds the call's run
    /// (crossing §3, §5).
    pub fn call_await<'a>(
        &'a mut self,
        cx: &'a Executor<'_, R>,
        rest: <<G as FamilyAt<'a, R, Ts>>::Rest as Keys<'a, R>>::Give,
    ) -> impl Future<Output = <<G as FamilyAt<'a, R, Ts>>::Result as Key<'a, R>>::Open> + Send + 'a
    where
        G: FamilyAt<'a, R, Ts>,
        <<G as FamilyAt<'a, R, Ts>>::Result as Key<'a, R>>::Open: Send,
    {
        let (runtime, at) = (cx.runtime(), Settled::of_the_entry());
        let receiver = <G::Takes as receiver::Lent>::lend(runtime, self.receiver.place_mut(), at);
        let run = Self::run(receiver, G::Rest::give_run(rest, runtime, at));
        let result = runtime.call_later(self.r#impl.value(), run, at);
        async move { <<G::Result as Key<'a, R>>::Open as Open<'a, R>>::open(runtime, result.await, at) }
    }
}

impl<R: CallsBack, G, Ts> Instance<R, G, Ts>
where
    G: GenericFunction<Takes = receiver::Value>,
{
    /// A returning call, the receiver given and the rest lent for `'l`; the
    /// impl is released after it.
    pub fn call_once<'l>(
        self,
        cx: &'l Executor<'_, R>,
        rest: <<G as FamilyAt<'l, R, Ts>>::Rest as Keys<'l, R>>::Give,
    ) -> <<G as FamilyAt<'l, R, Ts>>::Result as Key<'l, R>>::Open
    where
        G: FamilyAt<'l, R, Ts>,
    {
        let (runtime, at) = (cx.runtime(), Settled::of_the_entry());
        let Instance { receiver, r#impl } = self;
        let run = Self::run(receiver.into_value(), G::Rest::give_run(rest, runtime, at));
        let result = runtime.call_now(r#impl.value(), run, at);
        <<G::Result as Key<'l, R>>::Open as Open<'l, R>>::open(runtime, result, at)
    }

    /// An awaited call, the receiver given and the rest lent for the call
    /// future `'a`; the impl is released when the call ends.
    pub fn call_once_await<'a>(
        self,
        cx: &'a Executor<'_, R>,
        rest: <<G as FamilyAt<'a, R, Ts>>::Rest as Keys<'a, R>>::Give,
    ) -> impl Future<Output = <<G as FamilyAt<'a, R, Ts>>::Result as Key<'a, R>>::Open> + Send + 'a
    where
        G: FamilyAt<'a, R, Ts> + 'a,
        Ts: 'a,
        <<G as FamilyAt<'a, R, Ts>>::Result as Key<'a, R>>::Open: Send,
    {
        let (runtime, at) = (cx.runtime(), Settled::of_the_entry());
        let Instance { receiver, r#impl } = self;
        let run = Self::run(receiver.into_value(), G::Rest::give_run(rest, runtime, at));
        let result = runtime.call_later(r#impl.value(), run, at);
        async move {
            let result = result.await;
            drop(r#impl);
            <<G::Result as Key<'a, R>>::Open as Open<'a, R>>::open(runtime, result, at)
        }
    }
}

// SAFETY: an instance is two boxes, each one runtime value whatever its
// index, so it and its canonical form, the instance at `Ts`'s canonical form,
// have one size, alignment, field offsets and release. It reaches `Ts` only
// by its boxes' indices, which are no part of it, and nothing in it is
// interior-mutable. What it holds for the checker's flows is its receiver, a
// value of `Ts`'s variables, so `FACTS` states `Ts`'s parameters.
unsafe impl<R, G, Ts> Structural for Instance<R, G, Ts>
where
    R: Runtime,
    G: 'static,
    Ts: Structural,
{
    type Canon = Instance<R, G, Ts::Canon>;

    const FACTS: StructuralFacts = StructuralFacts { layout: LayoutFact::CanonicalOf, generics: Ts::FACTS.generics };
}

/// Two types are one.
#[doc(hidden)]
pub trait Same<T: ?Sized> {}

impl<T: ?Sized> Same<T> for T {}

/// Compiles only where `Params` and `Result` are the generic function `G`'s
/// parameters' and result's keys at `Ts`: an impl's signature, checked
/// where `#[extern_fn(impl_of = ..)]` writes it (impls §2).
#[doc(hidden)]
pub fn is_the_signature<'p, R: Runtime, G: FamilyAt<'p, R, Ts>, Ts, Params, Result>()
where
    Params: Same<G::Params>,
    Result: Same<G::Result>,
{
}
