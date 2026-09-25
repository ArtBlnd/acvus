//! A required instance, in the two forms its signature's receiver decides
//! (RFC-0067 rule 1): an `Instance` owns the receiver its call steps or
//! consumes, and an `InstanceOf` stands at a type and is handed `&I`.
//!
//! A handler declares a requirement by taking either as a parameter;
//! `#[extern_fn]` reads it and records it on `FnDecl::requires`, which is
//! what `Externs::combine` meets with the signature's instances. An
//! ordinary handler's word lands in its call site's table, an instance's in
//! its own entry (RFC-0070 rule 2).

use std::future::Future;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::Deref;

use acvus_mir::ty::{PolyTy, Task};
use acvus_utils::Interner;

use crate::crossing::Crossing;
use crate::ctx::Ctx;
use crate::handler::{ByRef, ByValue, Lends};
use crate::loan::{Ending, Lending, Loan, Mut, Shared, Through, Unnamed};
use crate::obj::OneValue;
use crate::owned::Owned;
use crate::reference::Ref;
use crate::runtime::Runtime;
use crate::ty_arg::PolyVars;

/// The address of a mono glue and the task it runs at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InstanceRun {
    at: usize,
    task: Task,
}

impl InstanceRun {
    /// # Safety
    /// `at` is the address of a `fn` of the signature's `Now` type when
    /// `task` is `Sync`, and of its `Later` type otherwise.
    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn from_glue(at: usize, task: Task) -> Self {
        InstanceRun { at, task }
    }

    #[inline(always)]
    pub fn at(self) -> usize {
        self.at
    }

    #[inline(always)]
    pub fn task(self) -> Task {
        self.task
    }
}

/// What a value made by `Runtime::instance_value` addresses (RFC-0070 rule 2).
pub struct InstanceEntry<Rt>
where
    Rt: Runtime,
{
    pub run: InstanceRun,
    /// One instance word per `Required` parameter of the declaration this
    /// instance came from, in declaration order, each made by
    /// `Runtime::instance_value` from an entry of its own.
    pub requires: Box<[Rt::Value]>,
}

pub struct Now;
pub struct Later;

/// The task a requirement written at this marker calls its instance at.
pub trait CalledAt {
    const TASK: Task;
}

impl CalledAt for Now {
    const TASK: Task = Task::Sync;
}

impl CalledAt for Later {
    const TASK: Task = Task::Async;
}

/// `'r` is the run: the entry an instance names is the prepared program's,
/// not a call's storage (RFC-0079 rule 6). The glue hands a handler its
/// required instances at the lifetime of its `Ctx`, which the runtime
/// builds over what it keeps for the run.
pub struct Instance<'r, S, R, Rt, T = Now>
where
    S: Signature<Rt>,
    Rt: Runtime,
{
    recv: R,
    value: Rt::Value,
    at: PhantomData<(&'r (), fn() -> (S, T))>,
}

// SAFETY: an `Instance` is at its own `'s`, holds one word the run keeps,
// and holds its receiver, whose carriers are at `'s` by the bound.
unsafe impl<'s, S, R, Rt, T> crate::Within<'s> for Instance<'s, S, R, Rt, T>
where
    S: Signature<Rt>,
    R: crate::Within<'s>,
    Rt: Runtime,
{
}

// SAFETY: an `Instance` is its receiver, held as a field, and one
// `Rt::Value` at every `S` and `T`.
unsafe impl<'r, M, S, R, Rt, T> crate::UniformPayload<M> for Instance<'r, S, R, Rt, T>
where
    S: Signature<Rt>,
    R: crate::UniformPayload<M>,
    Rt: Runtime,
{
}

impl<'r, S, R, Rt, T> Instance<'r, S, R, Rt, T>
where
    S: Signature<Rt>,
    Rt: Runtime,
{
    /// # Safety
    /// `value` was made by `Runtime::instance_value` from an entry of an
    /// instance of `S` standing at the type of `recv`, as the checker chose
    /// it at the site `recv` was passed to, and the entry is live for `'r`.
    #[inline(always)]
    pub(crate) unsafe fn own(recv: R, value: Rt::Value) -> Self
    where
        S::Mode: StepsItsReceiver,
        R: Holds<Rt, S::This>,
    {
        Instance {
            recv,
            value,
            at: PhantomData,
        }
    }

    #[inline(always)]
    pub fn into_inner(self) -> R {
        self.recv
    }
}

impl<'w, S, R, Rt> Instance<'w, S, R, Rt, Now>
where
    S: Signature<Rt>,
    Rt: Runtime,
{
    #[inline(always)]
    pub fn into_async(self) -> Instance<'w, S, R, Rt, Later> {
        Instance {
            recv: self.recv,
            value: self.value,
            at: PhantomData,
        }
    }

    /// The result at the requirer's own types: a value as itself, a
    /// borrow at the lifetime of the receiver it was lent from.
    #[inline(always)]
    pub fn call<'r>(&'r mut self, ctx: &mut Ctx<'_, Rt>, rest: S::Rest<'r>) -> S::Ret<'r>
    where
        S: CrossesRest<Rt> + Signature<Rt, Mode = Mut> + 'r,
        R: Holds<Rt, S::This>,
    {
        ctx.name_receiver(self.recv.word());
        // SAFETY: the rest crosses at the signature's own types, which the
        // checker unified with the instance's (RFC-0068 rule 6).
        let words = S::cross_rest(unsafe { Crossing::new(ctx.rt) }, rest);
        // SAFETY: `own`'s contract: the word addresses an entry of an
        // instance of `S` at the type of the receiver named above, which is
        // this instance's own.
        unsafe { S::call_now(self.value, ctx, words) }
    }
}

impl<'w, S, R, Rt> Instance<'w, S, R, Rt, Later>
where
    S: Signature<Rt>,
    Rt: Runtime,
{
    /// The call of a `sync =` twin: the twin is the body a `Sync` site
    /// runs, and at such a site the checker settled on an instance that
    /// returns.
    #[inline(always)]
    pub fn call<'r>(&'r mut self, ctx: &mut Ctx<'_, Rt>, rest: S::Rest<'r>) -> S::Ret<'r>
    where
        S: CrossesRest<Rt> + Signature<Rt, Mode = Mut> + 'r,
        R: Holds<Rt, S::This>,
    {
        debug_assert_eq!(
            // SAFETY: `own`'s contract: the word addresses an instance's entry.
            unsafe { Rt::instance_entry(&self.value) }.run.task(),
            Task::Sync,
            "a sync twin was handed an instance that suspends"
        );
        ctx.name_receiver(self.recv.word());
        // SAFETY: as `Instance::<_, _, _, Now>::call`'s.
        let words = S::cross_rest(unsafe { Crossing::new(ctx.rt) }, rest);
        // SAFETY: as `Instance::<_, _, _, Now>::call`'s, and the task above.
        unsafe { S::call_now(self.value, ctx, words) }
    }

    #[inline(always)]
    pub fn call_await<'a>(
        &'a mut self,
        ctx: &'a mut Ctx<'_, Rt>,
        rest: S::Rest<'a>,
    ) -> impl Future<Output = S::Ret<'a>> + Send + 'a
    where
        S: CrossesRest<Rt> + Signature<Rt, Mode = Mut> + 'a,
        R: Holds<Rt, S::This>,
        S::Ret<'a>: Send,
    {
        ctx.name_receiver(self.recv.word());
        // SAFETY: as `call`'s.
        let words = S::cross_rest(unsafe { Crossing::new(ctx.rt) }, rest);
        // SAFETY: as `call`'s.
        unsafe { S::call_later(self.value, ctx, words) }
    }
}

/// The call of an `Instance` whose signature takes its receiver by value
/// (RFC-0070 rule 4): the receiver goes to the instance's body, and the
/// `Instance` with it. It is a trait and not an inherent method because
/// `Instance::call` already names the call that steps its receiver by
/// `&mut self`, and one inherent name cannot take `self` both ways.
pub trait Consume<Rt>: Sized
where
    Rt: Runtime,
{
    type Signature: CrossesRest<Rt>;

    fn call<'r>(
        self,
        ctx: &mut Ctx<'_, Rt>,
        rest: <Self::Signature as CrossesRest<Rt>>::Rest<'r>,
    ) -> <Self::Signature as Signature<Rt>>::Ret<'r>
    where
        Self::Signature: 'r;
}

impl<'w, S, I, Rt> Consume<Rt> for Instance<'w, S, I, Rt, Now>
where
    S: CrossesRest<Rt> + Signature<Rt, This = I, Mode = Moved>,
    I: Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    type Signature = S;

    #[inline(always)]
    fn call<'r>(self, ctx: &mut Ctx<'_, Rt>, rest: S::Rest<'r>) -> S::Ret<'r>
    where
        S: 'r,
    {
        // The mono glue's `receiver_by_value` takes the value this receiver
        // holds, so it is not dropped here.
        let recv = std::mem::ManuallyDrop::new(self.recv);
        ctx.name_receiver(&**recv);
        // SAFETY: as `Instance::<_, _, _, Now>::call`'s.
        let words = S::cross_rest(unsafe { Crossing::new(ctx.rt) }, rest);
        // SAFETY: as `Instance::<_, _, _, Now>::call`'s; the receiver named
        // above is live across the call, which reads it once.
        unsafe { S::call_now(self.value, ctx, words) }
    }
}

impl<'w, S, I, Rt> Consume<Rt> for Instance<'w, S, I, Rt, Later>
where
    S: CrossesRest<Rt> + Signature<Rt, This = I, Mode = Moved>,
    I: Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    type Signature = S;

    /// The call of a `sync =` twin, as `Instance::<_, _, _, Later>::call`'s.
    #[inline(always)]
    fn call<'r>(self, ctx: &mut Ctx<'_, Rt>, rest: S::Rest<'r>) -> S::Ret<'r>
    where
        S: 'r,
    {
        debug_assert_eq!(
            // SAFETY: `own`'s contract: the word addresses an instance's entry.
            unsafe { Rt::instance_entry(&self.value) }.run.task(),
            Task::Sync,
            "a sync twin was handed an instance that suspends"
        );
        // As the `Now` impl's.
        let recv = std::mem::ManuallyDrop::new(self.recv);
        ctx.name_receiver(&**recv);
        // SAFETY: as the `Now` impl's.
        let words = S::cross_rest(unsafe { Crossing::new(ctx.rt) }, rest);
        // SAFETY: as the `Now` impl's, and the task above.
        unsafe { S::call_now(self.value, ctx, words) }
    }
}

/// What an `Instance` can hold as its receiver at a signature standing at
/// `This`: the value itself, or the `&mut` borrow of it its requirer was
/// lent. Either names the runtime's value the call's receiver is.
pub trait Holds<Rt, This>: sealed::Holds<Rt, This>
where
    Rt: Runtime,
{
    fn word(&self) -> &Rt::Value;
}

mod sealed {
    pub trait Holds<Rt, This> {}
}

impl<I, Rt> sealed::Holds<Rt, I> for I
where
    I: Deref<Target = Rt::Value>,
    Rt: Runtime,
{
}

impl<I, Rt> sealed::Holds<Rt, I> for &mut I
where
    I: Deref<Target = Rt::Value>,
    Rt: Runtime,
{
}

impl<I, Rt> Holds<Rt, I> for I
where
    I: Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    #[inline(always)]
    fn word(&self) -> &Rt::Value {
        self
    }
}

impl<I, Rt> Holds<Rt, I> for &mut I
where
    I: Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    #[inline(always)]
    fn word(&self) -> &Rt::Value {
        self
    }
}

/// A requirement of a signature whose receiver is `&I` (RFC-0067 rule 1):
/// one of the runtime's values, standing at the type, and handed each
/// receiver it reads, since a requirer applies it to many values of the
/// type (a map's keys, both sides of `==`).
pub struct InstanceOf<'r, S, I, Rt, T = Now>
where
    S: Signature<Rt, This = I>,
    Rt: Runtime,
{
    value: Rt::Value,
    at: PhantomData<(&'r (), fn() -> (S, I, T))>,
}

// SAFETY: an `InstanceOf` is at its own `'s`, and holds one word the run
// keeps.
unsafe impl<'s, S, I, Rt, T> crate::Within<'s> for InstanceOf<'s, S, I, Rt, T>
where
    S: Signature<Rt, This = I>,
    Rt: Runtime,
{
}

// SAFETY: an `InstanceOf` is one `Rt::Value` at every `S`, `I` and `T`.
unsafe impl<'r, M, S, I, Rt, T> crate::UniformPayload<M> for InstanceOf<'r, S, I, Rt, T>
where
    S: Signature<Rt, This = I>,
    Rt: Runtime,
{
}

impl<'r, S, I, Rt, T> Clone for InstanceOf<'r, S, I, Rt, T>
where
    S: Signature<Rt, This = I>,
    Rt: Runtime,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'r, S, I, Rt, T> Copy for InstanceOf<'r, S, I, Rt, T>
where
    S: Signature<Rt, This = I>,
    Rt: Runtime,
{
}

impl<'r, S, I, Rt, T> InstanceOf<'r, S, I, Rt, T>
where
    S: Signature<Rt, This = I>,
    Rt: Runtime,
{
    const ONE_VALUE: () = assert!(
        size_of::<InstanceOf<'r, S, I, Rt, T>>() == size_of::<Rt::Value>(),
        "an instance of a type is one of the runtime's values and nothing else"
    );

    /// # Safety
    /// `value` was made by `Runtime::instance_value` from an entry of an
    /// instance of `S` standing at the type `I` is filled with, and the
    /// entry is live for `'r`.
    #[inline(always)]
    pub(crate) unsafe fn at(value: Rt::Value) -> Self
    where
        S::Mode: ReadsItsReceiver,
    {
        let () = Self::ONE_VALUE;
        InstanceOf {
            value,
            at: PhantomData,
        }
    }
}

impl<'w, S, I, Rt> InstanceOf<'w, S, I, Rt, Now>
where
    S: Signature<Rt, This = I>,
    Rt: Runtime,
{
    #[inline(always)]
    pub fn into_async(self) -> InstanceOf<'w, S, I, Rt, Later> {
        InstanceOf {
            value: self.value,
            at: PhantomData,
        }
    }

    /// The result at the requirer's own types: a value as itself, a
    /// borrow at the lifetime of the receiver it was lent from.
    #[inline(always)]
    pub fn call<'r>(self, ctx: &mut Ctx<'_, Rt>, recv: &'r I, rest: S::Rest<'r>) -> S::Ret<'r>
    where
        S: CrossesRest<Rt> + Signature<Rt, Mode = Shared> + 'r,
        I: Deref<Target = Rt::Value>,
    {
        ctx.name_receiver(recv);
        // SAFETY: the rest crosses at the signature's own types, which the
        // checker unified with the instance's (RFC-0068 rule 6).
        let words = S::cross_rest(unsafe { Crossing::new(ctx.rt) }, rest);
        // SAFETY: `at`'s contract: the word addresses an entry of an
        // instance of `S` at the type `I` is filled with, and `recv` is at
        // that type by the impl's bound `S: Signature<Rt, This = I>`.
        unsafe { S::call_now(self.value, ctx, words) }
    }
}

impl<'w, S, I, Rt> InstanceOf<'w, S, I, Rt, Later>
where
    S: Signature<Rt, This = I>,
    Rt: Runtime,
{
    /// The call of a `sync =` twin, as `Instance::<_, _, _, Later>::call`'s.
    #[inline(always)]
    pub fn call<'r>(self, ctx: &mut Ctx<'_, Rt>, recv: &'r I, rest: S::Rest<'r>) -> S::Ret<'r>
    where
        S: CrossesRest<Rt> + Signature<Rt, Mode = Shared> + 'r,
        I: Deref<Target = Rt::Value>,
    {
        debug_assert_eq!(
            // SAFETY: `at`'s contract: the word addresses an instance's entry.
            unsafe { Rt::instance_entry(&self.value) }.run.task(),
            Task::Sync,
            "a sync twin was handed an instance that suspends"
        );
        ctx.name_receiver(recv);
        // SAFETY: as `InstanceOf::<_, _, _, Now>::call`'s.
        let words = S::cross_rest(unsafe { Crossing::new(ctx.rt) }, rest);
        // SAFETY: as `InstanceOf::<_, _, _, Now>::call`'s, and the task above.
        unsafe { S::call_now(self.value, ctx, words) }
    }

    #[inline(always)]
    pub fn call_await<'a>(
        self,
        ctx: &'a mut Ctx<'_, Rt>,
        recv: &'a I,
        rest: S::Rest<'a>,
    ) -> impl Future<Output = S::Ret<'a>> + Send + 'a
    where
        S: CrossesRest<Rt> + Signature<Rt, Mode = Shared> + 'a,
        I: Deref<Target = Rt::Value>,
        S::Ret<'a>: Send,
    {
        ctx.name_receiver(recv);
        // SAFETY: as `call`'s.
        let words = S::cross_rest(unsafe { Crossing::new(ctx.rt) }, rest);
        // SAFETY: as `call`'s.
        unsafe { S::call_later(self.value, ctx, words) }
    }
}

/// A signature's first parameter taken by value: its instance's body
/// takes the receiver, as `Shared` and `Mut` name the other two modes.
pub struct Moved;

/// A receiver mode an `Instance` owns: one its call steps (`Mut`) or
/// consumes (`Moved`) (RFC-0067 rule 1).
#[diagnostic::on_unimplemented(
    message = "an `Instance` owns the receiver of a signature whose call steps or consumes it, and this signature reads its receiver through `&` (`{Self}`)",
    label = "this `Instance` is at a signature whose receiver is `&I`",
    note = "a signature whose receiver is `&I` is required by `InstanceOf<S, I, Rt>`, which stands at the type and is handed each receiver it reads (RFC-0067 rule 1)"
)]
pub trait StepsItsReceiver {}

impl StepsItsReceiver for Mut {}
impl StepsItsReceiver for Moved {}

/// A receiver mode an `InstanceOf` is handed: `&I` (RFC-0067 rule 1).
#[diagnostic::on_unimplemented(
    message = "an `InstanceOf` stands at a type and reads the receivers it is handed, and this signature steps or consumes its receiver (`{Self}`)",
    label = "this `InstanceOf` is at a signature whose receiver is `&mut I` or `I`",
    note = "a signature whose receiver is `&mut I` or `I` is required by taking the receiver and its instance as one parameter, `it: Instance<S, I, Rt>`, which owns the receiver it steps (RFC-0067 rule 1)"
)]
pub trait ReadsItsReceiver {}

impl ReadsItsReceiver for Shared {}

/// A shared signature's marker filled with a declaration's own types: the
/// signature's type with each of its variables replaced by what the marker
/// names there. `extern_signature!` writes the impl; `#[extern_fn]` reads
/// it into `Requirement::pattern`.
pub trait RequirementOf {
    fn pattern(interner: &Interner, vars: &PolyVars) -> PolyTy;
}

/// A shared signature as a Rust caller of one of its instances sees it:
/// the shape of a call and nothing about a receiver beyond how the first
/// parameter takes it. `extern_signature!` writes the impl, so that a
/// handler which requires a signature restates none of its modes and none
/// of its widths.
pub trait Signature<Rt>: Send + Sync
where
    Rt: Runtime,
{
    /// What the signature's first parameter stands at: the variable a bound
    /// names, which RFC-0067 rule 2 makes the one an instance is matched by.
    type This;
    /// How the first parameter takes its receiver: `Shared` for `&This`,
    /// `Mut` for `&mut This`, `Moved` for `This`. The mode reaches a
    /// requiring handler through this projection alone, so the call it
    /// writes is where a wrong mode is refused (RFC-0070 rule 4).
    type Mode;
    /// The arguments after the first as the instance's glue takes them: a
    /// position at one of the signature's own type variables as the
    /// runtime's value, since one `fn` type serves every instance
    /// (RFC-0068 rule 6).
    type Words<'a>
    where
        Self: 'a;
    /// The result as the requirer receives it: a value as itself; a result
    /// standing at a `Ref<T, M, Rt>` marker as `&'r T` / `&'r mut T`, for
    /// the `'r` of the receiver the call lent (RFC-0068 rule 6).
    type Ret<'r>
    where
        Self: 'r;

    /// Obligation across artifacts: `extern_signature!` writes this `fn`
    /// type from the signature's own parameter list, taking the entry
    /// first, and `#[extern_fn]` assigns every mono glue to it before
    /// taking the address `call_now` runs.
    type Now;
    /// As `Now`, for a glue whose body suspends.
    type Later;

    /// # Safety
    /// `value` addresses the entry of an instance of this signature, the
    /// receiver named in `ctx` holds a value of the type that instance
    /// stands at, and that instance's body returns rather than suspends.
    unsafe fn call_now<'r>(
        value: Rt::Value,
        ctx: &mut Ctx<'_, Rt>,
        rest: Self::Words<'r>,
    ) -> Self::Ret<'r>
    where
        Self: 'r;

    /// # Safety
    /// As `call_now`'s, without its last clause.
    unsafe fn call_later<'a>(
        value: Rt::Value,
        ctx: &'a mut Ctx<'_, Rt>,
        rest: Self::Words<'a>,
    ) -> impl Future<Output = Self::Ret<'a>> + Send + 'a
    where
        Self: 'a,
        Self::Ret<'a>: Send;
}

/// The crossing of a signature's rest from the requirer's types to the
/// runtime's values: the glue's, written by `extern_signature!` beside the
/// `Signature` impl. It is its own trait because the crossing of a position
/// at a variable asks a bound of that variable, which a signature a
/// requirer only names, and never calls at such a position, must not ask.
///
/// # Safety
/// The words `cross_rest` hands back are exactly the rest the requirer
/// passed, at the types the checker settled for the signature's positions;
/// it crosses nothing else with the capability, a position only through its
/// own crossing; and it keeps no capability past the call.
pub unsafe trait CrossesRest<Rt>: Signature<Rt>
where
    Rt: Runtime,
{
    /// The arguments after the first as the requirer passes them: a
    /// position at one of the signature's own type variables at that
    /// variable, borrowed as the signature takes it.
    type Rest<'a>
    where
        Self: 'a;

    fn cross_rest<'a>(rt: Crossing<'_, Rt>, rest: Self::Rest<'a>) -> Self::Words<'a>
    where
        Self: 'a;
}

/// `Self`, named through a runtime: a signature's rest run is a type alias,
/// and a type alias names every parameter it takes (E0091), so a run of
/// concrete positions names its runtime here.
pub trait RestRun<Rt>
where
    Rt: Runtime,
{
    type Run;
}

impl<T, Rt> RestRun<Rt> for T
where
    Rt: Runtime,
{
    type Run = Self;
}

/// A position a signature takes by `&`, as the instance's own handler
/// spells it: `Self` is that handler's parameter type, and `M` the parameter
/// marker `#[extern_fn]` already writes for it, so the spelling — a Rust
/// reference read at entry, or the `Ref` carrier a body that keeps the
/// reference takes (RFC-0018) — and the representation are read off one
/// parameter.
///
/// # Safety
/// What `restore_shared` hands back reads exactly the caller's value at the
/// type the checker settled for the position; it crosses nothing else with
/// the capability; and it keeps no capability past the call.
pub unsafe trait RestoreShared<'b, M, Rt>: Sized
where
    Rt: Runtime,
{
    /// # Safety
    /// `crossed` is the caller's own value, live for `'b`, and holds what
    /// the instance stands at; `at` is empty.
    unsafe fn restore_shared<'r>(
        rt: crate::Crossing<'r, Rt>,
        at: &'b mut Option<Lending<'r, Shared, Rt>>,
        crossed: &'b Rt::Value,
    ) -> Self;
}

/// As `RestoreShared`, for a position a signature takes by `&mut`.
///
/// # Safety
/// As `RestoreShared`'s, for `restore_exclusive`.
pub unsafe trait RestoreExclusive<'b, M, Rt>: Sized
where
    Rt: Runtime,
{
    /// What `restore_exclusive` lends in place through `at`.
    type Ending: Ending<Rt>;

    /// A borrow lent in place is read through `at`, whose drop after the
    /// instance's body ends the loan (`Lending`).
    ///
    /// # Safety
    /// As `RestoreShared::restore_shared`'s, and no other name of the
    /// storage is live.
    unsafe fn restore_exclusive<'r>(
        rt: crate::Crossing<'r, Rt>,
        at: &'b mut Option<Lending<'r, Mut, Rt, Self::Ending>>,
        crossed: &'b mut Rt::Value,
    ) -> Self;
}

/// As `RestoreShared`, for a position a signature takes by value.
///
/// # Safety
/// As `RestoreShared`'s, for `restore_by_value`.
pub unsafe trait RestoreByValue<'b, M, Rt>: Sized
where
    Rt: Runtime,
{
    /// # Safety
    /// `crossed` was erased from what the instance stands at, and what it
    /// names is live for `'b`.
    unsafe fn restore_by_value(rt: crate::Crossing<'_, Rt>, crossed: Owned<Rt>) -> Self;
}

// SAFETY: the reference names the caller's own value, borrowed at `D` through
// `C`'s `Lends`, the type the checker settled for the position; the capability
// is not kept.
unsafe impl<'b, T, C, D, Rt> RestoreShared<'b, ByRef<T, Shared, C>, Rt> for &'b D
where
    C: Lends<D, Rt>,
    D: crate::Within<'b>,
    Rt: Runtime,
{
    #[inline(always)]
    unsafe fn restore_shared<'r>(
        rt: crate::Crossing<'r, Rt>,
        at: &'b mut Option<Lending<'r, Shared, Rt>>,
        crossed: &'b Rt::Value,
    ) -> &'b D {
        // SAFETY: the caller's contract: `crossed` is live for `'b`.
        let lending = at.insert(unsafe { Lending::of(rt.rt(), rt.reference(crossed)) });
        // SAFETY: the reference names the storage the caller lent, and what
        // that storage holds is live for `'b`.
        unsafe { C::deref(rt.rt(), lending.reference()) }
    }
}

// SAFETY: the `Ref` is `materialize` of a reference to the caller's own value;
// nothing else crosses, and the capability is not kept.
unsafe impl<'b, T, M, C, D, Rt> RestoreShared<'b, ByValue<Ref<'static, T, M, Rt>, C>, Rt> for D
where
    T: Send + Sync,
    M: Loan,
    D: OneValue<Rt> + crate::Within<'b>,
    Rt: Runtime,
{
    #[inline(always)]
    unsafe fn restore_shared<'r>(
        rt: crate::Crossing<'r, Rt>,
        _: &'b mut Option<Lending<'r, Shared, Rt>>,
        crossed: &'b Rt::Value,
    ) -> D {
        // SAFETY: the caller's contract: `crossed` is the caller's own
        // value at the type the checker gave this position, the storage
        // outlives the call, and RFC-0018 keeps the reference within it.
        unsafe { OneValue::materialize(rt, rt.reference(crossed)) }
    }
}

// SAFETY: as the shared impl's, exclusively.
unsafe impl<'b, T, C, D, Rt> RestoreExclusive<'b, ByRef<T, Mut, C>, Rt> for &'b mut D
where
    C: Lends<D, Rt>,
    D: crate::Within<'b>,
    Rt: Runtime,
{
    type Ending = Through<C, D>;

    #[inline(always)]
    unsafe fn restore_exclusive<'r>(
        rt: crate::Crossing<'r, Rt>,
        at: &'b mut Option<Lending<'r, Mut, Rt, Through<C, D>>>,
        crossed: &'b mut Rt::Value,
    ) -> &'b mut D {
        // SAFETY: as `RestoreShared`'s, exclusively: `crossed` is the only
        // live name of the storage for `'b`.
        let lending = at.insert(unsafe { Lending::of(rt.rt(), rt.reference(crossed)) });
        // SAFETY: as `RestoreShared`'s, exclusively; the borrow is read
        // through `lending`, so the loan ends when `at`'s holder drops it.
        unsafe { C::deref_mut(rt.rt(), lending.reference()) }
    }
}

// SAFETY: as the shared impl's, exclusively.
unsafe impl<'b, T, M, C, D, Rt> RestoreExclusive<'b, ByValue<Ref<'static, T, M, Rt>, C>, Rt> for D
where
    T: Send + Sync,
    M: Loan,
    D: OneValue<Rt> + crate::Within<'b>,
    Rt: Runtime,
{
    /// The `Ref` is materialized, and nothing is lent through the slot.
    type Ending = Unnamed;

    #[inline(always)]
    unsafe fn restore_exclusive<'r>(
        rt: crate::Crossing<'r, Rt>,
        _: &'b mut Option<Lending<'r, Mut, Rt, Unnamed>>,
        crossed: &'b mut Rt::Value,
    ) -> D {
        // SAFETY: as the shared impl's, exclusively.
        unsafe { OneValue::materialize(rt, rt.reference(crossed)) }
    }
}

// SAFETY: the value is `D`'s own `materialize` of the word the caller crossed;
// nothing else crosses, and the capability is not kept.
unsafe impl<'b, T, C, D, Rt> RestoreByValue<'b, ByValue<T, C>, Rt> for D
where
    T: OneValue<Rt, C>,
    D: OneValue<Rt, C> + crate::Within<'b>,
    Rt: Runtime,
{
    #[inline(always)]
    unsafe fn restore_by_value(rt: crate::Crossing<'_, Rt>, crossed: Owned<Rt>) -> D {
        // SAFETY: the caller's contract, at the one value an `Owned` holds.
        unsafe { <D as OneValue<Rt, C>>::materialize(rt, crossed.into_value(rt.holding())) }
    }
}

/// The receiver of a mono glue taken by value, at the handler's own type.
///
/// # Safety
/// `value` is the receiver the call named, at the type the checker settled
/// for it, which is `D` with every lifetime at `'static`; what it names is
/// live for `'s`.
#[doc(hidden)]
#[inline(always)]
pub unsafe fn receiver_by_value<'s, D, C, Rt>(
    rt: crate::Crossing<'_, Rt>,
    value: Rt::Value,
    _: &'s (),
) -> D
where
    D: OneValue<Rt, C> + crate::Within<'s>,
    Rt: Runtime,
{
    // SAFETY: the caller's contract.
    unsafe { <D as OneValue<Rt, C>>::materialize(rt, value) }
}

/// The receiver of a mono glue taken by `&` or `&mut`, at the handler's own
/// type, read through the `Lending` whose drop ends its loan.
///
/// # Safety
/// `lending` names the receiver's storage, which holds what `D` names with
/// every lifetime at `'static` and is live for `'s`, exclusively so for a
/// `Mut` loan.
#[doc(hidden)]
#[inline(always)]
pub unsafe fn receiver_borrowed<'s, D, M, C, Rt>(
    rt: &Rt,
    lending: &'s Lending<'_, M, Rt, Through<C, D>>,
) -> M::Of<'s, D>
where
    M: Loan,
    C: Lends<D, Rt>,
    D: Send + Sync + crate::Within<'s>,
    Rt: Runtime,
{
    // SAFETY: the caller's contract.
    unsafe { M::borrow::<D, C, Rt>(rt, lending.reference()) }
}
