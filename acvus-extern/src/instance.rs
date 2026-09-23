//! An instance as one value of its own, beside the value it serves
//! (RFC-0067 rule 1).
//!
//! A handler declares a requirement by taking an `Instance` parameter;
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

use crate::ctx::Ctx;
use crate::handler::{ByRef, ByValue, Lends};
use crate::loan::{Loan, Mut, Shared};
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

pub struct Instance<S, I, Rt, T = Now>
where
    S: Signature<Rt>,
    Rt: Runtime,
{
    value: Rt::Value,
    at: PhantomData<fn() -> (S, I, T)>,
}

// SAFETY: an `Instance` is one `Rt::Value` at every `S`, `I` and `T`.
unsafe impl<M, S, I, Rt, T> crate::UniformPayload<M> for Instance<S, I, Rt, T>
where
    S: Signature<Rt>,
    Rt: Runtime,
{
}

impl<S, I, Rt, T> Clone for Instance<S, I, Rt, T>
where
    S: Signature<Rt>,
    Rt: Runtime,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<S, I, Rt, T> Copy for Instance<S, I, Rt, T>
where
    S: Signature<Rt>,
    Rt: Runtime,
{
}

impl<S, I, Rt, T> Instance<S, I, Rt, T>
where
    S: Signature<Rt>,
    Rt: Runtime,
{
    const ONE_VALUE: () = assert!(
        size_of::<Instance<S, I, Rt, T>>() == size_of::<Rt::Value>(),
        "an instance is one of the runtime's values and nothing else"
    );

    /// # Safety
    /// `value` was made by `Runtime::instance_value` from an entry of an
    /// instance of `S` standing at the type `I` is filled with.
    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn at(value: Rt::Value) -> Self {
        let () = Self::ONE_VALUE;
        Instance {
            value,
            at: PhantomData,
        }
    }
}

impl<S, I, Rt> Instance<S, I, Rt, Now>
where
    S: Signature<Rt>,
    Rt: Runtime,
{
    #[inline(always)]
    pub fn into_async(self) -> Instance<S, I, Rt, Later> {
        Instance {
            value: self.value,
            at: PhantomData,
        }
    }

    /// The result at the requirer's own types: a value as itself, a
    /// borrow at the lifetime of the receiver it was lent from.
    #[inline(always)]
    pub fn call<'r>(
        self,
        ctx: &mut Ctx<'_, Rt>,
        recv: S::Recv<'r>,
        rest: S::Rest<'r>,
    ) -> S::Ret<'r>
    where
        S::Recv<'r>: Receiver<Rt>,
    {
        recv.name_in(ctx);
        // SAFETY: `at`'s contract: the word addresses an entry of an
        // instance of `S` at the type `I` is filled with, and `recv` is at
        // `I` through `S::Recv`.
        unsafe { S::call_now(self.value, ctx, rest) }
    }
}

impl<S, I, Rt> Instance<S, I, Rt, Later>
where
    S: Signature<Rt>,
    Rt: Runtime,
{
    /// The call of a `sync =` twin: the twin is the body a `Sync` site
    /// runs, and at such a site the checker settled on an instance that
    /// returns.
    #[inline(always)]
    pub fn call<'r>(
        self,
        ctx: &mut Ctx<'_, Rt>,
        recv: S::Recv<'r>,
        rest: S::Rest<'r>,
    ) -> S::Ret<'r>
    where
        S::Recv<'r>: Receiver<Rt>,
    {
        debug_assert_eq!(
            // SAFETY: `at`'s contract: the word addresses an instance's entry.
            unsafe { Rt::instance_entry(&self.value) }.run.task(),
            Task::Sync,
            "a sync twin was handed an instance that suspends"
        );
        recv.name_in(ctx);
        // SAFETY: as `Instance::<_, _, _, Now>::call`'s, and the task above.
        unsafe { S::call_now(self.value, ctx, rest) }
    }

    #[inline(always)]
    pub fn call_await<'a>(
        self,
        ctx: &'a mut Ctx<'_, Rt>,
        recv: S::Recv<'a>,
        rest: S::Rest<'a>,
    ) -> impl Future<Output = S::Ret<'a>> + Send + 'a
    where
        S::Recv<'a>: Receiver<Rt>,
        S::Ret<'a>: Send,
    {
        recv.name_in(ctx);
        // SAFETY: as `call`'s.
        unsafe { S::call_later(self.value, ctx, rest) }
    }
}

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
pub trait Signature<Rt>: Send + Sync + 'static
where
    Rt: Runtime,
{
    /// What the signature's first parameter stands at: the variable a bound
    /// names, which RFC-0067 rule 2 makes the one an instance is matched by.
    type This;
    /// The first parameter's mode: `&'a This`, `&'a mut This`, or `This`.
    /// The mode reaches a requiring handler through this projection alone,
    /// so the handler's own `Instance::call` is where a wrong mode is
    /// refused.
    type Recv<'a>;
    /// The arguments after the first, a position at one of the
    /// signature's own type variables as the caller's own value.
    type Rest<'a>;
    /// The result as the requirer receives it: a value as itself; a result
    /// standing at a `Ref<T, M, Rt>` marker as `&'r T` / `&'r mut T`, for
    /// the `'r` of the receiver the call lent (RFC-0068 rule 6).
    type Ret<'r>;

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
        rest: Self::Rest<'r>,
    ) -> Self::Ret<'r>;

    /// # Safety
    /// As `call_now`'s, without its last clause.
    unsafe fn call_later<'a>(
        value: Rt::Value,
        ctx: &'a mut Ctx<'_, Rt>,
        rest: Self::Rest<'a>,
    ) -> impl Future<Output = Self::Ret<'a>> + Send + 'a
    where
        Self::Ret<'a>: Send;
}

#[diagnostic::on_unimplemented(
    message = "`{Self}` is no receiver `Instance::call` can name",
    note = "a required instance is called with the receiver its signature declared, `&T` or `&mut T`, where `T` derefs to the runtime's value; a signature taking its receiver by value cannot be required (RFC-0070 rule 4)."
)]
pub trait Receiver<Rt>
where
    Rt: Runtime,
{
    fn name_in(self, ctx: &mut Ctx<'_, Rt>);
}

impl<I, Rt> Receiver<Rt> for &I
where
    I: Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    #[inline(always)]
    fn name_in(self, ctx: &mut Ctx<'_, Rt>) {
        ctx.name_receiver(&**self);
    }
}

impl<I, Rt> Receiver<Rt> for &mut I
where
    I: Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    #[inline(always)]
    fn name_in(self, ctx: &mut Ctx<'_, Rt>) {
        ctx.name_receiver(&**self);
    }
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
/// spells it. `Self` is the parameter marker `#[extern_fn]` already writes
/// for that parameter, so the spelling — a Rust reference read at entry, or
/// the `Ref` carrier a body that keeps the reference takes (RFC-0018) — and
/// the representation are read off one type.
pub trait RestoreShared<Rt>
where
    Rt: Runtime,
{
    type Out<'b>
    where
        Self: 'b;

    /// # Safety
    /// `crossed` is the caller's own value, live for `'b`, and holds what
    /// the instance stands at; `at` is dead.
    unsafe fn restore_shared<'b>(
        rt: &Rt,
        at: &'b mut Rt::Value,
        crossed: &'b Rt::Value,
    ) -> Self::Out<'b>;
}

/// As `RestoreShared`, for a position a signature takes by `&mut`.
pub trait RestoreExclusive<Rt>
where
    Rt: Runtime,
{
    type Out<'b>
    where
        Self: 'b;

    /// # Safety
    /// As `RestoreShared::restore_shared`'s, and no other name of the
    /// storage is live.
    unsafe fn restore_exclusive<'b>(
        rt: &Rt,
        at: &'b mut Rt::Value,
        crossed: &'b mut Rt::Value,
    ) -> Self::Out<'b>;
}

/// As `RestoreShared`, for a position a signature takes by value.
pub trait RestoreByValue<Rt>
where
    Rt: Runtime,
{
    type Out;

    /// # Safety
    /// `crossed` was erased from what the instance stands at.
    unsafe fn restore_by_value(rt: &Rt, crossed: Owned<Rt>) -> Self::Out;
}

impl<T, C, Rt> RestoreShared<Rt> for ByRef<T, Shared, C>
where
    T: Send + Sync + 'static,
    C: Lends<T, Rt>,
    Rt: Runtime,
{
    type Out<'b>
        = &'b T
    where
        Self: 'b;

    #[inline(always)]
    unsafe fn restore_shared<'b>(rt: &Rt, at: &'b mut Rt::Value, crossed: &'b Rt::Value) -> &'b T {
        // SAFETY: the caller's contract: `crossed` is live for `'b`.
        *at = unsafe { rt.reference(crossed) };
        // SAFETY: the reference names the storage the caller lent.
        unsafe { <Shared as Loan>::borrow::<T, C, Rt>(rt, at) }
    }
}

impl<T, M, C, Rt> RestoreShared<Rt> for ByValue<Ref<T, M, Rt>, C>
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    type Out<'b>
        = Ref<T, M, Rt>
    where
        Self: 'b;

    #[inline(always)]
    unsafe fn restore_shared<'b>(
        rt: &Rt,
        _: &'b mut Rt::Value,
        crossed: &'b Rt::Value,
    ) -> Ref<T, M, Rt> {
        // SAFETY: the caller's contract: `crossed` is the caller's own
        // value at the type the checker gave this position, the storage
        // outlives the call, and RFC-0018 keeps the reference within it.
        unsafe { <Ref<T, M, Rt> as OneValue<Rt>>::materialize(rt, rt.reference(crossed)) }
    }
}

impl<T, C, Rt> RestoreExclusive<Rt> for ByRef<T, Mut, C>
where
    T: Send + Sync + 'static,
    C: Lends<T, Rt>,
    Rt: Runtime,
{
    type Out<'b>
        = &'b mut T
    where
        Self: 'b;

    #[inline(always)]
    unsafe fn restore_exclusive<'b>(
        rt: &Rt,
        at: &'b mut Rt::Value,
        crossed: &'b mut Rt::Value,
    ) -> &'b mut T {
        // SAFETY: as `RestoreShared`'s, exclusively.
        *at = unsafe { rt.reference(crossed) };
        // SAFETY: as `RestoreShared`'s, exclusively.
        unsafe { <Mut as Loan>::borrow::<T, C, Rt>(rt, at) }
    }
}

impl<T, M, C, Rt> RestoreExclusive<Rt> for ByValue<Ref<T, M, Rt>, C>
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    type Out<'b>
        = Ref<T, M, Rt>
    where
        Self: 'b;

    #[inline(always)]
    unsafe fn restore_exclusive<'b>(
        rt: &Rt,
        _: &'b mut Rt::Value,
        crossed: &'b mut Rt::Value,
    ) -> Ref<T, M, Rt> {
        // SAFETY: as the shared impl's, exclusively.
        unsafe { <Ref<T, M, Rt> as OneValue<Rt>>::materialize(rt, rt.reference(crossed)) }
    }
}

impl<T, C, Rt> RestoreByValue<Rt> for ByValue<T, C>
where
    T: OneValue<Rt, C>,
    Rt: Runtime,
{
    type Out = T;

    #[inline(always)]
    unsafe fn restore_by_value(rt: &Rt, crossed: Owned<Rt>) -> T {
        // SAFETY: the caller's contract, at the one value an `Owned` holds.
        unsafe { <T as OneValue<Rt, C>>::materialize(rt, crossed.into_value()) }
    }
}
