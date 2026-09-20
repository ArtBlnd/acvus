//! An instance as one value of its own, beside the value it serves
//! (RFC-0067 Decision 1).
//!
//! A handler declares a requirement by taking an `Instance` parameter;
//! `#[extern_fn]` reads it and records it on `FnDecl::requires`, which is
//! what `Externs::combine` meets with the signature's instances, and the
//! call site's site table is where the resolved word lands.

use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::DerefMut;

use acvus_mir::ty::Task;
use futures::future::BoxFuture;

use crate::ctx::Ctx;
use crate::handler::{ByRef, ByValue};
use crate::loan::{Loan, Mut, Shared};
use crate::obj::OneValue;
use crate::owned::Owned;
use crate::reference::Ref;
use crate::runtime::Runtime;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InstanceRun {
    /// The address of the mono glue `#[extern_fn]` wrote beside the handler.
    pub at: usize,
    pub task: Task,
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
    /// `value` was made by `Runtime::instance_value` from the mono glue of
    /// an instance of `S` standing at the type `I` is filled with.
    #[inline(always)]
    pub unsafe fn at(value: Rt::Value) -> Self {
        let () = Self::ONE_VALUE;
        Instance {
            value,
            at: PhantomData,
        }
    }

    #[inline(always)]
    pub fn into_value(self) -> Rt::Value {
        self.value
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

    /// # Safety
    /// `recv` holds a value of the type this instance stands at, which is
    /// what `Externs::combine` met the requirement with.
    #[inline(always)]
    pub unsafe fn call<'r>(self, ctx: &mut Ctx<'_, Rt>, recv: &mut I, rest: S::Rest<'r>) -> S::Ret
    where
        I: DerefMut<Target = Rt::Value>,
    {
        ctx.name_receiver(&mut *recv);
        // SAFETY: the caller's contract, and the word is this signature's
        // own glue by `Instance::at`'s.
        unsafe { S::call_now(self.value, ctx, rest) }
    }
}

impl<S, I, Rt> Instance<S, I, Rt, Later>
where
    S: Signature<Rt>,
    Rt: Runtime,
{
    /// # Safety
    /// As `Instance::call`'s.
    #[inline(always)]
    pub unsafe fn call_await<'a>(
        self,
        ctx: &'a mut Ctx<'_, Rt>,
        recv: &'a mut I,
        rest: S::Rest<'a>,
    ) -> BoxFuture<'a, S::Ret>
    where
        I: DerefMut<Target = Rt::Value>,
        S::Ret: Send,
    {
        ctx.name_receiver(&mut *recv);
        // SAFETY: as `call`'s.
        unsafe { S::call_later(self.value, ctx, rest) }
    }
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
    /// names, which RFC-0019 makes the one an instance is matched by.
    type This;
    /// The first parameter's mode: `&'a This`, `&'a mut This`, or `This`.
    /// The mode reaches a requiring handler through this projection alone,
    /// so the handler's own `Instance::call` is where a wrong mode is
    /// refused.
    type Recv<'a>;
    /// The arguments after the first, a position at one of the
    /// signature's own type variables as the caller's own value.
    type Rest<'a>;
    type Ret;

    /// Obligation across artifacts: `extern_signature!` writes this `fn`
    /// type from the signature's own parameter list and `#[extern_fn]`
    /// assigns every mono glue to it before taking the address
    /// `call_now` runs.
    type Now;
    /// As `Now`, for a glue whose body suspends.
    type Later;

    /// # Safety
    /// `value` is the glue of an instance of this signature, the receiver
    /// named in `ctx` holds a value of the type that instance stands at,
    /// and that instance's body returns rather than suspends.
    unsafe fn call_now(value: Rt::Value, ctx: &mut Ctx<'_, Rt>, rest: Self::Rest<'_>) -> Self::Ret;

    /// # Safety
    /// As `call_now`'s, without its last clause.
    unsafe fn call_later<'a>(
        value: Rt::Value,
        ctx: &'a mut Ctx<'_, Rt>,
        rest: Self::Rest<'a>,
    ) -> BoxFuture<'a, Self::Ret>
    where
        Self::Ret: Send;
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
    T: OneValue<Rt, C>,
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
        // SAFETY: the caller's contract: the storage outlives the call,
        // which is where RFC-0018 keeps the reference.
        Ref::new(unsafe { rt.reference(crossed) })
    }
}

impl<T, C, Rt> RestoreExclusive<Rt> for ByRef<T, Mut, C>
where
    T: OneValue<Rt, C>,
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
        Ref::new(unsafe { rt.reference(crossed) })
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
