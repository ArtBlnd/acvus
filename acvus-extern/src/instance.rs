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
    /// The arguments after the first.
    type Rest<'a>;
    type Ret;

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
