//! The right to cross between a Rust type and the runtime's value.
//!
//! There are two capabilities and not one, and that is a decision. A
//! crossing boxes and unboxes through the runtime, so `Crossing` carries the
//! `&Rt` and stands where the `rt` parameter stood. Holding a bare word in an
//! `Owned` is the runtime's storage work — a register file, a context page, a
//! journal — where no `&Rt` is at hand, so `Holding` carries nothing.
//!
//! Both carry a lifetime so that a crossing impl, handed one at a lifetime
//! local to the call it serves, cannot keep it where a handler reads later: a
//! `static` and a thread-local ask `'static`.

use std::marker::PhantomData;
use std::ops::Deref;

use crate::instance::{Instance, InstanceOf, ReadsItsReceiver, Signature, StepsItsReceiver};
use crate::runtime::Runtime;

pub struct Crossing<'a, Rt>
where
    Rt: Runtime,
{
    rt: &'a Rt,
}

impl<'a, Rt> Clone for Crossing<'a, Rt>
where
    Rt: Runtime,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, Rt> Copy for Crossing<'a, Rt> where Rt: Runtime {}

impl<'a, Rt> Crossing<'a, Rt>
where
    Rt: Runtime,
{
    /// # Safety
    /// The caller is the glue a macro writes or the runtime, it crosses each
    /// value at the type the acvus checker settled for it, and it hands the
    /// capability to no handler body.
    #[inline(always)]
    pub unsafe fn new(rt: &'a Rt) -> Self {
        Crossing { rt }
    }

    #[inline(always)]
    pub fn rt(self) -> &'a Rt {
        self.rt
    }

    #[inline(always)]
    pub fn holding(self) -> Holding<'a, Rt> {
        Holding(PhantomData)
    }

    /// # Safety
    /// `value` was made by `Runtime::instance_value` from an entry of an
    /// instance of `S` standing at the type `I` is filled with, and the
    /// entry is live for `'r`.
    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn instance<'r, S, I, T>(self, value: Rt::Value) -> InstanceOf<'r, S, I, Rt, T>
    where
        S: Signature<Rt>,
        S::Mode: ReadsItsReceiver,
    {
        // SAFETY: the caller's contract.
        unsafe { InstanceOf::at(value) }
    }

    /// # Safety
    /// `value` was made by `Runtime::instance_value` from an entry of an
    /// instance of `S` standing at the type of `recv`, as the checker chose
    /// it at the site `recv` was passed to, and the entry is live for `'r`.
    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn instance_owning<'r, S, R, T>(
        self,
        recv: R,
        value: Rt::Value,
    ) -> Instance<'r, S, R, Rt, T>
    where
        S: Signature<Rt>,
        S::Mode: StepsItsReceiver,
    {
        // SAFETY: the caller's contract.
        unsafe { Instance::own(recv, value) }
    }
}

impl<'a, Rt> Deref for Crossing<'a, Rt>
where
    Rt: Runtime,
{
    type Target = Rt;

    #[inline(always)]
    fn deref(&self) -> &Rt {
        self.rt
    }
}

pub struct Holding<'a, Rt>(PhantomData<(&'a (), fn() -> Rt)>)
where
    Rt: Runtime;

impl<'a, Rt> Clone for Holding<'a, Rt>
where
    Rt: Runtime,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, Rt> Copy for Holding<'a, Rt> where Rt: Runtime {}

impl<'a, Rt> Holding<'a, Rt>
where
    Rt: Runtime,
{
    /// # Safety
    /// As `Crossing::new`'s.
    #[inline(always)]
    pub unsafe fn new() -> Self {
        Holding(PhantomData)
    }
}
