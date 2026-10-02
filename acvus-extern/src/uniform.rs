#![allow(unsafe_code)]

use std::marker::PhantomData;

use crate::crossing::Crossing;
use crate::kind::{OptionKind, ResultKind, ScalarKind, StringKind};
use crate::repr;
use crate::runtime::Runtime;

/// A value the runtime holds, its type kept by the marker `X` alone: its
/// layout is `R::Value`'s for every `X`. It is invariant in `X` (rfc_compiler
/// types §4), made only by `new` or by the `unsafe` `Crossing::take`, lent
/// only through the runtime, and never taken out; it is moved and dropped
/// whole. So it always owns a live value made as `X`, which its views and its
/// drop rest on.
#[repr(transparent)]
pub struct Uniform<R, X>
where
    R: Runtime,
{
    value: R::Value,
    marker: PhantomData<fn(X) -> X>,
}

impl<R, X> Uniform<R, X>
where
    R: Runtime,
{
    /// `X: 'static` is the runtime contract's: generic code, whose type
    /// variables carry no `'static`, cannot wrap one in a `Uniform`.
    pub fn new(rt: &R, value: X) -> Self
    where
        X: Crossing<R> + 'static,
    {
        Uniform { value: value.give(rt), marker: PhantomData }
    }

    pub fn as_ref<O>(&self, rt: &R, lent: impl FnOnce(&X::Viewed) -> O) -> O
    where
        X: Lend<R>,
    {
        // SAFETY: `self` owns a live value made as `X`.
        unsafe { X::view(rt, &self.value, lent) }
    }

    pub fn as_mut<O>(&mut self, rt: &R, lent: impl FnOnce(&mut X::Viewed) -> O) -> O
    where
        X: Lend<R>,
    {
        // SAFETY: `self` owns a live value made as `X`, held exclusively
        // through `&mut self`.
        unsafe { X::view_mut(rt, &mut self.value, lent) }
    }

    pub fn get(&self, rt: &R) -> X
    where
        X: Lend<R, Viewed = X> + Copy,
    {
        self.as_ref(rt, |value| *value)
    }
}

impl<R, X> Drop for Uniform<R, X>
where
    R: Runtime,
{
    fn drop(&mut self) {
        // SAFETY: `self` owns its value, and is dropped once.
        unsafe { R::release(self.value) }
    }
}

impl<R, X> Crossing<R> for Uniform<R, X>
where
    R: Runtime,
{
    fn give(self, _rt: &R) -> R::Value {
        let value = self.value;
        std::mem::forget(self);
        value
    }

    unsafe fn take(_rt: &R, value: R::Value) -> Self {
        Uniform { value, marker: PhantomData }
    }
}

/// A type's view, the counterpart of `Crossing`: which kind lends a value of
/// it, and what that kind lends as. A struct, an enum or a tuple has none,
/// since none is lent (the extern note: `Structural`).
pub trait Lend<R>: Crossing<R>
where
    R: Runtime,
{
    type Viewed: ?Sized;

    /// # Safety
    ///
    /// `value` is live and was given as this type, or as one of its layout as
    /// the extern note's Type variables admit.
    unsafe fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&Self::Viewed) -> O) -> O;

    /// # Safety
    ///
    /// As `view`, and the caller holds `value` exclusively while `lent` runs.
    unsafe fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut Self::Viewed) -> O) -> O;
}

macro_rules! scalars {
    ($($scalar:ty),* $(,)?) => {$(
        impl<R> Lend<R> for $scalar
        where
            R: Runtime,
        {
            type Viewed = $scalar;

            unsafe fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&$scalar) -> O) -> O {
                // SAFETY: the caller's premise.
                unsafe { <R::Scalar as ScalarKind<R, $scalar>>::view(rt, value, lent) }
            }

            unsafe fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut $scalar) -> O) -> O {
                // SAFETY: the caller's premise.
                unsafe { <R::Scalar as ScalarKind<R, $scalar>>::view_mut(rt, value, lent) }
            }
        }
    )*};
}

scalars!(i8, i16, i32, i64, u8, u16, u32, u64, f64, bool, char, ());

impl<R> Lend<R> for String
where
    R: Runtime,
{
    type Viewed = String;

    unsafe fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&String) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::String as StringKind<R>>::view(rt, value, lent) }
    }

    unsafe fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut String) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::String as StringKind<R>>::view_mut(rt, value, lent) }
    }
}

impl<R, Y> Lend<R> for Option<Y>
where
    R: Runtime,
    Y: Crossing<R>,
{
    type Viewed = Option<Uniform<R, Y>>;

    unsafe fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&Self::Viewed) -> O) -> O {
        // SAFETY: the caller's premise; the option's payload was given as
        // `Y`, so it may be seen as a `Uniform<R, Y>`.
        unsafe { <R::Option as OptionKind<R>>::view(rt, value, |option| lent(repr::uniform_option(option))) }
    }

    unsafe fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut Self::Viewed) -> O) -> O {
        // SAFETY: as `view`, exclusively.
        unsafe { <R::Option as OptionKind<R>>::view_mut(rt, value, |option| lent(repr::uniform_option_mut(option))) }
    }
}

impl<R, Y, E> Lend<R> for Result<Y, E>
where
    R: Runtime,
    Y: Crossing<R>,
    E: Crossing<R>,
{
    type Viewed = Result<Uniform<R, Y>, Uniform<R, E>>;

    unsafe fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&Self::Viewed) -> O) -> O {
        // SAFETY: the caller's premise; either payload was given as its own
        // type, so it may be seen as a `Uniform` of it.
        unsafe { <R::Result as ResultKind<R>>::view(rt, value, |result| lent(repr::uniform_result(result))) }
    }

    unsafe fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut Self::Viewed) -> O) -> O {
        // SAFETY: as `view`, exclusively.
        unsafe { <R::Result as ResultKind<R>>::view_mut(rt, value, |result| lent(repr::uniform_result_mut(result))) }
    }
}
