use std::marker::PhantomData;

use crate::crossing::Crossing;
use crate::kind::{OptionKind, ResultKind, ScalarKind, StringKind};
use crate::repr;
use crate::runtime::Runtime;

/// A value the runtime holds, its type kept by the marker `X` alone: its
/// layout is `R::Value`'s for every `X`. It is invariant in `X` (rfc_compiler
/// types §4), made only through the runtime, lent only through it, and never
/// taken out; it is moved and dropped whole.
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
        X::view(rt, &self.value, lent)
    }

    pub fn as_mut<O>(&mut self, rt: &R, lent: impl FnOnce(&mut X::Viewed) -> O) -> O
    where
        X: Lend<R>,
    {
        X::view_mut(rt, &mut self.value, lent)
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
        R::release(self.value);
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

    fn take(_rt: &R, value: R::Value) -> Self {
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

    fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&Self::Viewed) -> O) -> O;
    fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut Self::Viewed) -> O) -> O;
}

macro_rules! scalars {
    ($($scalar:ty),* $(,)?) => {$(
        impl<R> Lend<R> for $scalar
        where
            R: Runtime,
        {
            type Viewed = $scalar;

            fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&$scalar) -> O) -> O {
                <R::Scalar as ScalarKind<R, $scalar>>::view(rt, value, lent)
            }

            fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut $scalar) -> O) -> O {
                <R::Scalar as ScalarKind<R, $scalar>>::view_mut(rt, value, lent)
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

    fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&String) -> O) -> O {
        <R::String as StringKind<R>>::view(rt, value, lent)
    }

    fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut String) -> O) -> O {
        <R::String as StringKind<R>>::view_mut(rt, value, lent)
    }
}

impl<R, Y> Lend<R> for Option<Y>
where
    R: Runtime,
    Y: Crossing<R>,
{
    type Viewed = Option<Uniform<R, Y>>;

    fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&Self::Viewed) -> O) -> O {
        <R::Option as OptionKind<R>>::view(rt, value, |option| lent(repr::uniform_option(option)))
    }

    fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut Self::Viewed) -> O) -> O {
        <R::Option as OptionKind<R>>::view_mut(rt, value, |option| lent(repr::uniform_option_mut(option)))
    }
}

impl<R, Y, E> Lend<R> for Result<Y, E>
where
    R: Runtime,
    Y: Crossing<R>,
    E: Crossing<R>,
{
    type Viewed = Result<Uniform<R, Y>, Uniform<R, E>>;

    fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&Self::Viewed) -> O) -> O {
        <R::Result as ResultKind<R>>::view(rt, value, |result| lent(repr::uniform_result(result)))
    }

    fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut Self::Viewed) -> O) -> O {
        <R::Result as ResultKind<R>>::view_mut(rt, value, |result| lent(repr::uniform_result_mut(result)))
    }
}
