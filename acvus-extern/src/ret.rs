//! A shared signature's result as it crosses between an instance's glue and
//! a requirer (RFC-0068 rule 6). `extern_signature!` names one of the three
//! shapes as its module's `Shape`: `Concrete` where the result names none of
//! the signature's type variables, `OptionOfVar` where it is an `Option` of
//! one of them, and `Whole` where a variable stands anywhere else in it. The
//! crossing of each shape is this module's, so the signature's module asserts
//! nothing (RFC-0080 rule 2).

use std::marker::PhantomData;

use crate::{Crossing, OneValue, Runtime};

pub struct Concrete<T>(PhantomData<fn() -> T>);

pub struct OptionOfVar;

pub struct Whole;

mod sealed {
    pub trait Shape {}

    pub trait Returned<S, Rt> {}
}

impl<T> sealed::Shape for Concrete<T> {}
impl sealed::Shape for OptionOfVar {}
impl sealed::Shape for Whole {}

/// `Ret` is the one type every instance of a signature at this shape returns
/// its result as, and the type the signature module's `Now` and `Later`
/// return.
pub trait Shape<Rt>: sealed::Shape
where
    Rt: Runtime,
{
    type Ret;
}

impl<T, Rt> Shape<Rt> for Concrete<T>
where
    Rt: Runtime,
{
    type Ret = T;
}

impl<Rt> Shape<Rt> for OptionOfVar
where
    Rt: Runtime,
{
    type Ret = Option<Rt::Value>;
}

impl<Rt> Shape<Rt> for Whole
where
    Rt: Runtime,
{
    type Ret = Rt::Value;
}

/// # Safety
/// The word `cross` hands back is exactly the result at the type the checker
/// settled for it; it crosses nothing else with the capability; and it keeps
/// no capability past the call.
pub unsafe trait Returned<S, Rt>: sealed::Returned<S, Rt> + Sized
where
    S: Shape<Rt>,
    Rt: Runtime,
{
    fn cross(rt: Crossing<'_, Rt>, r: Self) -> S::Ret;
}

impl<T, Rt> sealed::Returned<Concrete<T>, Rt> for T where Rt: Runtime {}

impl<T, Rt> sealed::Returned<OptionOfVar, Rt> for Option<T>
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
}

impl<T, Rt> sealed::Returned<Whole, Rt> for T
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
}

// SAFETY: a concrete result crosses as itself; the capability crosses
// nothing and is not kept.
unsafe impl<T, Rt> Returned<Concrete<T>, Rt> for T
where
    Rt: Runtime,
{
    #[inline(always)]
    fn cross(_: Crossing<'_, Rt>, r: Self) -> T {
        r
    }
}

// SAFETY: the word is `T`'s own `erase` of the present value, at the `T` the
// checker settled; nothing else crosses and the capability is not kept.
unsafe impl<T, Rt> Returned<OptionOfVar, Rt> for Option<T>
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    #[inline(always)]
    fn cross(rt: Crossing<'_, Rt>, r: Self) -> Option<Rt::Value> {
        r.map(|v| <T as OneValue<Rt>>::erase(v, rt))
    }
}

// SAFETY: the word is `T`'s own `erase` of the result, at the `T` the checker
// settled; nothing else crosses and the capability is not kept.
unsafe impl<T, Rt> Returned<Whole, Rt> for T
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    #[inline(always)]
    fn cross(rt: Crossing<'_, Rt>, r: Self) -> Rt::Value {
        <T as OneValue<Rt>>::erase(r, rt)
    }
}
