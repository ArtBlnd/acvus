use crate::kind::{OptionKind, ResultKind, ScalarKind, StringKind, TupleKind};
use crate::runtime::Runtime;

/// A type's crossing by value. `give` hands the runtime a value and `take`
/// takes one back whole, each through the type's own kind, so an entry or a
/// derive opens an argument or a field by its type and never by its name.
pub trait Crossing<R>: Sized
where
    R: Runtime,
{
    fn give(self, rt: &R) -> R::Value;
    fn take(rt: &R, value: R::Value) -> Self;
}

macro_rules! scalars {
    ($($scalar:ty),* $(,)?) => {$(
        impl<R> Crossing<R> for $scalar
        where
            R: Runtime,
        {
            fn give(self, rt: &R) -> R::Value {
                <R::Scalar as ScalarKind<R, $scalar>>::make(rt, self)
            }

            fn take(rt: &R, value: R::Value) -> Self {
                <R::Scalar as ScalarKind<R, $scalar>>::take(rt, value)
            }
        }
    )*};
}

scalars!(i8, i16, i32, i64, u8, u16, u32, u64, f64, bool, char, ());

impl<R> Crossing<R> for String
where
    R: Runtime,
{
    fn give(self, rt: &R) -> R::Value {
        <R::String as StringKind<R>>::make(rt, self)
    }

    fn take(rt: &R, value: R::Value) -> Self {
        <R::String as StringKind<R>>::take(rt, value)
    }
}

impl<R, T> Crossing<R> for Option<T>
where
    R: Runtime,
    T: Crossing<R>,
{
    fn give(self, rt: &R) -> R::Value {
        <R::Option as OptionKind<R>>::make(rt, self.map(|payload| payload.give(rt)))
    }

    fn take(rt: &R, value: R::Value) -> Self {
        <R::Option as OptionKind<R>>::take(rt, value).map(|payload| T::take(rt, payload))
    }
}

impl<R, T, E> Crossing<R> for Result<T, E>
where
    R: Runtime,
    T: Crossing<R>,
    E: Crossing<R>,
{
    fn give(self, rt: &R) -> R::Value {
        let result = self.map(|payload| payload.give(rt)).map_err(|payload| payload.give(rt));
        <R::Result as ResultKind<R>>::make(rt, result)
    }

    fn take(rt: &R, value: R::Value) -> Self {
        <R::Result as ResultKind<R>>::take(rt, value).map(|payload| T::take(rt, payload)).map_err(|payload| E::take(rt, payload))
    }
}

/// Tuples cross at every arity std's own tuple impls reach, one rule for all.
macro_rules! tuples {
    ($(($($part:ident $value:ident),+))+) => {$(
        impl<R, $($part),+> Crossing<R> for ($($part,)+)
        where
            R: Runtime,
            $($part: Crossing<R>,)+
        {
            fn give(self, rt: &R) -> R::Value {
                let ($($value,)+) = self;
                <R::Tuple as TupleKind<R>>::make(rt, [$($value.give(rt)),+])
            }

            fn take(rt: &R, value: R::Value) -> Self {
                let [$($value),+] = <R::Tuple as TupleKind<R>>::take(rt, value);
                ($($part::take(rt, $value),)+)
            }
        }
    )+};
}

tuples! {
    (A a)
    (A a, B b)
    (A a, B b, C c)
    (A a, B b, C c, D d)
    (A a, B b, C c, D d, E e)
    (A a, B b, C c, D d, E e, F f)
    (A a, B b, C c, D d, E e, F f, G g)
    (A a, B b, C c, D d, E e, F f, G g, H h)
    (A a, B b, C c, D d, E e, F f, G g, H h, I i)
    (A a, B b, C c, D d, E e, F f, G g, H h, I i, J j)
    (A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k)
    (A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k, L l)
}
