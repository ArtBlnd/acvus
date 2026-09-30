//! The language's types' crossings (runtime contract §1): each is its type's
//! own, composed of its kind's operations over its parts' crossings. An extern
//! type's crossing is the derive's.

use crate::boundary::entry::Settled;
use crate::boundary::uniform::Uniform;
use crate::runtime::{Crosses, Runtime};

macro_rules! primitives {
    ($($t:ty),+ $(,)?) => {$(
        impl<R: Runtime> Crosses<R> for $t {
            fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
                rt.give_primitive(self, at)
            }

            fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
                rt.take_primitive(v, at)
            }
        }
    )+};
}

primitives!(i8, i16, i32, i64, u8, u16, u32, u64, f64, bool, char, ());

impl<R: Runtime> Crosses<R> for String {
    fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
        rt.give_string(self, at)
    }

    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
        rt.take_string(v, at)
    }
}

impl<R: Runtime, T: Crosses<R>> Crosses<R> for Option<T> {
    fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
        rt.give_option(self.map(|v| v.give(rt, at)), at)
    }

    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
        rt.take_option(v, at).map(|v| T::take(rt, v, at))
    }
}

impl<R: Runtime, T: Crosses<R>, E: Crosses<R>> Crosses<R> for Result<T, E> {
    fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
        rt.give_result(self.map(|v| v.give(rt, at)).map_err(|e| e.give(rt, at)), at)
    }

    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
        rt.take_result(v, at).map(|v| T::take(rt, v, at)).map_err(|e| E::take(rt, e, at))
    }
}

/// A type variable's value is the runtime's value already: it crosses as
/// itself.
impl<R: Runtime, X> Crosses<R> for Uniform<R, X> {
    fn give(self, _rt: &R, _at: Settled<'_>) -> R::Value {
        self.into_value()
    }

    fn take(_rt: &R, v: R::Value, _at: Settled<'_>) -> Self {
        Uniform::of_the_crossing(v)
    }
}

macro_rules! tuples {
    ($($n:literal ($($p:ident $v:ident),+))+) => {$(
        impl<R: Runtime, $($p: Crosses<R>),+> Crosses<R> for ($($p,)+) {
            fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
                let ($($v,)+) = self;
                rt.give_tuple::<$n>([$($v.give(rt, at)),+], at)
            }

            fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
                let [$($v),+] = rt.take_tuple::<$n>(v, at);
                ($($p::take(rt, $v, at),)+)
            }
        }
    )+};
}

tuples! {
    1 (A a)
    2 (A a, B b)
    3 (A a, B b, C c)
    4 (A a, B b, C c, D d)
    5 (A a, B b, C c, D d, E e)
    6 (A a, B b, C c, D d, E e, F f)
    7 (A a, B b, C c, D d, E e, F f, G g)
    8 (A a, B b, C c, D d, E e, F f, G g, H h)
    9 (A a, B b, C c, D d, E e, F f, G g, H h, I i)
    10 (A a, B b, C c, D d, E e, F f, G g, H h, I i, J j)
    11 (A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k)
    12 (A a, B b, C c, D d, E e, F f, G g, H h, I i, J j, K k, L l)
}
