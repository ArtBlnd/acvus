//! `Crosses` for the built-in types (runtime contract §2, crossing §2): the
//! scalars, `String`, `Vec` and `Uniform` cross opaque; tuples, `Option` and
//! `Result` are the language's values.

use crate::boundary::entry::Settled;
use crate::boundary::uniform::Uniform;
use crate::runtime::{Crosses, Runtime};
use crate::shape::{LentParts, LentPartsMut, Projected};

/// `Option`'s and `Result`'s tags: their variants' places in the order of
/// their names' text (compiler: types §2).
pub(crate) mod tags {
    pub(crate) const NONE: u32 = 0;
    pub(crate) const SOME: u32 = 1;
    pub(crate) const ERR: u32 = 0;
    pub(crate) const OK: u32 = 1;
}

/// The items of an opaque type: given and taken whole, lent as a guard.
macro_rules! opaque_crossing {
    () => {
        type Lent<'c, Rt: Runtime>
            = crate::seal::opaque::Lent<'c, Rt, Self>
        where
            Self: 'c;
        type LentMut<'c, Rt: Runtime>
            = crate::seal::opaque::LentMut<'c, Rt, Self>
        where
            Self: 'c;

        fn give<Rt: Runtime>(self, rt: &Rt, at: Settled<'_>) -> Rt::Value {
            crate::seal::opaque::give(self, rt, at)
        }

        fn take<Rt: Runtime>(v: Rt::Value, rt: &Rt, at: Settled<'_>) -> Self {
            crate::seal::opaque::take(v, rt, at)
        }

        fn lent<'c, Rt: Runtime>(rt: &'c Rt, place: &'c Rt::Value, at: Settled<'c>) -> Self::Lent<'c, Rt>
        where
            Self: 'c,
        {
            crate::seal::opaque::lent(rt, place, at)
        }

        fn lent_mut<'c, Rt: Runtime>(rt: &'c Rt, place: &'c mut Rt::Value, at: Settled<'c>) -> Self::LentMut<'c, Rt>
        where
            Self: 'c,
        {
            crate::seal::opaque::lent_mut(rt, place, at)
        }
    };
}

/// The lending items of a transparent structured type: lent as its
/// projection. Its `give` and `take` are written at each constructor.
macro_rules! projected_crossing {
    () => {
        type Lent<'c, Rt: Runtime>
            = <Self as Projected<Rt>>::Ref<'c>
        where
            Self: 'c;
        type LentMut<'c, Rt: Runtime>
            = <Self as Projected<Rt>>::Mut<'c>
        where
            Self: 'c;

        fn lent<'c, Rt: Runtime>(rt: &'c Rt, place: &'c Rt::Value, at: Settled<'c>) -> Self::Lent<'c, Rt>
        where
            Self: 'c,
        {
            <Self as Projected<Rt>>::project(LentParts::of_the_entry(rt, place, at))
        }

        fn lent_mut<'c, Rt: Runtime>(rt: &'c Rt, place: &'c mut Rt::Value, at: Settled<'c>) -> Self::LentMut<'c, Rt>
        where
            Self: 'c,
        {
            let aggregate = <Self as Projected<Rt>>::AGGREGATE;
            <Self as Projected<Rt>>::project_mut(LentPartsMut::of_the_entry(rt, place, aggregate, at))
        }
    };
}

macro_rules! opaque_types {
    ($($t:ty),+ $(,)?) => {$(
        impl Crosses for $t {
            opaque_crossing!();
        }
    )+};
}

opaque_types!(i8, i16, i32, i64, u8, u16, u32, u64, f64, bool, char, String, ());

// A sequence leaves as a `Vec` (limitations §1): a runtime gives it at the
// settled array type.
impl<T: Crosses> Crosses for Vec<T> {
    opaque_crossing!();
}

impl<R: Runtime, X> Crosses for Uniform<R, X> {
    opaque_crossing!();
}

impl<T: Crosses> Crosses for Option<T> {
    projected_crossing!();

    fn give<Rt: Runtime>(self, rt: &Rt, at: Settled<'_>) -> Rt::Value {
        match self {
            None => rt.make_variant(tags::NONE, None, at),
            Some(value) => {
                let payload = value.give(rt, at);
                rt.make_variant(tags::SOME, Some(payload), at)
            }
        }
    }

    fn take<Rt: Runtime>(v: Rt::Value, rt: &Rt, at: Settled<'_>) -> Self {
        Uniform::<Rt, Self>::of_the_crossing(v).view(rt, at).map(|value| value.take(rt, at))
    }
}

impl<T: Crosses, E: Crosses> Crosses for Result<T, E> {
    projected_crossing!();

    fn give<Rt: Runtime>(self, rt: &Rt, at: Settled<'_>) -> Rt::Value {
        let (tag, payload) = match self {
            Ok(value) => (tags::OK, value.give(rt, at)),
            Err(error) => (tags::ERR, error.give(rt, at)),
        };
        rt.make_variant(tag, Some(payload), at)
    }

    fn take<Rt: Runtime>(v: Rt::Value, rt: &Rt, at: Settled<'_>) -> Self {
        match Uniform::<Rt, Self>::of_the_crossing(v).view(rt, at) {
            Ok(value) => Ok(value.take(rt, at)),
            Err(error) => Err(error.take(rt, at)),
        }
    }
}

/// Tuples up to twelve parts, the arity the crossing's tuples stop at.
macro_rules! tuple_crossings {
    ($(($($p:ident $v:ident),+))+) => {$(
        impl<$($p: Crosses),+> Crosses for ($($p,)+) {
            projected_crossing!();

            fn give<Rt: Runtime>(self, rt: &Rt, at: Settled<'_>) -> Rt::Value {
                let ($($v,)+) = self;
                let parts = [$($v.give(rt, at)),+];
                rt.make_tuple(parts.into_iter(), at)
            }

            fn take<Rt: Runtime>(v: Rt::Value, rt: &Rt, at: Settled<'_>) -> Self {
                let ($($v,)+) = Uniform::<Rt, Self>::of_the_crossing(v).view(rt, at);
                ($($v.take(rt, at),)+)
            }
        }
    )+};
}

tuple_crossings! {
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
