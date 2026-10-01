//! The language's types' crossings (runtime contract §1): each is its type's
//! own, composed of its kind's operations over its parts' crossings. An extern
//! type's crossing is the derive's.
//!
//! A value lent, held or named by a reference, is opened as its projection
//! (crossing §2): a reference to a type the runtime holds as itself, and of an
//! option, a result or a tuple the same structure over its parts' projections.

use crate::boundary::entry::Settled;
use crate::boundary::uniform::Uniform;
use crate::runtime::{Crosses, OptionKind, PrimitiveKind, RefKind, Referent, ResultKind, Runtime, StringKind, TupleKind};

/// A type the runtime holds as itself: opened as a reference to it by its
/// kind.
macro_rules! held_as_itself {
    ($t:ty, $kind:ident, $kind_trait:ident) => {
        type Open<'a> = &'a $t;
        type OpenMut<'a> = &'a mut $t;

        fn open<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> &'a $t
        where
            Self: 'a,
        {
            <R::$kind as $kind_trait<R>>::shared(rt, place, at)
        }

        fn open_mut<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> &'a mut $t
        where
            Self: 'a,
        {
            <R::$kind as $kind_trait<R>>::exclusive(rt, place, at)
        }

        fn open_ref<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> &'a $t
        where
            Self: 'a,
        {
            <R::Ref as RefKind<R>>::target(rt, reference, at)
        }

        fn open_ref_mut<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> &'a mut $t
        where
            Self: 'a,
        {
            <R::Ref as RefKind<R>>::target_mut(rt, reference, at)
        }
    };
}

macro_rules! primitive_crossings {
    ($($t:ty => $decl:expr),+ $(,)?) => {$(
        impl Referent for $t {}

        impl<R: Runtime> Crosses<R> for $t {
            held_as_itself!($t, Primitive, PrimitiveKind);

            fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
                <R::Primitive as PrimitiveKind<R>>::give(rt, self, at)
            }

            fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
                <R::Primitive as PrimitiveKind<R>>::take(rt, v, at)
            }
        }
    )+};
}

primitives!(primitive_crossings);

impl Referent for String {}
impl Referent for str {}
impl<R: Runtime, X> Referent for Uniform<R, X> {}
impl<R: Runtime, X> Referent for [Uniform<R, X>] {}

impl<R: Runtime> Crosses<R> for String {
    held_as_itself!(String, String, StringKind);

    fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
        <R::String as StringKind<R>>::give(rt, self, at)
    }

    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
        <R::String as StringKind<R>>::take(rt, v, at)
    }
}

impl<R: Runtime, T: Crosses<R>> Crosses<R> for Option<T> {
    type Open<'a>
        = Option<T::Open<'a>>
    where
        T: 'a;
    type OpenMut<'a>
        = Option<T::OpenMut<'a>>
    where
        T: 'a;

    fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
        <R::Option as OptionKind<R>>::give(rt, self.map(|v| v.give(rt, at)), at)
    }

    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
        <R::Option as OptionKind<R>>::take(rt, v, at).map(|v| T::take(rt, v, at))
    }

    fn open<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> Self::Open<'a>
    where
        T: 'a,
    {
        <R::Option as OptionKind<R>>::shared(rt, place, at).map(|part| T::open(rt, part, at))
    }

    fn open_mut<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
    where
        T: 'a,
    {
        <R::Option as OptionKind<R>>::exclusive(rt, place, at).map(|part| T::open_mut(rt, part, at))
    }

    fn open_ref<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::Open<'a>
    where
        T: 'a,
    {
        <R::Option as OptionKind<R>>::shared_ref(rt, reference, at).map(|part| T::open(rt, part, at))
    }

    fn open_ref_mut<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
    where
        T: 'a,
    {
        <R::Option as OptionKind<R>>::exclusive_ref(rt, reference, at).map(|part| T::open_mut(rt, part, at))
    }
}

impl<R: Runtime, T: Crosses<R>, E: Crosses<R>> Crosses<R> for Result<T, E> {
    type Open<'a>
        = Result<T::Open<'a>, E::Open<'a>>
    where
        Self: 'a;
    type OpenMut<'a>
        = Result<T::OpenMut<'a>, E::OpenMut<'a>>
    where
        Self: 'a;

    fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
        let parts = self.map(|v| v.give(rt, at)).map_err(|e| e.give(rt, at));
        <R::Result as ResultKind<R>>::give(rt, parts, at)
    }

    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
        <R::Result as ResultKind<R>>::take(rt, v, at).map(|v| T::take(rt, v, at)).map_err(|e| E::take(rt, e, at))
    }

    fn open<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> Self::Open<'a>
    where
        Self: 'a,
    {
        let parts = <R::Result as ResultKind<R>>::shared(rt, place, at);
        parts.map(|part| T::open(rt, part, at)).map_err(|part| E::open(rt, part, at))
    }

    fn open_mut<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
    where
        Self: 'a,
    {
        let parts = <R::Result as ResultKind<R>>::exclusive(rt, place, at);
        parts.map(|part| T::open_mut(rt, part, at)).map_err(|part| E::open_mut(rt, part, at))
    }

    fn open_ref<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::Open<'a>
    where
        Self: 'a,
    {
        let parts = <R::Result as ResultKind<R>>::shared_ref(rt, reference, at);
        parts.map(|part| T::open(rt, part, at)).map_err(|part| E::open(rt, part, at))
    }

    fn open_ref_mut<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
    where
        Self: 'a,
    {
        let parts = <R::Result as ResultKind<R>>::exclusive_ref(rt, reference, at);
        parts.map(|part| T::open_mut(rt, part, at)).map_err(|part| E::open_mut(rt, part, at))
    }
}

/// A type variable's value is the runtime's value already: it crosses as
/// itself, and is lent as itself.
impl<R: Runtime, X> Crosses<R> for Uniform<R, X> {
    type Open<'a>
        = &'a Uniform<R, X>
    where
        X: 'a;
    type OpenMut<'a>
        = &'a mut Uniform<R, X>
    where
        X: 'a;

    fn give(self, _rt: &R, _at: Settled<'_>) -> R::Value {
        self.into_value()
    }

    fn take(_rt: &R, v: R::Value, _at: Settled<'_>) -> Self {
        Uniform::of_the_crossing(v)
    }

    fn open<'a>(_rt: &'a R, place: &'a R::Value, _at: Settled<'a>) -> Self::Open<'a>
    where
        X: 'a,
    {
        Uniform::at_place(place)
    }

    fn open_mut<'a>(_rt: &'a R, place: &'a mut R::Value, _at: Settled<'a>) -> Self::OpenMut<'a>
    where
        X: 'a,
    {
        Uniform::at_place_mut(place)
    }

    fn open_ref<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::Open<'a>
    where
        X: 'a,
    {
        <R::Ref as RefKind<R>>::target(rt, reference, at)
    }

    fn open_ref_mut<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
    where
        X: 'a,
    {
        <R::Ref as RefKind<R>>::target_mut(rt, reference, at)
    }
}

/// A tuple, made, taken and projected by its kind over its parts' crossings.
macro_rules! tuple_crossings {
    ($($n:literal ($($p:ident $v:ident $i:tt),+))+) => {$(
        impl<R: Runtime, $($p: Crosses<R>),+> Crosses<R> for ($($p,)+) {
            type Open<'a>
                = ($($p::Open<'a>,)+)
            where
                Self: 'a;
            type OpenMut<'a>
                = ($($p::OpenMut<'a>,)+)
            where
                Self: 'a;

            fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
                let ($($v,)+) = self;
                <R::Tuple as TupleKind<R>>::give::<$n>(rt, [$($v.give(rt, at)),+], at)
            }

            fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
                let [$($v),+] = <R::Tuple as TupleKind<R>>::take::<$n>(rt, v, at);
                ($($p::take(rt, $v, at),)+)
            }

            fn open<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> Self::Open<'a>
            where
                Self: 'a,
            {
                let [$($v),+] = <R::Tuple as TupleKind<R>>::shared::<$n>(rt, place, at);
                ($($p::open(rt, $v, at),)+)
            }

            fn open_mut<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
            where
                Self: 'a,
            {
                let [$($v),+] = <R::Tuple as TupleKind<R>>::exclusive::<$n>(rt, place, at);
                ($($p::open_mut(rt, $v, at),)+)
            }

            fn open_ref<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::Open<'a>
            where
                Self: 'a,
            {
                let [$($v),+] = <R::Tuple as TupleKind<R>>::shared_ref::<$n>(rt, reference, at);
                ($($p::open(rt, $v, at),)+)
            }

            fn open_ref_mut<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
            where
                Self: 'a,
            {
                let [$($v),+] = <R::Tuple as TupleKind<R>>::exclusive_ref::<$n>(rt, reference, at);
                ($($p::open_mut(rt, $v, at),)+)
            }
        }
    )+};
}

tuples!(tuple_crossings);
