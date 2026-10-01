//! The language's types' crossings (runtime contract §1): each is its type's
//! own, composed of its kind's operations over its parts' crossings. An extern
//! type's crossing is the derive's.

use std::ops::{Deref, DerefMut};

use crate::boundary::entry::Settled;
use crate::boundary::uniform::Uniform;
use crate::runtime::{Crosses, OptionKind, PrimitiveKind, RefKind, Referent, ResultKind, Runtime, StringKind, TupleKind};

/// A view of parts, lent: an option's, a result's or a tuple's parts as
/// `Uniform`s, the view dereferencing to them.
pub struct Parted<V>(V);

impl<V> Deref for Parted<V> {
    type Target = V;

    fn deref(&self) -> &V {
        &self.0
    }
}

impl<V> DerefMut for Parted<V> {
    fn deref_mut(&mut self) -> &mut V {
        &mut self.0
    }
}

macro_rules! primitive_crossings {
    ($($t:ty => $decl:expr),+ $(,)?) => {$(
        impl Referent for $t {}

        impl<R: Runtime> Crosses<R> for $t {
            type Open<'a> = <R::Primitive as PrimitiveKind<R>>::Shared<'a, $t>;
            type OpenMut<'a> = <R::Primitive as PrimitiveKind<R>>::Exclusive<'a, $t>;

            fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
                <R::Primitive as PrimitiveKind<R>>::give(rt, self, at)
            }

            fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
                <R::Primitive as PrimitiveKind<R>>::take(rt, v, at)
            }

            fn open<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> Self::Open<'a>
            where
                Self: 'a,
            {
                <R::Primitive as PrimitiveKind<R>>::shared(rt, place, at)
            }

            fn open_mut<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
            where
                Self: 'a,
            {
                <R::Primitive as PrimitiveKind<R>>::exclusive(rt, place, at)
            }

            fn open_ref<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::Open<'a>
            where
                Self: 'a,
            {
                <R::Primitive as PrimitiveKind<R>>::shared_ref(rt, reference, at)
            }

            fn open_ref_mut<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
            where
                Self: 'a,
            {
                <R::Primitive as PrimitiveKind<R>>::exclusive_ref(rt, reference, at)
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
    type Open<'a> = <R::String as StringKind<R>>::Shared<'a>;
    type OpenMut<'a> = <R::String as StringKind<R>>::Exclusive<'a>;

    fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
        <R::String as StringKind<R>>::give(rt, self, at)
    }

    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
        <R::String as StringKind<R>>::take(rt, v, at)
    }

    fn open<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> Self::Open<'a>
            where
                Self: 'a,
            {
        <R::String as StringKind<R>>::shared(rt, place, at)
    }

    fn open_mut<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
            where
                Self: 'a,
            {
        <R::String as StringKind<R>>::exclusive(rt, place, at)
    }

    fn open_ref<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::Open<'a>
            where
                Self: 'a,
            {
        <R::String as StringKind<R>>::shared_ref(rt, reference, at)
    }

    fn open_ref_mut<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
            where
                Self: 'a,
            {
        <R::String as StringKind<R>>::exclusive_ref(rt, reference, at)
    }
}

impl<R: Runtime, T: Crosses<R>> Crosses<R> for Option<T> {
    type Open<'a>
        = Parted<Option<&'a Uniform<R, T>>>
    where
        T: 'a;
    type OpenMut<'a>
        = Parted<Option<&'a mut Uniform<R, T>>>
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
        Parted(<R::Option as OptionKind<R>>::shared(rt, place, at).map(Uniform::at_place))
    }

    fn open_mut<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
    where
        T: 'a,
    {
        Parted(<R::Option as OptionKind<R>>::exclusive(rt, place, at).map(Uniform::at_place_mut))
    }

    fn open_ref<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::Open<'a>
    where
        T: 'a,
    {
        Parted(<R::Option as OptionKind<R>>::shared_ref(rt, reference, at).map(Uniform::at_place))
    }

    fn open_ref_mut<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
    where
        T: 'a,
    {
        Parted(<R::Option as OptionKind<R>>::exclusive_ref(rt, reference, at).map(Uniform::at_place_mut))
    }
}

impl<R: Runtime, T: Crosses<R>, E: Crosses<R>> Crosses<R> for Result<T, E> {
    type Open<'a>
        = Parted<Result<&'a Uniform<R, T>, &'a Uniform<R, E>>>
    where
        T: 'a,
        E: 'a;
    type OpenMut<'a>
        = Parted<Result<&'a mut Uniform<R, T>, &'a mut Uniform<R, E>>>
    where
        T: 'a,
        E: 'a;

    fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
        let parts = self.map(|v| v.give(rt, at)).map_err(|e| e.give(rt, at));
        <R::Result as ResultKind<R>>::give(rt, parts, at)
    }

    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
        <R::Result as ResultKind<R>>::take(rt, v, at)
            .map(|v| T::take(rt, v, at))
            .map_err(|e| E::take(rt, e, at))
    }

    fn open<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> Self::Open<'a>
    where
        Self: 'a,
    {
        let parts = <R::Result as ResultKind<R>>::shared(rt, place, at);
        Parted(parts.map(Uniform::at_place).map_err(Uniform::at_place))
    }

    fn open_mut<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
    where
        Self: 'a,
    {
        let parts = <R::Result as ResultKind<R>>::exclusive(rt, place, at);
        Parted(parts.map(Uniform::at_place_mut).map_err(Uniform::at_place_mut))
    }

    fn open_ref<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::Open<'a>
    where
        Self: 'a,
    {
        let parts = <R::Result as ResultKind<R>>::shared_ref(rt, reference, at);
        Parted(parts.map(Uniform::at_place).map_err(Uniform::at_place))
    }

    fn open_ref_mut<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
    where
        Self: 'a,
    {
        let parts = <R::Result as ResultKind<R>>::exclusive_ref(rt, reference, at);
        Parted(parts.map(Uniform::at_place_mut).map_err(Uniform::at_place_mut))
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

macro_rules! tuple_crossings {
    ($($n:literal ($($p:ident $v:ident $i:tt),+))+) => {$(
        impl<R: Runtime, $($p: Crosses<R>),+> Crosses<R> for ($($p,)+) {
            type Open<'a>
                = Parted<($(&'a Uniform<R, $p>,)+)>
            where
                Self: 'a;
            type OpenMut<'a>
                = Parted<($(&'a mut Uniform<R, $p>,)+)>
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
                Parted(($(Uniform::at_place($v),)+))
            }

            fn open_mut<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
            where
                Self: 'a,
            {
                let [$($v),+] = <R::Tuple as TupleKind<R>>::exclusive::<$n>(rt, place, at);
                Parted(($(Uniform::at_place_mut($v),)+))
            }

            fn open_ref<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::Open<'a>
            where
                Self: 'a,
            {
                let [$($v),+] = <R::Tuple as TupleKind<R>>::shared_ref::<$n>(rt, reference, at);
                Parted(($(Uniform::at_place($v),)+))
            }

            fn open_ref_mut<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
            where
                Self: 'a,
            {
                let [$($v),+] = <R::Tuple as TupleKind<R>>::exclusive_ref::<$n>(rt, reference, at);
                Parted(($(Uniform::at_place_mut($v),)+))
            }
        }
    )+};
}

tuples!(tuple_crossings);
