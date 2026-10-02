//! The language's types' crossings (crossing §2). What comes into a call is
//! opened as the Rust type its place writes (`Open`); what goes out is a
//! runtime value (`Give`); a function type's part is named by its key
//! (`Key`), and a Rust type written in a signature names its key (`Place`).
//! Each composes its kind's operations over its parts'. An extern type's are
//! the derive's, and its kind's here.

use acvus_utils::Interner;

use crate::boundary::entry::Settled;
use crate::boundary::uniform::{Abstract, Element, InPlace, Index, Never, Uniform};
use crate::decl::{DeclTy, DeclType, Nth};
use crate::runtime::{
    Give, Key, MutKey, Opaque, OpaqueKind, Open, OptionKind, Place, PrimitiveKind, RefKey, RefKind, Referent, Repr, ResultKind,
    Runtime, StringKind, TupleKind, ViewKind, Viewed,
};
use crate::{Mut, Ref};

/// A value kind's own: boxed, taken and viewed in place by its kind.
macro_rules! value_kind {
    ($([$($g:tt)*] $t:ty => $kind:ident: $kind_trait:ident $(<$p:ty>)?;)+) => {$(
        impl<R: Runtime, $($g)*> InPlace<R> for $t {
            fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
                <R::$kind as $kind_trait<R>>::give$(::<$p>)?(rt, self, at)
            }

            fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
                <R::$kind as $kind_trait<R>>::take$(::<$p>)?(rt, v, at)
            }

            fn shared<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> &'a Self
            where
                Self: 'a,
            {
                <R::$kind as $kind_trait<R>>::shared$(::<$p>)?(rt, place, at)
            }

            fn exclusive<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> &'a mut Self
            where
                Self: 'a,
            {
                <R::$kind as $kind_trait<R>>::exclusive$(::<$p>)?(rt, place, at)
            }
        }

        own_kind! { [$($g)*] $t; }
    )+};
}

/// What a type held in place by its kind is besides: viewed in place where a
/// reference names it; a box at it viewed as it; opened as itself; a
/// function type's part given boxed and opened as itself; and named by itself
/// in a signature.
macro_rules! own_kind {
    ($([$($g:tt)*] $t:ty;)+) => {$(
        impl<R: Runtime, $($g)*> Referent<R> for $t {
            fn view<'l>(rt: &'l R, reference: R::Value, at: Settled<'l>) -> &'l Self
            where
                Self: 'l,
            {
                <$t as InPlace<R>>::shared(rt, <R::Ref as RefKind<R>>::place(rt, reference, at), at)
            }

            fn view_mut<'l>(rt: &'l R, reference: R::Value, at: Settled<'l>) -> &'l mut Self
            where
                Self: 'l,
            {
                <$t as InPlace<R>>::exclusive(rt, <R::Ref as RefKind<R>>::place_mut(rt, reference, at), at)
            }
        }

        impl<R: Runtime, $($g)*> Viewed<R> for $t {
            type View = $t;
        }

        impl<R: Runtime, $($g)*> Repr<R> for $t {
            type Held = Uniform<R, $t>;
        }

        impl<R: Runtime, $($g)*> Index<R> for $t {
            type View = $t;
            type Key = $t;
        }

        impl<'l, R: Runtime, $($g)*> Open<'l, R> for $t {
            fn decl(names: &Interner) -> DeclTy {
                <$t as DeclType>::decl_ty(names)
            }

            fn open(rt: &'l R, v: R::Value, at: Settled<'l>) -> Self {
                <$t as InPlace<R>>::take(rt, v, at)
            }
        }

        impl<'l, R: Runtime, $($g)*> Key<'l, R> for $t {
            type Give = Uniform<R, $t>;
            type Open = $t;
        }

        impl<'l, R: Runtime, $($g)*> Place<'l, R> for $t {
            type Key = $t;
        }
    )+};
}

macro_rules! primitive_kinds {
    ($($t:ty => $decl:expr),+ $(,)?) => {
        value_kind! { $([] $t => Primitive: PrimitiveKind<$t>;)+ }
    };
}

primitives!(primitive_kinds);

value_kind! {
    [] String => String: StringKind;
}

/// An extern-defined value, held by the runtime as the Rust value itself,
/// viewed in place by the opaque kind.
impl<R: Runtime, T: Opaque + DeclType> InPlace<R> for T {
    fn give(self, rt: &R, at: Settled<'_>) -> R::Value {
        <R::Opaque as OpaqueKind<R>>::give(rt, self, at)
    }

    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self {
        <R::Opaque as OpaqueKind<R>>::take(rt, v, at)
    }

    fn shared<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> &'a Self
    where
        Self: 'a,
    {
        <R::Opaque as OpaqueKind<R>>::shared(rt, place, at)
    }

    fn exclusive<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> &'a mut Self
    where
        Self: 'a,
    {
        <R::Opaque as OpaqueKind<R>>::exclusive(rt, place, at)
    }
}

impl<R: Runtime, T: Opaque + DeclType> Referent<R> for T {
    fn view<'l>(rt: &'l R, reference: R::Value, at: Settled<'l>) -> &'l Self
    where
        Self: 'l,
    {
        <T as InPlace<R>>::shared(rt, <R::Ref as RefKind<R>>::place(rt, reference, at), at)
    }

    fn view_mut<'l>(rt: &'l R, reference: R::Value, at: Settled<'l>) -> &'l mut Self
    where
        Self: 'l,
    {
        <T as InPlace<R>>::exclusive(rt, <R::Ref as RefKind<R>>::place_mut(rt, reference, at), at)
    }
}

/// Text: a view of a range of a `String`.
impl<R: Runtime> Referent<R> for str {
    fn view<'l>(rt: &'l R, reference: R::Value, at: Settled<'l>) -> &'l Self
    where
        Self: 'l,
    {
        <R::View as ViewKind<R>>::str(rt, reference, at)
    }

    fn view_mut<'l>(rt: &'l R, reference: R::Value, at: Settled<'l>) -> &'l mut Self
    where
        Self: 'l,
    {
        <R::View as ViewKind<R>>::str_mut(rt, reference, at)
    }
}

impl<R: Runtime> Viewed<R> for str {
    type View = str;
}

/// A run: a view of a range of elements, seen as the slice of their boxes.
impl<R: Runtime, T: Element<R> + DeclType> Referent<R> for [T] {
    fn view<'l>(rt: &'l R, reference: R::Value, at: Settled<'l>) -> &'l Self
    where
        Self: 'l,
    {
        T::slice(<R::View as ViewKind<R>>::places(rt, reference, at))
    }

    fn view_mut<'l>(rt: &'l R, reference: R::Value, at: Settled<'l>) -> &'l mut Self
    where
        Self: 'l,
    {
        T::slice_mut(<R::View as ViewKind<R>>::places_mut(rt, reference, at))
    }
}

/// A run of a declared type's values is viewed as the slice of them as held.
impl<R: Runtime, K: Repr<R>> Viewed<R> for [K] {
    type View = [K::Held];
}

/// A box is the runtime's value already: it comes in and goes out as itself.
impl<'l, R: Runtime, X: DeclType> Open<'l, R> for Uniform<R, X> {
    fn decl(names: &Interner) -> DeclTy {
        X::decl_ty(names)
    }

    fn open(_rt: &'l R, v: R::Value, _at: Settled<'l>) -> Self {
        Uniform::of_the_crossing(v)
    }
}

impl<'l, R: Runtime, X: DeclType> Give<'l, R> for Uniform<R, X> {
    fn decl(names: &Interner) -> DeclTy {
        X::decl_ty(names)
    }

    fn give(self, _rt: &'l R, _at: Settled<'l>) -> R::Value {
        self.into_value()
    }
}

/// A box at an index with no Rust view is viewed as the box itself, and is a
/// function type's part as itself.
impl<R: Runtime, X: Abstract + DeclType> Referent<R> for Uniform<R, X> {
    fn view<'l>(rt: &'l R, reference: R::Value, at: Settled<'l>) -> &'l Self
    where
        Self: 'l,
    {
        Uniform::at_place(<R::Ref as RefKind<R>>::place(rt, reference, at))
    }

    fn view_mut<'l>(rt: &'l R, reference: R::Value, at: Settled<'l>) -> &'l mut Self
    where
        Self: 'l,
    {
        Uniform::at_place_mut(<R::Ref as RefKind<R>>::place_mut(rt, reference, at))
    }
}

impl<R: Runtime, X: Abstract + DeclType> Viewed<R> for Uniform<R, X> {
    type View = Uniform<R, X>;
}

impl<R: Runtime, X: Abstract + DeclType> Repr<R> for Uniform<R, X> {
    type Held = Uniform<R, X>;
}

impl<'l, R: Runtime, X: Abstract + DeclType> Key<'l, R> for Uniform<R, X> {
    type Give = Uniform<R, X>;
    type Open = Uniform<R, X>;
}

/// A box written in a signature names its index's key: a value type's own,
/// or the box itself at an index with no Rust view.
impl<'l, R: Runtime, X: Index<R>> Place<'l, R> for Uniform<R, X>
where
    X::Key: Key<'l, R>,
{
    type Key = X::Key;
}

impl<R: Runtime, const N: u32> Index<R> for Nth<N> {
    type View = Uniform<R, Nth<N>>;
    type Key = Uniform<R, Nth<N>>;
}

impl<R: Runtime> Index<R> for Never {
    type View = Uniform<R, Never>;
    type Key = Uniform<R, Never>;
}

/// The unit goes out as itself: a tuple of no parts.
impl<'l, R: Runtime> Give<'l, R> for () {
    fn decl(_: &Interner) -> DeclTy {
        DeclTy::Unit
    }

    fn give(self, rt: &'l R, at: Settled<'l>) -> R::Value {
        <R::Primitive as PrimitiveKind<R>>::give(rt, self, at)
    }
}

impl<'l, R: Runtime, F: Open<'l, R>> Open<'l, R> for Option<F> {
    fn decl(names: &Interner) -> DeclTy {
        DeclTy::Option(Box::new(F::decl(names)))
    }

    fn open(rt: &'l R, v: R::Value, at: Settled<'l>) -> Self {
        <R::Option as OptionKind<R>>::take(rt, v, at).map(|part| F::open(rt, part, at))
    }
}

impl<'l, R: Runtime, F: Give<'l, R>> Give<'l, R> for Option<F> {
    fn decl(names: &Interner) -> DeclTy {
        DeclTy::Option(Box::new(F::decl(names)))
    }

    fn give(self, rt: &'l R, at: Settled<'l>) -> R::Value {
        <R::Option as OptionKind<R>>::give(rt, self.map(|part| part.give(rt, at)), at)
    }
}

impl<'l, R: Runtime, K: Key<'l, R>> Key<'l, R> for Option<K> {
    type Give = Option<K::Give>;
    type Open = Option<K::Open>;
}

impl<'l, R: Runtime, P: Place<'l, R>> Place<'l, R> for Option<P>
where
    P::Key: Sized,
{
    type Key = Option<P::Key>;
}

impl<'l, R: Runtime, T: Open<'l, R>, E: Open<'l, R>> Open<'l, R> for Result<T, E> {
    fn decl(names: &Interner) -> DeclTy {
        DeclTy::Result(Box::new(T::decl(names)), Box::new(E::decl(names)))
    }

    fn open(rt: &'l R, v: R::Value, at: Settled<'l>) -> Self {
        <R::Result as ResultKind<R>>::take(rt, v, at).map(|ok| T::open(rt, ok, at)).map_err(|err| E::open(rt, err, at))
    }
}

impl<'l, R: Runtime, T: Give<'l, R>, E: Give<'l, R>> Give<'l, R> for Result<T, E> {
    fn decl(names: &Interner) -> DeclTy {
        DeclTy::Result(Box::new(T::decl(names)), Box::new(E::decl(names)))
    }

    fn give(self, rt: &'l R, at: Settled<'l>) -> R::Value {
        let parts = self.map(|ok| ok.give(rt, at)).map_err(|err| err.give(rt, at));
        <R::Result as ResultKind<R>>::give(rt, parts, at)
    }
}

impl<'l, R: Runtime, T: Key<'l, R>, E: Key<'l, R>> Key<'l, R> for Result<T, E> {
    type Give = Result<T::Give, E::Give>;
    type Open = Result<T::Open, E::Open>;
}

impl<'l, R: Runtime, T: Place<'l, R>, E: Place<'l, R>> Place<'l, R> for Result<T, E>
where
    T::Key: Sized,
    E::Key: Sized,
{
    type Key = Result<T::Key, E::Key>;
}

/// A tuple, made and taken by its kind over its parts.
macro_rules! tuple_forms {
    ($($n:literal ($($p:ident $v:ident $i:tt),+))+) => {$(
        impl<'l, R: Runtime, $($p: Open<'l, R>),+> Open<'l, R> for ($($p,)+) {
            fn decl(names: &Interner) -> DeclTy {
                DeclTy::Tuple(vec![$($p::decl(names)),+])
            }

            fn open(rt: &'l R, v: R::Value, at: Settled<'l>) -> Self {
                let [$($v),+] = <R::Tuple as TupleKind<R>>::take::<$n>(rt, v, at);
                ($($p::open(rt, $v, at),)+)
            }
        }

        impl<'l, R: Runtime, $($p: Give<'l, R>),+> Give<'l, R> for ($($p,)+) {
            fn decl(names: &Interner) -> DeclTy {
                DeclTy::Tuple(vec![$($p::decl(names)),+])
            }

            fn give(self, rt: &'l R, at: Settled<'l>) -> R::Value {
                let ($($v,)+) = self;
                <R::Tuple as TupleKind<R>>::give::<$n>(rt, [$($v.give(rt, at)),+], at)
            }
        }

        impl<'l, R: Runtime, $($p: Key<'l, R>),+> Key<'l, R> for ($($p,)+) {
            type Give = ($($p::Give,)+);
            type Open = ($($p::Open,)+);
        }

        impl<'l, R: Runtime, $($p: Place<'l, R>),+> Place<'l, R> for ($($p,)+)
        where
            $($p::Key: Sized),+
        {
            type Key = ($($p::Key,)+);
        }
    )+};
}

tuples!(tuple_forms);

/// A reference that comes in: a view of the place it names, at the loan it
/// lends for.
impl<'l, R: Runtime, X: ?Sized + Referent<R> + 'l> Open<'l, R> for &'l X {
    fn decl(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: false, target: Box::new(X::decl_ty(names)) }
    }

    fn open(rt: &'l R, v: R::Value, at: Settled<'l>) -> Self {
        X::view(rt, v, at)
    }
}

impl<'l, R: Runtime, X: ?Sized + Referent<R> + 'l> Open<'l, R> for &'l mut X {
    fn decl(names: &Interner) -> DeclTy {
        DeclTy::Ref { mutable: true, target: Box::new(X::decl_ty(names)) }
    }

    fn open(rt: &'l R, v: R::Value, at: Settled<'l>) -> Self {
        X::view_mut(rt, v, at)
    }
}

impl<'l, R: Runtime, X: ?Sized + Place<'l, R> + 'l> Place<'l, R> for &'l X {
    type Key = RefKey<X::Key>;
}

impl<'l, R: Runtime, X: ?Sized + Place<'l, R> + 'l> Place<'l, R> for &'l mut X {
    type Key = MutKey<X::Key>;
}

/// A run named in a signature is its elements' declared type's.
impl<'l, R: Runtime, E: Place<'l, R>> Place<'l, R> for [E]
where
    E::Key: Sized,
{
    type Key = [E::Key];
}

impl<'l, R: Runtime> Place<'l, R> for str {
    type Key = str;
}

impl<'l, R: Runtime, K: ?Sized + Viewed<R>> Key<'l, R> for RefKey<K>
where
    K::View: Referent<R> + 'l,
{
    type Give = Ref<'l, R, K::View>;
    type Open = &'l K::View;
}

impl<'l, R: Runtime, K: ?Sized + Viewed<R>> Key<'l, R> for MutKey<K>
where
    K::View: Referent<R> + 'l,
{
    type Give = Mut<'l, R, K::View>;
    type Open = &'l mut K::View;
}
