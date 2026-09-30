//! A result that borrows (crossing §4): a value of a transparent structure
//! (`Option`, `Result`, a tuple) whose every leaf is a reference to a part of
//! what the call lent, made while the guard that lends it still does. The
//! form names the result's shape at the guard's borrow `'s`, of a target
//! that outlives any borrow of it (a `'static` referent), so a result is
//! finished only from the guard (`Returning::finish_borrowed`): nothing else
//! has the type the form asks for (K16).

use crate::boundary::entry::Settled;
use crate::callback::{Lent, LentMut};
use crate::runtime::{OptionKind, RefKind, RefTarget, Referent, ResultKind, Runtime, TupleKind};

mod sealed {
    pub trait Sealed {}
}

/// The shape of a borrowed result at the guard's borrow `'s`, and how its
/// runtime value is made: each reference a reference value, each structure
/// by its kind.
pub trait BorrowForm<R: Runtime>: sealed::Sealed {
    type At<'s>;

    #[doc(hidden)]
    fn give<'s>(v: Self::At<'s>, rt: &'s R, at: Settled<'s>) -> R::Value;
}

/// A result type's form, read off the type: a reference is `Lent` or
/// `LentMut`, and a structure of borrowed results is the structure of their
/// forms.
pub trait Borrows {
    type Form;
}

impl<P: ?Sized> sealed::Sealed for Lent<P> {}
impl<P: ?Sized> sealed::Sealed for LentMut<P> {}
impl<F> sealed::Sealed for Option<F> {}
impl<F, G> sealed::Sealed for Result<F, G> {}

impl<R: Runtime, P: ?Sized + Referent + 'static> BorrowForm<R> for Lent<P> {
    type At<'s>
        = &'s P;

    fn give<'s>(v: &'s P, rt: &'s R, at: Settled<'s>) -> R::Value
    {
        <R::Ref as RefKind<R>>::reference(rt, RefTarget::Shared(v), at)
    }
}

impl<R: Runtime, P: ?Sized + Referent + 'static> BorrowForm<R> for LentMut<P> {
    type At<'s>
        = &'s mut P;

    fn give<'s>(v: &'s mut P, rt: &'s R, at: Settled<'s>) -> R::Value
    {
        <R::Ref as RefKind<R>>::reference(rt, RefTarget::Exclusive(v), at)
    }
}

impl<R: Runtime, F: BorrowForm<R>> BorrowForm<R> for Option<F> {
    type At<'s>
        = Option<F::At<'s>>;

    fn give<'s>(v: Self::At<'s>, rt: &'s R, at: Settled<'s>) -> R::Value
    {
        <R::Option as OptionKind<R>>::give(rt, v.map(|part| F::give(part, rt, at)), at)
    }
}

impl<R: Runtime, F: BorrowForm<R>, G: BorrowForm<R>> BorrowForm<R> for Result<F, G> {
    type At<'s>
        = Result<F::At<'s>, G::At<'s>>;

    fn give<'s>(v: Self::At<'s>, rt: &'s R, at: Settled<'s>) -> R::Value
    {
        let parts = v.map(|part| F::give(part, rt, at)).map_err(|part| G::give(part, rt, at));
        <R::Result as ResultKind<R>>::give(rt, parts, at)
    }
}

impl<P: ?Sized> Borrows for &P {
    type Form = Lent<P>;
}

impl<P: ?Sized> Borrows for &mut P {
    type Form = LentMut<P>;
}

impl<B: Borrows> Borrows for Option<B> {
    type Form = Option<B::Form>;
}

impl<B: Borrows, C: Borrows> Borrows for Result<B, C> {
    type Form = Result<B::Form, C::Form>;
}

/// Tuples up to twelve parts, the arity the crossing's tuples stop at.
macro_rules! tuples {
    ($($n:literal ($($f:ident $v:ident),+))+) => {$(
        impl<$($f),+> sealed::Sealed for ($($f,)+) {}

        impl<R: Runtime, $($f: BorrowForm<R>),+> BorrowForm<R> for ($($f,)+) {
            type At<'s>
                = ($($f::At<'s>,)+);

            fn give<'s>(v: Self::At<'s>, rt: &'s R, at: Settled<'s>) -> R::Value
            {
                let ($($v,)+) = v;
                <R::Tuple as TupleKind<R>>::give::<$n>(rt, [$($f::give($v, rt, at)),+], at)
            }
        }

        impl<$($f: Borrows),+> Borrows for ($($f,)+) {
            type Form = ($($f::Form,)+);
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
