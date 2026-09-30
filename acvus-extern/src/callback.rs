//! A script closure an extern calls (crossing §3-§4): a returning call lends
//! references and its result borrows only what it lent; an awaited call takes
//! its arguments by move and lends none of the handler's storage (K3).

use std::marker::PhantomData;

use crate::boundary::entry::Settled;
use crate::boundary::seal::LoanForms;
use crate::boundary::uniform::Uniform;
use crate::context::{Awaited, Opens, Returning};
use crate::runtime::{Crosses, Mutability, RefKind, RefTarget, Referent, Runtime};

/// A callback of function type `Fn(A) -> U` at effect `E`, lent to the call
/// `'c` (limitations §3: it carries the call's lifetime, so no handler keeps
/// it); `'a` is the loans its captures hold. Only the entry makes one. `A` and
/// `U` are loan forms: a returning call lends its arguments for `'l` and its
/// result may borrow what it lent, no longer (crossing §4); an awaited call
/// moves its arguments and its result borrows nothing.
pub struct Callback<'c, 'a, A, U, E, R: Runtime> {
    closure: R::Value,
    _at: PhantomData<(fn(&'c ()) -> &'c (), &'a (), fn(A) -> U, E)>,
}

impl<A, U, E, R: Runtime> Callback<'_, '_, A, U, E, R> {
    pub(crate) fn of_the_entry(closure: R::Value) -> Self {
        Callback { closure, _at: PhantomData }
    }
}

/// A type's form over the loan `'l` a returning call lends: the Rust type it is
/// at `'l`. A crossing type is itself; `Lent<T>` is `&'l T` and `LentMut<T>` is
/// `&'l mut T`. Its methods make and read values at the form's type, so it
/// is sealed: only this crate implements it.
pub trait LoanForm<R: Runtime>: LoanForms {
    type At<'l>;

    /// How a returning call passes the value `lend` makes: lent where it is a
    /// reference value, by value where it is given.
    #[doc(hidden)]
    const LENT: Option<Mutability>;

    /// The value a returning call's argument, or a Rust function's result,
    /// crosses as: a reference value to a lent leaf, a given value for a
    /// crossing leaf.
    #[doc(hidden)]
    fn lend<'l>(value: Self::At<'l>, rt: &'l R, at: Settled<'l>) -> R::Value;
    /// The value an awaited call's argument crosses as.
    #[doc(hidden)]
    fn give_moved<'a>(value: Self::At<'static>, rt: &'a R, at: Settled<'a>) -> R::Value;
    /// A returning call's result read back at its form. The result is handed
    /// over, so the read owns it: a reference value is released once its
    /// target is read, and a value is taken.
    #[doc(hidden)]
    fn read<'l>(v: R::Value, rt: &'l R, at: Settled<'l>) -> Self::At<'l>;
}

/// `&'l T`: a part of what the returning call lent, shared.
pub struct Lent<T: ?Sized>(PhantomData<fn() -> *const T>);
/// `&'l mut T`: a part of what the returning call lent, exclusively.
pub struct LentMut<T: ?Sized>(PhantomData<fn() -> *const T>);

impl<R: Runtime, T: Crosses<R>> LoanForm<R> for T {
    type At<'l> = T;

    const LENT: Option<Mutability> = None;

    fn lend<'l>(value: T, rt: &'l R, at: Settled<'l>) -> R::Value {
        value.give(rt, at)
    }

    fn give_moved<'a>(value: T, rt: &'a R, at: Settled<'a>) -> R::Value {
        value.give(rt, at)
    }

    fn read<'l>(v: R::Value, rt: &'l R, at: Settled<'l>) -> T {
        T::take(rt, v, at)
    }
}

impl<R: Runtime, T: ?Sized + Referent + 'static> LoanForm<R> for Lent<T> {
    type At<'l> = &'l T;

    const LENT: Option<Mutability> = Some(Mutability::Shared);

    fn lend<'l>(value: &'l T, rt: &'l R, at: Settled<'l>) -> R::Value {
        <R::Ref as RefKind<R>>::reference(rt, RefTarget::Shared(value), at)
    }

    fn give_moved<'a>(value: &'static T, rt: &'a R, at: Settled<'a>) -> R::Value {
        <R::Ref as RefKind<R>>::reference(rt, RefTarget::Shared(value), at)
    }

    fn read<'l>(v: R::Value, rt: &'l R, at: Settled<'l>) -> &'l T {
        let reference = Uniform::<R, ()>::of_the_crossing(v);
        let target = <R::Ref as RefKind<R>>::target(rt, v, at);
        drop(reference);
        target
    }
}

impl<R: Runtime, T: ?Sized + Referent + 'static> LoanForm<R> for LentMut<T> {
    type At<'l> = &'l mut T;

    const LENT: Option<Mutability> = Some(Mutability::Exclusive);

    fn lend<'l>(value: &'l mut T, rt: &'l R, at: Settled<'l>) -> R::Value {
        <R::Ref as RefKind<R>>::reference(rt, RefTarget::Exclusive(value), at)
    }

    fn give_moved<'a>(value: &'static mut T, rt: &'a R, at: Settled<'a>) -> R::Value {
        <R::Ref as RefKind<R>>::reference(rt, RefTarget::Exclusive(value), at)
    }

    fn read<'l>(v: R::Value, rt: &'l R, at: Settled<'l>) -> &'l mut T {
        let reference = Uniform::<R, ()>::of_the_crossing(v);
        let target = <R::Ref as RefKind<R>>::target_mut(rt, v, at);
        drop(reference);
        target
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A callback's parameters, a tuple of loan forms: lent together for `'l` to a
/// returning call, or, where every one is a crossing type, moved into an
/// awaited one.
pub trait CallArgs<R: Runtime>: sealed::Sealed {
    type Lent<'l>;
    type Moved;

    /// The argument run of a returning call, one value per parameter.
    #[doc(hidden)]
    fn lend_run<'l>(args: Self::Lent<'l>, rt: &'l R, at: Settled<'l>) -> impl ExactSizeIterator<Item = R::Value>;
    /// How a returning call passes each value of its run.
    #[doc(hidden)]
    fn lent() -> impl ExactSizeIterator<Item = Option<Mutability>>;
    /// The argument run of an awaited call.
    #[doc(hidden)]
    fn move_run<'a>(args: Self::Moved, rt: &'a R, at: Settled<'a>) -> impl ExactSizeIterator<Item = R::Value>;
}

/// Tuples of loan forms up to twelve, the arity the crossing's tuples stop at.
macro_rules! forms {
    ($($n:literal ($($f:ident $v:ident),*))+) => {$(
        impl<$($f),*> sealed::Sealed for ($($f,)*) {}

        impl<R: Runtime, $($f: LoanForm<R>),*> CallArgs<R> for ($($f,)*) {
            type Lent<'l> = ($($f::At<'l>,)*);
            type Moved = ($($f::At<'static>,)*);

            fn lent() -> impl ExactSizeIterator<Item = Option<Mutability>> {
                let lent: [Option<Mutability>; $n] = [$($f::LENT),*];
                lent.into_iter()
            }

            #[allow(unused_variables)]
            fn lend_run<'l>(args: Self::Lent<'l>, rt: &'l R, at: Settled<'l>) -> impl ExactSizeIterator<Item = R::Value> {
                let ($($v,)*) = args;
                let run: [R::Value; $n] = [$($f::lend($v, rt, at)),*];
                run.into_iter()
            }

            #[allow(unused_variables)]
            fn move_run<'a>(args: Self::Moved, rt: &'a R, at: Settled<'a>) -> impl ExactSizeIterator<Item = R::Value> {
                let ($($v,)*) = args;
                let run: [R::Value; $n] = [$($f::give_moved($v, rt, at)),*];
                run.into_iter()
            }
        }
    )+};
}

forms! {
    0 ()
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

impl<'c, 'a, A, U, E, R> Callback<'c, 'a, A, U, E, R>
where
    A: CallArgs<R>,
    U: LoanForm<R>,
    R: CallsBack,
{
    /// A returning call: the arguments lent for `'l`, the result borrowing at
    /// most what they lend.
    pub fn call<'l>(&'l self, cx: &'l Returning<'c, R>, args: A::Lent<'l>) -> U::At<'l> {
        let runtime = cx.runtime();
        runtime.call_now::<R, A, U>(runtime, self.closure, args, Settled::of_the_entry())
    }
}

impl<'c, 'a, A, U, E, R> Callback<'c, 'a, A, U, E, R>
where
    A: CallArgs<R>,
    U: Crosses<R>,
    R: CallsBack,
{
    /// An awaited call: the arguments moved, the result borrowing nothing
    /// (crossing §3-§4).
    pub fn call_await(&self, cx: &Awaited<'c, R>, args: A::Moved) -> impl Future<Output = U> + Send {
        let runtime = cx.runtime();
        runtime.call_later::<R, A, U>(runtime, self.closure, args, Settled::of_the_entry())
    }
}

/// Calling a script closure, returning or awaited (runtime contract §2). `rt`
/// is the runtime the call runs at, the runtime itself or the adapter over it
/// (K15), whose values are the runtime's: the arguments are made at it, and
/// the callee's entry is opened at it.
pub trait CallsBack: Runtime {
    fn call_now<'l, R, A, U>(&self, rt: &'l R, closure: Self::Value, args: A::Lent<'l>, at: Settled<'l>) -> U::At<'l>
    where
        R: Runtime<Value = Self::Value>,
        A: CallArgs<R>,
        U: LoanForm<R>;
    fn call_later<'a, R, A, U>(
        &self,
        rt: &'a R,
        closure: Self::Value,
        args: A::Moved,
        at: Settled<'a>,
    ) -> impl Future<Output = U> + Send
    where
        R: Runtime<Value = Self::Value>,
        A: CallArgs<R>,
        U: Crosses<R>;
}


