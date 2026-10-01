//! A script closure an extern calls (crossing §3-§4): a returning call lends
//! references and its result borrows only what it lent; an awaited call takes
//! its arguments by move and lends none of the handler's storage (K3).

use std::marker::PhantomData;

use crate::boundary::entry::Settled;
use crate::boundary::seal::Crossing;
use crate::boundary::uniform::Uniform;
use crate::context::{Awaited, Opens, Returning};
use crate::runtime::{Crosses, Opaque, RefKind, RefTarget, Referent, Runtime};

/// A callback of function type `Fn(A) -> U` at effect `E`, lent to the call
/// `'c` (limitations §3: it carries the call's lifetime, so no handler keeps
/// it). Only the entry makes one. `A` and `U` are loan forms: a returning call lends its arguments for `'l` and its
/// result may borrow what it lent, no longer (crossing §4); an awaited call
/// moves its arguments and its result borrows nothing.
pub struct Callback<'c, A, U, E, R: Runtime> {
    closure: R::Value,
    _at: PhantomData<(fn(&'c ()) -> &'c (), fn(A) -> U, E)>,
}

impl<A, U, E, R: Runtime> Callback<'_, A, U, E, R> {
    pub(crate) fn of_the_entry(closure: R::Value) -> Self {
        Callback { closure, _at: PhantomData }
    }
}

/// A type's form over the loan `'l` a returning call lends: the Rust type it is
/// at `'l`. A crossing type is itself; `Lent<T>` is `&'l T` and `LentMut<T>` is
/// `&'l mut T`. Its methods make and read values at the form's type, so it
/// is sealed: only this crate implements it.
pub trait LoanForm<R: Runtime>: Crossing {
    type At<'l>
    where
        Self: 'l;

    /// The value a returning call's argument crosses as: a reference value to
    /// a lent leaf, a given value for a crossing leaf.
    #[doc(hidden)]
    fn lend<'l>(value: Self::At<'l>, rt: &'l R, at: Settled<'l>) -> R::Value
    where
        Self: 'l;
    /// A returning call's result read back at its form: a reference value's
    /// target, or the value taken.
    #[doc(hidden)]
    fn read<'l>(v: R::Value, rt: &'l R, at: Settled<'l>) -> Self::At<'l>
    where
        Self: 'l;
}

/// `&'l T`: a part of what the returning call lent, shared.
pub struct Lent<T: ?Sized>(PhantomData<fn() -> *const T>);
/// `&'l mut T`: a part of what the returning call lent, exclusively.
pub struct LentMut<T: ?Sized>(PhantomData<fn() -> *const T>);

/// A value that crosses is its own loan form: given to the call, and taken
/// back from it.
macro_rules! given {
    ($([$($g:tt)*] $t:ty;)+) => {$(
        impl<R: Runtime, $($g)*> LoanForm<R> for $t
        where
            $t: Crosses<R>,
        {
            type At<'l>
                = $t
            where
                Self: 'l;

            fn lend<'l>(value: $t, rt: &'l R, at: Settled<'l>) -> R::Value
            where
                Self: 'l,
            {
                value.give(rt, at)
            }

            fn read<'l>(v: R::Value, rt: &'l R, at: Settled<'l>) -> $t
            where
                Self: 'l,
            {
                <$t as Crosses<R>>::take(rt, v, at)
            }
        }
    )+};
}

given! {
    [T: Opaque] T;
    [] String;
    [T] Option<T>;
    [T, E] Result<T, E>;
    [X] Uniform<R, X>;
}

macro_rules! given_primitives {
    ($($t:ty => $decl:expr),+ $(,)?) => {
        given! { $([] $t;)+ }
    };
}

primitives!(given_primitives);

macro_rules! given_tuples {
    ($($n:literal ($($p:ident $v:ident $i:tt),+))+) => {
        given! { $([$($p),+] ($($p,)+);)+ }
    };
}

tuples!(given_tuples);

impl<R: Runtime, T: ?Sized + Referent> LoanForm<R> for Lent<T> {
    type At<'l>
        = &'l T
    where
        Self: 'l;

    fn lend<'l>(value: &'l T, rt: &'l R, at: Settled<'l>) -> R::Value
    where
        Self: 'l,
    {
        <R::Ref as RefKind<R>>::reference(rt, RefTarget::Shared(value), at)
    }

    fn read<'l>(v: R::Value, rt: &'l R, at: Settled<'l>) -> &'l T
    where
        Self: 'l,
    {
        <R::Ref as RefKind<R>>::target(rt, v, at)
    }
}

impl<R: Runtime, T: ?Sized + Referent> LoanForm<R> for LentMut<T> {
    type At<'l>
        = &'l mut T
    where
        Self: 'l;

    fn lend<'l>(value: &'l mut T, rt: &'l R, at: Settled<'l>) -> R::Value
    where
        Self: 'l,
    {
        <R::Ref as RefKind<R>>::reference(rt, RefTarget::Exclusive(value), at)
    }

    fn read<'l>(v: R::Value, rt: &'l R, at: Settled<'l>) -> &'l mut T
    where
        Self: 'l,
    {
        <R::Ref as RefKind<R>>::target_mut(rt, v, at)
    }
}

/// A callback's parameters, a tuple of loan forms: lent together for `'l` to a
/// returning call, or, where every one is a crossing type, moved into an
/// awaited one.
pub trait CallArgs<R: Runtime>: Crossing {
    type Lent<'l>
    where
        Self: 'l;

    /// The argument run of a returning call, one value per parameter.
    #[doc(hidden)]
    fn lend_run<'l>(args: Self::Lent<'l>, rt: &'l R, at: Settled<'l>) -> impl ExactSizeIterator<Item = R::Value>
    where
        Self: 'l;
}

/// A callback's parameters an awaited call is given: each a crossing type,
/// moved into the call, which lends none of the handler's storage (K3).
pub trait MovedArgs<R: Runtime>: CallArgs<R> {
    type Moved;

    /// The argument run of an awaited call.
    #[doc(hidden)]
    fn move_run<'a>(args: Self::Moved, rt: &'a R, at: Settled<'a>) -> impl ExactSizeIterator<Item = R::Value>;
}

/// A callback's parameters of each arity: none, and each tuple of loan
/// forms.
macro_rules! forms {
    ($($n:literal ($($f:ident $v:ident $i:tt),*))+) => {$(
        impl<R: Runtime, $($f: LoanForm<R>),*> CallArgs<R> for ($($f,)*) {
            type Lent<'l>
                = ($($f::At<'l>,)*)
            where
                Self: 'l;

            #[allow(unused_variables)]
            fn lend_run<'l>(args: Self::Lent<'l>, rt: &'l R, at: Settled<'l>) -> impl ExactSizeIterator<Item = R::Value>
            where
                Self: 'l,
            {
                let ($($v,)*) = args;
                let run: [R::Value; $n] = [$($f::lend($v, rt, at)),*];
                run.into_iter()
            }
        }

        impl<R: Runtime, $($f: Crosses<R> + LoanForm<R>),*> MovedArgs<R> for ($($f,)*) {
            type Moved = ($($f,)*);

            #[allow(unused_variables)]
            fn move_run<'a>(args: Self::Moved, rt: &'a R, at: Settled<'a>) -> impl ExactSizeIterator<Item = R::Value> {
                let ($($v,)*) = args;
                let run: [R::Value; $n] = [$($v.give(rt, at)),*];
                run.into_iter()
            }
        }
    )+};
}

forms! { 0 () }
tuples!(forms);

impl<'c, A, U, E, R> Callback<'c, A, U, E, R>
where
    A: CallArgs<R>,
    U: LoanForm<R>,
    R: CallsBack,
{
    /// A returning call: the arguments lent for `'l`, the result borrowing at
    /// most what they lend.
    pub fn call<'l>(&'l self, cx: &'l Returning<'c, R>, args: A::Lent<'l>) -> U::At<'l>
    where
        A: 'l,
        U: 'l,
    {
        let runtime = cx.runtime();
        runtime.call_now::<A, U>(self.closure, args, Settled::of_the_entry())
    }
}

impl<'c, A, U, E, R> Callback<'c, A, U, E, R>
where
    A: MovedArgs<R>,
    U: Crosses<R>,
    R: CallsBack,
{
    /// An awaited call: the arguments moved, the result borrowing nothing
    /// (crossing §3-§4).
    pub fn call_await(&self, cx: &Awaited<'c, R>, args: A::Moved) -> impl Future<Output = U> + Send {
        let runtime = cx.runtime();
        runtime.call_later::<A, U>(self.closure, args, Settled::of_the_entry())
    }
}

/// Calling a script closure, returning or awaited (runtime contract §2): the
/// arguments are made at the runtime, and the callee's entry is opened at it.
pub trait CallsBack: Runtime {
    fn call_now<'l, A, U>(&'l self, closure: Self::Value, args: A::Lent<'l>, at: Settled<'l>) -> U::At<'l>
    where
        A: CallArgs<Self> + 'l,
        U: LoanForm<Self> + 'l;
    fn call_later<'a, A, U>(
        &'a self,
        closure: Self::Value,
        args: A::Moved,
        at: Settled<'a>,
    ) -> impl Future<Output = U> + Send
    where
        A: MovedArgs<Self>,
        U: Crosses<Self>;
}


