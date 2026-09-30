//! A script closure an extern calls (crossing §3-§4): a returning call lends
//! references and its result borrows only what it lent; an awaited call takes
//! its arguments by move and lends none of the handler's storage (K3).

use std::marker::PhantomData;

use crate::boundary::entry::{Entry, Settled};
use crate::done::Done;
use crate::context::{Awaited, Opens, Returning};
use crate::runtime::{Crosses, Runtime};

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
/// `&'l mut T`; `Opt`, `Res` and `Tup` build forms from forms, so every result
/// that borrows what the call lent is written, at any depth.
pub trait LoanForm {
    type At<'l>;
}

/// `&'l T`: a part of what the returning call lent, shared.
pub struct Lent<T: ?Sized>(PhantomData<fn() -> *const T>);
/// `&'l mut T`: a part of what the returning call lent, exclusively.
pub struct LentMut<T: ?Sized>(PhantomData<fn() -> *const T>);
/// `Option` of a form.
pub struct Opt<F>(PhantomData<fn() -> F>);
/// `Result` of two forms.
pub struct Res<F, G>(PhantomData<fn() -> (F, G)>);
/// A tuple of forms, `Tup<(F, G)>`.
pub struct Tup<Fs>(PhantomData<fn() -> Fs>);

impl<T: Crosses> LoanForm for T {
    type At<'l> = T;
}

impl<T: ?Sized + 'static> LoanForm for Lent<T> {
    type At<'l> = &'l T;
}

impl<T: ?Sized + 'static> LoanForm for LentMut<T> {
    type At<'l> = &'l mut T;
}

impl<F: LoanForm> LoanForm for Opt<F> {
    type At<'l> = Option<F::At<'l>>;
}

impl<F: LoanForm, G: LoanForm> LoanForm for Res<F, G> {
    type At<'l> = Result<F::At<'l>, G::At<'l>>;
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
}

/// Tuples of loan forms up to twelve, the arity the crossing's tuples stop at.
macro_rules! forms {
    ($(($($f:ident),*))+) => {$(
        impl<$($f: LoanForm),*> sealed::Sealed for ($($f,)*) {}

        impl<R: Runtime, $($f: LoanForm),*> CallArgs<R> for ($($f,)*) {
            type Lent<'l> = ($($f::At<'l>,)*);
            type Moved = ($($f::At<'static>,)*);
        }

        impl<$($f: LoanForm),*> LoanForm for Tup<($($f,)*)> {
            type At<'l> = ($($f::At<'l>,)*);
        }
    )+};
}

forms! {
    ()
    (A)
    (A, B)
    (A, B, C)
    (A, B, C, D)
    (A, B, C, D, E)
    (A, B, C, D, E, F)
    (A, B, C, D, E, F, G)
    (A, B, C, D, E, F, G, H)
    (A, B, C, D, E, F, G, H, I)
    (A, B, C, D, E, F, G, H, I, J)
    (A, B, C, D, E, F, G, H, I, J, K)
    (A, B, C, D, E, F, G, H, I, J, K, L)
}

impl<'c, 'a, A, U, E, R> Callback<'c, 'a, A, U, E, R>
where
    A: CallArgs<R>,
    U: LoanForm,
    R: CallsBack,
{
    /// A returning call: the arguments lent for `'l`, the result borrowing at
    /// most what they lend.
    pub fn call<'l>(&'l self, cx: &'l Returning<'c, R>, args: A::Lent<'l>) -> U::At<'l> {
        cx.runtime().call_now::<R, A, U>(self.closure, args, Settled::of_the_entry())
    }
}

impl<'c, 'a, A, U, E, R> Callback<'c, 'a, A, U, E, R>
where
    A: CallArgs<R>,
    U: Crosses,
    R: CallsBack,
{
    /// An awaited call: the arguments moved, the result borrowing nothing
    /// (crossing §3-§4).
    pub fn call_await(&self, cx: &Awaited<'c, R>, args: A::Moved) -> impl Future<Output = U> + Send {
        cx.runtime().call_later::<R, A, U>(self.closure, args, Settled::of_the_entry())
    }
}

/// Calling a script closure, returning or awaited (runtime contract §2). The
/// arguments are typed at the runtime the call runs at, `R`: the runtime
/// itself, or the adapter over it (K15), whose values are the runtime's.
pub trait CallsBack: Runtime {
    fn call_now<'l, R, A, U>(&self, closure: Self::Value, args: A::Lent<'l>, at: Settled<'l>) -> U::At<'l>
    where
        R: Runtime<Value = Self::Value>,
        A: CallArgs<R>,
        U: LoanForm;
    fn call_later<R, A, U>(&self, closure: Self::Value, args: A::Moved, at: Settled<'_>) -> impl Future<Output = U> + Send
    where
        R: Runtime<Value = Self::Value>,
        A: CallArgs<R>,
        U: Crosses;
}

/// A Rust function an extern returns as a script function value (runtime
/// contract §2, Rust functions): each call of it is a call through an entry,
/// crossing its arguments at that call's settled types. Only the boundary makes
/// one, from a handler's glue.
pub struct RustFn<R: Runtime> {
    call: fn(Entry<'_, R>) -> Done<R>,
}

/// Making a script function value from a Rust function. `R` is the runtime the
/// glue runs at: the runtime itself, or the adapter over it (K15).
pub trait RustFns: Runtime {
    fn rust_fn<R>(&self, f: RustFn<R>) -> Self::Value
    where
        R: Runtime<Value = Self::Value>;
}
