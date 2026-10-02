//! A function value an extern calls (crossing §3-§5): returning or awaited,
//! it lends as Rust lends, and its result borrows only what it lent. An
//! awaited call's run is held by its call future, so nothing it was lent is
//! read past the future.
//!
//! A function value is held as any value is, in the owning holder indexed by
//! its function type, `Uniform<R, Callable<'f, A, U, E>>`: taken by value it
//! is the handler's, released by its drop and kept only where a result holds
//! it; lent, it is `&Uniform<R, Callable<..>>`. Either body calls one through
//! its executor (macros §1).

use std::marker::PhantomData;

use crate::boundary::entry::Settled;
use crate::boundary::uniform::{Abstract, Index, Uniform};
use crate::decl::DeclType;
use crate::executor::Executor;
use crate::runtime::{Give, Key, Open, Runtime};

/// The function type `Fn(A) -> U` at effect `E`, whose values hold the loans
/// `'f` their captures hold (compiler: ownership §5): an index only, so no
/// Rust value of it exists. `A` is a tuple of the parameters' keys and `U`
/// the result's (`runtime::Key`). `E` is one of the declaration's effect
/// variables (types §2).
pub struct Callable<'f, A, U, E> {
    _never: PhantomData<(&'f (), fn(A) -> U, E)>,
}

/// A function value as a part of a function type: its form at the call's
/// loan `'l` holds what that call lent (compiler: types §5). A function value
/// a call returns is not used after what the call lent dies:
///
/// ```compile_fail,E0597
/// use acvus_extern::Uniform;
/// use acvus_extern::callback::{Callable, CallsBack, Func};
/// use acvus_extern::decl::EffectVar;
/// use acvus_extern::executor::Executor;
/// use acvus_extern::runtime::Ref;
///
/// fn made<'f, R: CallsBack, E: EffectVar>(
///     cx: &Executor<'_, R>,
///     make: &Uniform<R, Callable<'f, (Ref<i64>,), Func<(i64,), i64, E>, E>>,
/// ) -> i64 {
///     let made = {
///         let local = 1i64;
///         make.call(cx, (&local,))
///     };
///     made.call(cx, (2,))
/// }
/// ```
pub struct Func<A, U, E> {
    _never: PhantomData<fn(A) -> (U, E)>,
}

impl<'l, R: Runtime, A, U, E> Key<'l, R> for Func<A, U, E>
where
    Func<A, U, E>: DeclType,
    Callable<'l, A, U, E>: DeclType,
{
    type Give = Uniform<R, Callable<'l, A, U, E>>;
    type Open = Uniform<R, Callable<'l, A, U, E>>;
}

/// A function value has no Rust view: its box is viewed as itself.
impl<A, U, E> Abstract for Callable<'_, A, U, E> {}

impl<'f, R: Runtime, A, U, E> Index<R> for Callable<'f, A, U, E> {
    type View = Uniform<R, Callable<'f, A, U, E>>;
    type Key = Uniform<R, Callable<'f, A, U, E>>;
}

/// A call's parameters, a tuple of keys: each given as a runtime value at
/// the loan `'l` the call lends for.
pub trait Keys<'l, R: Runtime> {
    type Give;

    /// The argument run, one value per parameter.
    #[doc(hidden)]
    fn give_run(args: Self::Give, rt: &'l R, at: Settled<'l>) -> impl ExactSizeIterator<Item = R::Value>;
}

/// A call's parameters of each arity: none, and each tuple of keys.
macro_rules! keys {
    ($($n:literal ($($p:ident $v:ident $i:tt),*))+) => {$(
        impl<'l, R: Runtime, $($p: Key<'l, R>),*> Keys<'l, R> for ($($p,)*) {
            type Give = ($($p::Give,)*);

            #[allow(unused_variables)]
            fn give_run(args: Self::Give, rt: &'l R, at: Settled<'l>) -> impl ExactSizeIterator<Item = R::Value> {
                let ($($v,)*) = args;
                let run: [R::Value; $n] = [$(Give::give($v, rt, at)),*];
                run.into_iter()
            }
        }
    )+};
}

keys! { 0 () }
tuples!(keys);

impl<'f, A, U, E, R: CallsBack> Uniform<R, Callable<'f, A, U, E>> {
    /// A returning call, its arguments lent for `'l` and its result
    /// borrowing only what it lent (crossing §4).
    pub fn call<'l>(&'l self, cx: &'l Executor<'_, R>, args: <A as Keys<'l, R>>::Give) -> <U as Key<'l, R>>::Open
    where
        A: Keys<'l, R>,
        U: Key<'l, R>,
    {
        let (runtime, at) = (cx.runtime(), Settled::of_the_entry());
        let result = runtime.call_now(self.value(), A::give_run(args, runtime, at), at);
        <U::Open as Open<'l, R>>::open(runtime, result, at)
    }

    /// An awaited call, its arguments lent for the call future `'a`, which
    /// holds the call's run (crossing §3, §5).
    pub fn call_await<'a>(&'a self, cx: &'a Executor<'_, R>, args: <A as Keys<'a, R>>::Give) -> impl Future<Output = <U as Key<'a, R>>::Open> + Send + 'a
    where
        A: Keys<'a, R>,
        U: Key<'a, R>,
        U::Open: Send,
    {
        let (runtime, at) = (cx.runtime(), Settled::of_the_entry());
        let result = runtime.call_later(self.value(), A::give_run(args, runtime, at), at);
        async move { <U::Open as Open<'a, R>>::open(runtime, result.await, at) }
    }
}

/// Calling a function value, returning or awaited (runtime contract §2): the
/// runtime calls it with a run of values the caller made, and gives back the
/// value of its result, which the caller reads at its type.
pub trait CallsBack: Runtime {
    /// A returning call of `function` on the argument run `run`: the value
    /// of its result. A function value that may suspend is driven to its
    /// end here. A reference value in the run is valid for `'l` only, and
    /// a given value is the runtime's from the call on.
    fn call_now<'l>(&'l self, function: Self::Value, run: impl IntoIterator<Item = Self::Value>, at: Settled<'l>) -> Self::Value;

    /// An owned function value of the function `function` names, released
    /// by its own holder's drop: what an instance keeps of the impl its site
    /// chose (runtime contract §2, Impls).
    fn own<'c>(&'c self, function: &'c Self::Value, at: Settled<'c>) -> Self::Value;

    /// An awaited call of `function` on the argument run `run`: the value of
    /// its result. The call's run is held by the future and ends when it ends
    /// or is dropped (crossing §5); a reference value in the run is valid for
    /// `'a` only, and a given value is the runtime's from the call on.
    fn call_later<'a>(
        &'a self,
        function: Self::Value,
        run: impl IntoIterator<Item = Self::Value>,
        at: Settled<'a>,
    ) -> impl Future<Output = Self::Value> + Send + 'a;
}
