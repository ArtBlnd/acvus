//! A script closure an extern calls (crossing §3-§4): a returning call lends
//! references and its result borrows only what it lent; an awaited call takes
//! its arguments by move and lends none of the handler's storage (K3).

use std::marker::PhantomData;

use crate::boundary::entry::{Args, Entry, Settled};
use crate::boundary::seal::LoanForms;
use crate::boundary::uniform::Uniform;
use crate::context::{Awaited, Opens, Returning};
use crate::done::Done;
use crate::runtime::{Crosses, Mutability, RefTarget, Runtime, TupleBody, VariantBody};

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
/// that borrows what the call lent is written, at any depth. Its methods make
/// and read values at the form's type, so it is sealed: only this crate
/// implements it.
pub trait LoanForm: LoanForms {
    type At<'l>;

    /// How a returning call passes the value `lend` makes: lent where it is a
    /// reference value, by value where it is given.
    #[doc(hidden)]
    const LENT: Option<Mutability>;

    /// The value a returning call's argument crosses as: a reference value to
    /// a lent leaf, a given value for a crossing leaf.
    #[doc(hidden)]
    fn lend<'l, R: Runtime>(value: Self::At<'l>, rt: &'l R, at: Settled<'l>) -> R::Value;
    /// The value an awaited call's argument crosses as.
    #[doc(hidden)]
    fn give_moved<'a, R: Runtime>(value: Self::At<'static>, rt: &'a R, at: Settled<'a>) -> R::Value;
    /// A returning call's result read back at its form. The result is handed
    /// over, so the read owns it: a reference value is released once its
    /// target is read, and a value is taken.
    #[doc(hidden)]
    fn read<'l, R: Runtime>(v: R::Value, rt: &'l R, at: Settled<'l>) -> Self::At<'l>;
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

/// `Option`'s and `Result`'s tags: their variants' places in the order of
/// their names' text (compiler: types §2).
const NONE: u32 = 0;
const SOME: u32 = 1;
const ERR: u32 = 0;
const OK: u32 = 1;

/// A variant handed over by a read, taken apart: its tag and its payload,
/// each part held by a `Uniform` from here on.
fn variant_parts<R: Runtime>(v: R::Value, rt: &R, at: Settled<'_>) -> (u32, Option<Uniform<R, ()>>) {
    let (tag, payload) = rt.take_variant(v, at).into_payload();
    (tag, payload.map(Uniform::of_the_crossing))
}

impl<T: Crosses> LoanForm for T {
    type At<'l> = T;

    const LENT: Option<Mutability> = None;

    fn lend<'l, R: Runtime>(value: T, rt: &'l R, at: Settled<'l>) -> R::Value {
        rt.give(value, at)
    }

    fn give_moved<'a, R: Runtime>(value: T, rt: &'a R, at: Settled<'a>) -> R::Value {
        rt.give(value, at)
    }

    fn read<'l, R: Runtime>(v: R::Value, rt: &'l R, at: Settled<'l>) -> T {
        Uniform::<R, T>::of_the_crossing(v).take(rt, at)
    }
}

impl<T: ?Sized + 'static> LoanForm for Lent<T> {
    type At<'l> = &'l T;

    const LENT: Option<Mutability> = Some(Mutability::Shared);

    fn lend<'l, R: Runtime>(value: &'l T, rt: &'l R, at: Settled<'l>) -> R::Value {
        rt.reference(RefTarget::Shared(value), at)
    }

    fn give_moved<'a, R: Runtime>(value: &'static T, rt: &'a R, at: Settled<'a>) -> R::Value {
        rt.reference(RefTarget::Shared(value), at)
    }

    fn read<'l, R: Runtime>(v: R::Value, rt: &'l R, at: Settled<'l>) -> &'l T {
        let reference = Uniform::<R, ()>::of_the_crossing(v);
        let target = rt.target(v, at);
        drop(reference);
        target
    }
}

impl<T: ?Sized + 'static> LoanForm for LentMut<T> {
    type At<'l> = &'l mut T;

    const LENT: Option<Mutability> = Some(Mutability::Exclusive);

    fn lend<'l, R: Runtime>(value: &'l mut T, rt: &'l R, at: Settled<'l>) -> R::Value {
        rt.reference(RefTarget::Exclusive(value), at)
    }

    fn give_moved<'a, R: Runtime>(value: &'static mut T, rt: &'a R, at: Settled<'a>) -> R::Value {
        rt.reference(RefTarget::Exclusive(value), at)
    }

    fn read<'l, R: Runtime>(v: R::Value, rt: &'l R, at: Settled<'l>) -> &'l mut T {
        let reference = Uniform::<R, ()>::of_the_crossing(v);
        let target = rt.target_mut(v, at);
        drop(reference);
        target
    }
}

impl<F: LoanForm> LoanForm for Opt<F> {
    type At<'l> = Option<F::At<'l>>;

    const LENT: Option<Mutability> = None;

    fn lend<'l, R: Runtime>(value: Option<F::At<'l>>, rt: &'l R, at: Settled<'l>) -> R::Value {
        match value {
            None => rt.make_variant(NONE, None, at),
            Some(value) => {
                let payload = F::lend(value, rt, at);
                rt.make_variant(SOME, Some(payload), at)
            }
        }
    }

    fn give_moved<'a, R: Runtime>(value: Option<F::At<'static>>, rt: &'a R, at: Settled<'a>) -> R::Value {
        match value {
            None => rt.make_variant(NONE, None, at),
            Some(value) => {
                let payload = F::give_moved(value, rt, at);
                rt.make_variant(SOME, Some(payload), at)
            }
        }
    }

    fn read<'l, R: Runtime>(v: R::Value, rt: &'l R, at: Settled<'l>) -> Option<F::At<'l>> {
        match variant_parts(v, rt, at) {
            (NONE, None) => None,
            (SOME, Some(payload)) => Some(F::read(payload.into_value(), rt, at)),
            (tag, payload) => unreachable!("an Option has no variant {tag} with payload {}", payload.is_some()),
        }
    }
}

impl<F: LoanForm, G: LoanForm> LoanForm for Res<F, G> {
    type At<'l> = Result<F::At<'l>, G::At<'l>>;

    const LENT: Option<Mutability> = None;

    fn lend<'l, R: Runtime>(value: Result<F::At<'l>, G::At<'l>>, rt: &'l R, at: Settled<'l>) -> R::Value {
        let (tag, payload) = match value {
            Ok(value) => (OK, F::lend(value, rt, at)),
            Err(error) => (ERR, G::lend(error, rt, at)),
        };
        rt.make_variant(tag, Some(payload), at)
    }

    fn give_moved<'a, R: Runtime>(value: Result<F::At<'static>, G::At<'static>>, rt: &'a R, at: Settled<'a>) -> R::Value {
        let (tag, payload) = match value {
            Ok(value) => (OK, F::give_moved(value, rt, at)),
            Err(error) => (ERR, G::give_moved(error, rt, at)),
        };
        rt.make_variant(tag, Some(payload), at)
    }

    fn read<'l, R: Runtime>(v: R::Value, rt: &'l R, at: Settled<'l>) -> Result<F::At<'l>, G::At<'l>> {
        match variant_parts(v, rt, at) {
            (OK, Some(payload)) => Ok(F::read(payload.into_value(), rt, at)),
            (ERR, Some(payload)) => Err(G::read(payload.into_value(), rt, at)),
            (tag, payload) => unreachable!("a Result has no variant {tag} with payload {}", payload.is_some()),
        }
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
        impl<$($f: LoanForm),*> sealed::Sealed for ($($f,)*) {}

        impl<R: Runtime, $($f: LoanForm),*> CallArgs<R> for ($($f,)*) {
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

        impl<$($f: LoanForm),*> LoanForm for Tup<($($f,)*)> {
            type At<'l> = ($($f::At<'l>,)*);

            const LENT: Option<Mutability> = None;

            #[allow(unused_variables)]
            fn lend<'l, R: Runtime>(value: Self::At<'l>, rt: &'l R, at: Settled<'l>) -> R::Value {
                let ($($v,)*) = value;
                let parts: [R::Value; $n] = [$($f::lend($v, rt, at)),*];
                rt.make_tuple(parts.into_iter(), at)
            }

            #[allow(unused_variables)]
            fn give_moved<'a, R: Runtime>(value: Self::At<'static>, rt: &'a R, at: Settled<'a>) -> R::Value {
                let ($($v,)*) = value;
                let parts: [R::Value; $n] = [$($f::give_moved($v, rt, at)),*];
                rt.make_tuple(parts.into_iter(), at)
            }

            #[allow(unused_variables, unused_mut, clippy::unused_unit)]
            fn read<'l, R: Runtime>(v: R::Value, rt: &'l R, at: Settled<'l>) -> Self::At<'l> {
                let parts: Vec<Uniform<R, ()>> =
                    rt.take_tuple(v, at).into_parts().map(Uniform::of_the_crossing).collect();
                assert_eq!(parts.len(), $n, "a tuple read at its form has the form's arity");
                let mut parts = parts.into_iter();
                ($($f::read(parts.next().expect("the arity was checked").into_value(), rt, at),)*)
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
    U: LoanForm,
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
    U: Crosses,
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
        U: LoanForm;
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
        U: Crosses;
}

/// A Rust function an extern returns as a script function value (runtime
/// contract §2, Rust functions): each call of it is a call through an entry,
/// crossing its arguments at that call's settled types. Its glue is this
/// crate's: it opens the arguments by their modes `A`, calls the typed
/// function and finishes its `U`, so the glue agrees with the function type
/// `A -> U` it was made at.
pub struct RustFn<R: Runtime> {
    glue: Box<Glue<R>>,
}

/// A call through an entry, at the runtime `R`.
type Glue<R> = dyn for<'e> Fn(Entry<'e, R>) -> Done<R> + Send + Sync;

impl<R: Runtime> RustFn<R> {
    /// The function `f` over the arguments its modes `A` open, returning `U`.
    pub fn new<A, U>(f: impl for<'c> Fn(A::Opened<'c>) -> U + Send + Sync + 'static) -> Self
    where
        A: Args<R> + 'static,
        U: Crosses,
    {
        let glue = move |entry: Entry<'_, R>| entry.open::<A, ()>(|cx, args, ()| cx.finish(f(args)));
        RustFn { glue: Box::new(glue) }
    }

    /// A call of the function, through the entry its caller made.
    pub fn call(&self, entry: Entry<'_, R>) -> Done<R> {
        (self.glue)(entry)
    }
}

/// Making a script function value from a Rust function. `R` is the runtime the
/// glue runs at: the runtime itself, or the adapter over it (K15).
pub trait RustFns: Runtime {
    fn rust_fn<R>(&self, f: RustFn<R>) -> Self::Value
    where
        R: Runtime<Value = Self::Value>;
}
