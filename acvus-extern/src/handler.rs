//! Typed handlers, ready for a runtime to call (RFC-0059).
//!
//! A declaration's Rust body is reached through a closure the proc macro
//! writes, wrapped in a `Glue` whose type parameters say how each parameter
//! comes out of the call's argument run and how the result is written back.
//! Every fact the ABI needs is a constant of those types, summed once in the
//! `Handler` impl and read back through `Handler::width`.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use acvus_mir::ty::{PolyTy, Task};
use futures::future::BoxFuture;

use crate::obj::{Cross, CrossSpecialized, Form, One, OneValue};
use crate::runtime::Runtime;
use crate::slice::Elements;

/// Which crossing a parameter or a result takes (RFC-0040).
pub struct Uniform;
/// The crossing of a `Monomorphize` member instance.
pub struct Specialized;

/// How one Rust parameter takes its argument out of a call's argument run.
/// The mode — by value, by shared reference, by exclusive reference — is
/// written in Rust and read by the macro (RFC-0015); the width is the
/// type's.
pub trait Arg<'a, Rt>: Sized
where
    Rt: Runtime,
{
    /// What the closure's parameter is.
    type Out;
    /// The run this parameter takes out of the call's argument run. A bound
    /// that admits only a parameter which survives the caller suspending
    /// says `Form = One`: the `Pair` a slice is borrows the caller's frame
    /// (RFC-0047 §3).
    type Form: Form;

    /// How many of the run's values this parameter consumes.
    const WIDTH: usize = <Self::Form as Form>::WIDTH;

    /// # Safety
    /// `run` is this parameter's own `WIDTH` values of a call's argument
    /// run, and any storage a reference it yields names is live and unmoved
    /// for `'a` — exclusively so for an exclusive reference (RFC-0018).
    unsafe fn take(rt: &'a Rt, run: &'a [Rt::Value]) -> Self::Out;
}

/// A parameter taken by value: the crossing builds the Rust value.
pub struct ByValue<T, C = Uniform>(PhantomData<fn() -> (T, C)>);
/// A parameter taken by shared reference: the body reads the storage the
/// caller lent (RFC-0018).
pub struct ByRef<T, C = Uniform>(PhantomData<fn() -> (T, C)>);
/// As `ByRef`, exclusively.
pub struct ByRefMut<T, C = Uniform>(PhantomData<fn() -> (T, C)>);

impl<'a, T, Rt> Arg<'a, Rt> for ByValue<T, Uniform>
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    type Out = T;
    type Form = <T as Cross<Rt>>::Form;

    unsafe fn take(rt: &'a Rt, run: &'a [Rt::Value]) -> T {
        // SAFETY: the caller's contract, which is `Cross::from_run`'s.
        unsafe { <T as Cross<Rt>>::from_run(rt, run) }
    }
}

impl<'a, T, Rt> Arg<'a, Rt> for ByValue<T, Specialized>
where
    T: CrossSpecialized<Rt>,
    Rt: Runtime,
{
    type Out = T;
    type Form = One;

    unsafe fn take(rt: &'a Rt, run: &'a [Rt::Value]) -> T {
        // SAFETY: as the uniform impl's.
        unsafe { <T as CrossSpecialized<Rt>>::from_run(rt, run) }
    }
}

impl<'a, T, Rt> Arg<'a, Rt> for ByRef<T, Uniform>
where
    T: Borrowable<Rt>,
    Rt: Runtime,
{
    type Out = &'a T;
    type Form = One;

    unsafe fn take(rt: &'a Rt, run: &'a [Rt::Value]) -> &'a T {
        // SAFETY: the caller's contract: a live storage of `T` (RFC-0018).
        unsafe { <T as OneValue<Rt>>::deref(rt, &run[0]) }
    }
}

impl<'a, T, Rt> Arg<'a, Rt> for ByRefMut<T, Uniform>
where
    T: Borrowable<Rt>,
    Rt: Runtime,
{
    type Out = &'a mut T;
    type Form = One;

    unsafe fn take(rt: &'a Rt, run: &'a [Rt::Value]) -> &'a mut T {
        // SAFETY: the caller's contract: a live storage of `T`, exclusively
        // named (RFC-0018).
        unsafe { <T as OneValue<Rt>>::deref_mut(rt, &run[0]) }
    }
}

impl<'a, T, Rt> Arg<'a, Rt> for ByRef<T, Specialized>
where
    T: CrossSpecialized<Rt>,
    Rt: Runtime,
{
    type Out = &'a T;
    type Form = One;

    unsafe fn take(rt: &'a Rt, run: &'a [Rt::Value]) -> &'a T {
        // SAFETY: as the uniform impl's, at the specialized representation.
        unsafe { <T as CrossSpecialized<Rt>>::deref(rt, &run[0]) }
    }
}

impl<'a, T, Rt> Arg<'a, Rt> for ByRefMut<T, Specialized>
where
    T: CrossSpecialized<Rt>,
    Rt: Runtime,
{
    type Out = &'a mut T;
    type Form = One;

    unsafe fn take(rt: &'a Rt, run: &'a [Rt::Value]) -> &'a mut T {
        // SAFETY: as the uniform impl's, at the specialized representation.
        unsafe { <T as CrossSpecialized<Rt>>::deref_mut(rt, &run[0]) }
    }
}

/// A type a parameter may take by reference: one whose values are places the
/// language names. An `Option` is not — `None` is one value and `Some(v)` is
/// `v`'s own value (RFC-0039) — and the missing impl is the refusal.
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no storage of its own type, so a parameter cannot borrow one",
    label = "this parameter is taken by reference",
    note = "an Option has no storage of its own type to borrow: `None` is one value and `Some(v)` is `v`'s own value, so nothing behind a reference is shaped like an `Option<T>`. Take `Option<&T>`, or the option by value.",
    note = "a Rust slice is not one of the language's types: take `Slice<T, Rt>`, the language's `&[T]` (RFC-0047)."
)]
pub trait Borrowable<Rt>: OneValue<Rt>
where
    Rt: Runtime,
{
}

/// How a handler's result reaches the machine: the Rust value the closure
/// returns, and the run of the runtime's values it is written into.
pub trait Ret<Rt>: Sized
where
    Rt: Runtime,
{
    /// What the closure returns.
    type Of;
    /// The run the result is written into. A bound that admits only a result
    /// the caller can take away says `Form = One`: the `Pair` a slice is
    /// borrows the caller's frame (RFC-0047 §3).
    type Form: Form;

    /// How many of the runtime's values the result occupies.
    const WIDTH: usize = <Self::Form as Form>::WIDTH;

    fn into_run(value: Self::Of, rt: &Rt, out: &mut [Rt::Value]);
}

/// The arguments of a closure call, written into the callee's parameter
/// registers — the run the window a handler was lent begins with (RFC-0052
/// §7). Each member crosses at its own width, as a result does through `Ret`.
pub trait IntoRun<Rt>: Sized
where
    Rt: Runtime,
{
    const WIDTH: usize;

    fn into_run(self, rt: &Rt, out: &mut [Rt::Value]);
}

impl<Rt> IntoRun<Rt> for ()
where
    Rt: Runtime,
{
    const WIDTH: usize = 0;

    fn into_run(self, _: &Rt, _: &mut [Rt::Value]) {}
}

macro_rules! into_run_tuple {
    ($($A:ident: $at:tt),*) => {
        impl<Rt, $($A,)*> IntoRun<Rt> for ($($A,)*)
        where
            Rt: Runtime,
            $($A: Cross<Rt>,)*
        {
            const WIDTH: usize = 0 $(+ <<$A as Cross<Rt>>::Form as Form>::WIDTH)*;

            fn into_run(self, rt: &Rt, out: &mut [Rt::Value]) {
                let mut _at = 0usize;
                $(
                    let _width = <<$A as Cross<Rt>>::Form as Form>::WIDTH;
                    <$A as Cross<Rt>>::into_run(self.$at, rt, &mut out[_at.._at + _width]);
                    _at += _width;
                )*
            }
        }
    };
}

into_run_tuple!(A0: 0);
into_run_tuple!(A0: 0, A1: 1);
into_run_tuple!(A0: 0, A1: 1, A2: 2);

/// A result crossing as itself, at whichever width its type declares.
pub struct Val<T, C = Uniform>(PhantomData<fn() -> (T, C)>);

impl<T, Rt> Ret<Rt> for Val<T, Uniform>
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    type Of = T;
    type Form = <T as Cross<Rt>>::Form;

    fn into_run(value: T, rt: &Rt, out: &mut [Rt::Value]) {
        <T as Cross<Rt>>::into_run(value, rt, out)
    }
}

impl<T, Rt> Ret<Rt> for Val<T, Specialized>
where
    T: CrossSpecialized<Rt>,
    Rt: Runtime,
{
    type Of = T;
    type Form = One;

    fn into_run(value: T, rt: &Rt, out: &mut [Rt::Value]) {
        <T as CrossSpecialized<Rt>>::into_run(value, rt, out)
    }
}

/// How many of the runtime's values a call's arguments occupy and how many
/// its result writes back. Both numbers are sums of the `WIDTH` constants of
/// the declaration's types; `Glue`'s `Handler` impl is where they are added,
/// and `prepare` reads the answer rather than counting anything itself.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Width {
    pub args: usize,
    pub ret: usize,
}

/// The widest argument run a register form covers: a call of this many
/// values or fewer takes each in a register, and a wider one is lent its
/// window. RFC-0044 stage 2c fixed the cut at three.
pub const REGISTER_FORM: usize = 3;

impl Width {
    /// Whether the call takes its arguments in registers.
    pub fn in_registers(&self) -> bool {
        self.args <= REGISTER_FORM
    }
}

/// A declaration compiled to a Rust body, with the crossing on both sides of
/// it. One trait object per declared instance; the call operation holds it
/// and calls through it once.
pub trait Handler<Rt>: Send + Sync
where
    Rt: Runtime,
{
    /// One call site's own handle on this handler. A box's data pointer is
    /// the handler, so the call loads it and jumps; an `Arc`'s payload sits
    /// behind a header of no static offset, which costs the call a read of
    /// the vtable's alignment and the arithmetic over it.
    fn clone_box(&self) -> Box<dyn Handler<Rt>>;

    fn width(&self) -> Width;

    /// `frame` is the window above the calling frame, which a handler that
    /// calls a closure calls it in and a handler that calls none ignores
    /// (RFC-0050 rule 6).
    ///
    /// # Safety
    /// `run` holds `width().args` of the runtime's values in declaration
    /// order, `out` has room for `width().ret`, and every reference the
    /// handler takes out of `run` names storage live for the call
    /// (RFC-0018).
    unsafe fn call(&self, rt: &Rt, frame: Rt::Frame<'_>, run: &[Rt::Value], out: &mut [Rt::Value]);

    /// The register forms. A declaration whose arguments are `k` values wide
    /// and whose result is one value takes them in registers, and `prepare`
    /// calls the one form `width()` names.
    ///
    /// # Safety
    /// `width()` is `Width { args: k, ret: 1 }` for the `k` this form names,
    /// and the arguments are this call's own, in declaration order.
    unsafe fn call0(&self, rt: &Rt, frame: Rt::Frame<'_>) -> Rt::Value {
        // SAFETY: the caller's contract, which is `call_run`'s at no values.
        unsafe { self.call_run(rt, frame, &[]) }
    }

    /// # Safety
    /// As `call0`, at one value.
    unsafe fn call1(&self, rt: &Rt, frame: Rt::Frame<'_>, a: Rt::Value) -> Rt::Value {
        // SAFETY: the caller's contract, which is `call_run`'s at one value.
        unsafe { self.call_run(rt, frame, &[a]) }
    }

    /// # Safety
    /// As `call0`, at two values.
    unsafe fn call2(&self, rt: &Rt, frame: Rt::Frame<'_>, a: Rt::Value, b: Rt::Value) -> Rt::Value {
        // SAFETY: the caller's contract, which is `call_run`'s at two.
        unsafe { self.call_run(rt, frame, &[a, b]) }
    }

    /// # Safety
    /// As `call0`, at three values.
    unsafe fn call3(
        &self,
        rt: &Rt,
        frame: Rt::Frame<'_>,
        a: Rt::Value,
        b: Rt::Value,
        c: Rt::Value,
    ) -> Rt::Value {
        // SAFETY: the caller's contract, which is `call_run`'s at three.
        unsafe { self.call_run(rt, frame, &[a, b, c]) }
    }

    /// The window form: the arguments are lent as the run they already sit
    /// in, and the result is one value.
    ///
    /// # Safety
    /// As `call`, with `width().ret == 1`.
    unsafe fn call_run(&self, rt: &Rt, frame: Rt::Frame<'_>, run: &[Rt::Value]) -> Rt::Value {
        let mut out = [Rt::Value::default()];
        // SAFETY: the caller's contract.
        unsafe { self.call(rt, frame, run, &mut out) };
        out[0]
    }

    /// The slice form: the result is the run of a container's elements, and
    /// the caller takes the pair away in registers instead of lending a
    /// place to write it into.
    ///
    /// # Safety
    /// `width()` is `Width { args: 1, ret: 2 }`, and `a` is this call's own
    /// argument.
    unsafe fn call_slice(&self, rt: &Rt, frame: Rt::Frame<'_>, a: Rt::Value) -> Elements<Rt> {
        let mut out = [Rt::Value::default(); 2];
        // SAFETY: the caller's contract, which is `call`'s at one argument
        // and a result two values wide.
        unsafe { self.call(rt, frame, &[a], &mut out) };
        // SAFETY: `out` is the pair the slice's `into_run` just wrote, and
        // the elements it names are the caller's loan (RFC-0018).
        unsafe { Elements::from_words(rt.slice_from_run(&out)) }
    }
}

impl<Rt> Clone for Box<dyn Handler<Rt>>
where
    Rt: Runtime,
{
    fn clone(&self) -> Self {
        (**self).clone_box()
    }
}

/// A handler whose every crossing is one of the runtime's values: no
/// parameter and no result is a slice. This is what a task above `Sync`
/// runs. Such a call is awaited, and a slice is a borrow of the frame the
/// call laid its arguments on, whose loan is gone by the time the caller
/// resumes (RFC-0047 §3, rule 6).
pub trait ValuesOnly<Rt>: Handler<Rt>
where
    Rt: Runtime,
{
}

/// A declaration whose Rust body is an `async fn`: the call hands the
/// runtime a future, and the arguments the future owns (RFC-0046). One
/// `Pin<Box<dyn Future>>` per call; the future in the wide run is
/// RFC-0050's.
pub trait AsyncHandler<Rt>: Send + Sync
where
    Rt: Runtime,
{
    /// As `Handler::clone_box`.
    fn clone_box(&self) -> Box<dyn AsyncHandler<Rt>>;

    fn width(&self) -> Width;

    /// # Safety
    /// `run` holds `width().args` of the runtime's values in declaration
    /// order, and any storage a reference the body takes out of them names
    /// is live for as long as the future — which, a spawn's arguments being
    /// owned, it is (RFC-0046).
    unsafe fn call(
        &self,
        rt: Rt,
        run: &[Rt::Value],
    ) -> Pin<Box<dyn Future<Output = Rt::Value> + Send>>;
}

impl<Rt> Clone for Box<dyn AsyncHandler<Rt>>
where
    Rt: Runtime,
{
    fn clone(&self) -> Self {
        (**self).clone_box()
    }
}

/// A Rust closure with the crossing on both sides of it: `A` is the tuple of
/// the declaration's parameter modes, in the order the machine lays a call's
/// arguments (RFC-0052 §7), and `R` its result.
pub struct Glue<Rt, F, A, R> {
    f: F,
    shape: PhantomData<fn() -> (Rt, A, R)>,
}

// SAFETY: a `Glue` holds the closure and nothing else; the `PhantomData` is
// over `fn() -> _` and carries no value.
unsafe impl<Rt, F, A, R> Send for Glue<Rt, F, A, R> where F: Send {}
// SAFETY: as `Send`.
unsafe impl<Rt, F, A, R> Sync for Glue<Rt, F, A, R> where F: Sync {}

/// As `Glue`, for a body that awaits. The closure is shared because the
/// future it returns outlives the call that made it, so the call clones the
/// closure into the future rather than borrowing it.
pub struct AsyncGlue<Rt, F, A> {
    f: Arc<F>,
    shape: PhantomData<fn() -> (Rt, A)>,
}

// SAFETY: as `Glue`'s, the closure being behind a shared pointer.
unsafe impl<Rt, F, A> Send for AsyncGlue<Rt, F, A> where F: Send + Sync {}
// SAFETY: as `Send`.
unsafe impl<Rt, F, A> Sync for AsyncGlue<Rt, F, A> where F: Send + Sync {}

/// One arity: the two `Handler` impls and the two constructors that carry
/// their bounds. A closure is inferred higher-ranked only where the bound is
/// in scope at its own site, which is why the constructors exist and neither
/// `Glue` nor `AsyncGlue` has a `new`.
macro_rules! arity {
    ($glue:ident, $async_glue:ident, [$($result:tt)*] $(, $arg:ident: $out:ident)*) => {
        pub fn $glue<Rt, F, $($arg,)* R>(f: F) -> Glue<Rt, F, ($($arg,)*), R>
        where
            Rt: Runtime,
            F: for<'a, 'w> Fn(&'a Rt, Rt::Frame<'w> $(, <$arg as Arg<'a, Rt>>::Out)*)
                -> R::Of,
            $($arg: for<'a> Arg<'a, Rt>,)*
            R: $($result)*,
        {
            Glue { f, shape: PhantomData }
        }

        impl<Rt, F, $($arg,)* R> Handler<Rt> for Glue<Rt, F, ($($arg,)*), R>
        where
            Rt: Runtime,
            F: Clone + Send + Sync + 'static,
            F: for<'a, 'w> Fn(&'a Rt, Rt::Frame<'w> $(, <$arg as Arg<'a, Rt>>::Out)*)
                -> R::Of,
            $($arg: for<'a> Arg<'a, Rt> + 'static,)*
            R: Ret<Rt> + 'static,
        {
            fn clone_box(&self) -> Box<dyn Handler<Rt>> {
                Box::new(Glue::<Rt, F, ($($arg,)*), R> {
                    f: self.f.clone(),
                    shape: PhantomData,
                })
            }

            fn width(&self) -> Width {
                Width {
                    args: 0 $(+ <$arg as Arg<'_, Rt>>::WIDTH)*,
                    ret: <R as Ret<Rt>>::WIDTH,
                }
            }

            #[allow(unused_variables, unused_mut, unused_assignments)]
            unsafe fn call(
                &self,
                rt: &Rt,
                frame: Rt::Frame<'_>,
                run: &[Rt::Value],
                out: &mut [Rt::Value],
            ) {
                let mut _at = 0usize;
                $(
                    let _width = <$arg as Arg<'_, Rt>>::WIDTH;
                    // SAFETY: the caller's contract: `run` is this
                    // declaration's whole argument run, so each parameter's
                    // own values are the next `WIDTH` of it.
                    let $out = unsafe { $arg::take(rt, &run[_at.._at + _width]) };
                    _at += _width;
                )*
                <R as Ret<Rt>>::into_run((self.f)(rt, frame $(, $out)*), rt, out)
            }
        }

        impl<Rt, F, $($arg,)* R> ValuesOnly<Rt> for Glue<Rt, F, ($($arg,)*), R>
        where
            Rt: Runtime,
            F: Clone + Send + Sync + 'static,
            F: for<'a, 'w> Fn(&'a Rt, Rt::Frame<'w> $(, <$arg as Arg<'a, Rt>>::Out)*)
                -> R::Of,
            $($arg: for<'a> Arg<'a, Rt, Form = One> + 'static,)*
            R: Ret<Rt, Form = One> + 'static,
        {
        }

        pub fn $async_glue<Rt, F, $($arg,)*>(f: F) -> AsyncGlue<Rt, F, ($($arg,)*)>
        where
            Rt: Runtime,
            F: for<'a, 'w> Fn(&'a Rt, &'a mut Rt::Frame<'w> $(, <$arg as Arg<'a, Rt>>::Out)*)
                -> BoxFuture<'a, Rt::Value>,
            $($arg: for<'a> Arg<'a, Rt>,)*
        {
            AsyncGlue { f: Arc::new(f), shape: PhantomData }
        }

        impl<Rt, F, $($arg,)*> AsyncHandler<Rt> for AsyncGlue<Rt, F, ($($arg,)*)>
        where
            Rt: Runtime,
            F: Send + Sync + 'static,
            F: for<'a, 'w> Fn(&'a Rt, &'a mut Rt::Frame<'w> $(, <$arg as Arg<'a, Rt>>::Out)*)
                -> BoxFuture<'a, Rt::Value>,
            $($arg: for<'a> Arg<'a, Rt, Form = One> + 'static,)*
        {
            fn clone_box(&self) -> Box<dyn AsyncHandler<Rt>> {
                Box::new(AsyncGlue::<Rt, F, ($($arg,)*)> {
                    f: Arc::clone(&self.f),
                    shape: PhantomData,
                })
            }

            fn width(&self) -> Width {
                Width {
                    args: 0 $(+ <$arg as Arg<'_, Rt>>::WIDTH)*,
                    ret: 1,
                }
            }

            #[allow(unused_variables, unused_mut, unused_assignments)]
            unsafe fn call(
                &self,
                rt: Rt,
                run: &[Rt::Value],
            ) -> Pin<Box<dyn Future<Output = Rt::Value> + Send>> {
                let held: Vec<Rt::Value> = run.to_vec();
                let f = Arc::clone(&self.f);
                Box::pin(async move {
                    let mut rooted = rt.rooted();
                    let mut frame = Rt::frame_of(&mut rooted);
                    let mut _at = 0usize;
                    $(
                        let _width = <$arg as Arg<'_, Rt>>::WIDTH;
                        // SAFETY: as the synchronous impl's, over the run
                        // the future owns.
                        let $out = unsafe { $arg::take(&rt, &held[_at.._at + _width]) };
                        _at += _width;
                    )*
                    f(&rt, &mut frame $(, $out)*).await
                })
            }
        }
    };
}

// A slice is the one result wider than a value, and RFC-0047 §3 admits it
// from a declaration of one parameter and no other: every arity but one takes
// `Form = One`, so a slice returned anywhere else is a compile error.
arity!(glue0, async_glue0, [Ret<Rt, Form = One>]);
arity!(glue1, async_glue1, [Ret<Rt>], A0: a0);
arity!(glue2, async_glue2, [Ret<Rt, Form = One>], A0: a0, A1: a1);
arity!(glue3, async_glue3, [Ret<Rt, Form = One>], A0: a0, A1: a1, A2: a2);
arity!(glue4, async_glue4, [Ret<Rt, Form = One>], A0: a0, A1: a1, A2: a2, A3: a3);
arity!(glue5, async_glue5, [Ret<Rt, Form = One>], A0: a0, A1: a1, A2: a2, A3: a3, A4: a4);
arity!(
    glue6,
    async_glue6,
    [Ret<Rt, Form = One>],
    A0: a0,
    A1: a1,
    A2: a2,
    A3: a3,
    A4: a4,
    A5: a5
);
arity!(
    glue7,
    async_glue7,
    [Ret<Rt, Form = One>],
    A0: a0,
    A1: a1,
    A2: a2,
    A3: a3,
    A4: a4,
    A5: a5,
    A6: a6
);
arity!(
    glue8,
    async_glue8,
    [Ret<Rt, Form = One>],
    A0: a0,
    A1: a1,
    A2: a2,
    A3: a3,
    A4: a4,
    A5: a5,
    A6: a6,
    A7: a7
);

/// One handler per rung of `Task` (RFC-0046). `Sync` runs to its result in
/// the caller's frame; `Heavy` is a Rust body all the same, but one the
/// runtime hands to `Executor::spawn_blocking` and awaits; `Async` runs on
/// the async runtime and owns its interner because it lives across await
/// points.
///
/// The handler is shared, not owned: one declared instance is reached from
/// every call site the checker settled on it, and each site's operation
/// holds the object the registry built.
#[derive(Clone)]
pub enum ExternHandler<R: Runtime> {
    Sync(Box<dyn Handler<R>>),
    Heavy(Box<dyn Handler<R>>),
    Async(Box<dyn AsyncHandler<R>>),
}

impl<R: Runtime> ExternHandler<R> {
    pub fn sync(handler: impl Handler<R> + 'static) -> Self {
        Self::Sync(Box::new(handler))
    }

    /// A call the caller waits for is resumed after the frame it ran on is
    /// gone, so neither its arguments nor its result may borrow that frame.
    /// That is why this takes `ValuesOnly` and `sync` does not.
    pub fn heavy(handler: impl ValuesOnly<R> + 'static) -> Self {
        Self::Heavy(Box::new(handler))
    }

    pub fn awaited(handler: impl AsyncHandler<R> + 'static) -> Self {
        Self::Async(Box::new(handler))
    }

    /// Whether the call reaches its result without the caller suspending.
    /// A `Heavy` handler does not: it is offloaded and awaited.
    pub fn is_sync(&self) -> bool {
        match self {
            Self::Sync(_) => true,
            Self::Heavy(_) | Self::Async(_) => false,
        }
    }

    /// The task this handler runs at, which is the task its declaration
    /// named.
    pub fn task(&self) -> Task {
        match self {
            Self::Sync(_) => Task::Sync,
            Self::Async(_) => Task::Async,
            Self::Heavy(_) => Task::Heavy,
        }
    }

    pub fn width(&self) -> Width {
        match self {
            Self::Sync(f) | Self::Heavy(f) => f.width(),
            Self::Async(f) => f.width(),
        }
    }
}

pub struct Instance<R: Runtime> {
    pub signature: PolyTy,
    pub handler: ExternHandler<R>,
    /// The greatest task this instance runs — a ceiling, "at most", not
    /// the instance's own task. The `async fn` glue admits `Heavy` as well
    /// as `Async`, because it awaits either; the plain `fn` glue admits
    /// only `Sync`.
    pub admits: Task,
}

/// The number a call carries in `Callee::Extern` is an index into
/// `into_handlers`, and the compiler assigns it from `signatures`: the two
/// lists are the same list in the same order, and `acvus_mir::ty::Instances`
/// is the compiler's half of that contract.
pub struct Instances<R: Runtime> {
    pub concrete: Vec<Instance<R>>,
    pub generic: Option<ExternHandler<R>>,
}

impl<R: Runtime> Instances<R> {
    pub fn generic(handler: ExternHandler<R>) -> Self {
        Self {
            concrete: Vec::new(),
            generic: Some(handler),
        }
    }

    /// Adds the instances of `more` whose signature is not already here:
    /// two declarations of one family cast at one member are one instance.
    pub fn add_concrete(&mut self, more: Vec<Instance<R>>) {
        for instance in more {
            let present = self
                .concrete
                .iter()
                .any(|existing| existing.signature == instance.signature);
            if !present {
                self.concrete.push(instance);
            }
        }
    }

    pub fn signatures(&self) -> acvus_mir::ty::Instances {
        acvus_mir::ty::Instances {
            concrete: self
                .concrete
                .iter()
                .map(|i| acvus_mir::ty::InstanceSig {
                    ty: i.signature.clone(),
                    admits: i.admits,
                })
                .collect(),
            generic: self.generic.is_some(),
        }
    }

    pub fn into_handlers(self) -> Vec<ExternHandler<R>> {
        self.concrete
            .into_iter()
            .map(|i| i.handler)
            .chain(self.generic)
            .collect()
    }
}
