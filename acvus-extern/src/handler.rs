//! Typed handlers, ready for a runtime to call (RFC-0059).
//!
//! A declaration's Rust body is reached through a closure the proc macro
//! writes, wrapped in a `Glue` whose type parameters say how each parameter
//! comes out of the call's argument run and how the result is written back.
//! Every fact the ABI needs is a constant of those types, summed once in the
//! `Handler` impl and read back through `Handler::width`.

use std::marker::PhantomData;
use std::sync::Arc;

use acvus_mir::ty::{PolyTy, Task};
use futures::future::BoxFuture;

use crate::obj::{Cross, CrossSpecialized, Form, One, OneValue, Pair};
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
/// it.
///
/// Obligation across artifacts: that a call form's body inlines into the
/// operation holding it is asserted by `acvus-interpreter-test/benches/
/// asm_probe.rs` on the release machine, which is why every form below is
/// `#[inline]` and none of them is reachable through a `dyn`.
pub trait Handler<Rt>: Send + Sync + 'static
where
    Rt: Runtime,
{
    const WIDTH: Width;

    /// `frame` is the window above the calling frame, which a handler that
    /// calls a closure calls it in and a handler that calls none ignores
    /// (RFC-0050 rule 6).
    ///
    /// # Safety
    /// `run` holds `WIDTH.args` of the runtime's values in declaration
    /// order, `out` has room for `WIDTH.ret`, and every reference the
    /// handler takes out of `run` names storage live for the call
    /// (RFC-0018).
    unsafe fn call(&self, rt: &Rt, frame: Rt::Frame<'_>, run: &[Rt::Value], out: &mut [Rt::Value]);

    /// The register forms. A declaration whose arguments are `k` values wide
    /// and whose result is one value takes them in registers, and `prepare`
    /// builds the operation for the one form `WIDTH` names.
    ///
    /// # Safety
    /// `WIDTH` is `Width { args: k, ret: 1 }` for the `k` this form names,
    /// and the arguments are this call's own, in declaration order.
    #[inline]
    unsafe fn call0(&self, rt: &Rt, frame: Rt::Frame<'_>) -> Rt::Value {
        // SAFETY: the caller's contract, which is `call_run`'s at no values.
        unsafe { self.call_run(rt, frame, &[]) }
    }

    /// # Safety
    /// As `call0`, at one value.
    #[inline]
    unsafe fn call1(&self, rt: &Rt, frame: Rt::Frame<'_>, a: Rt::Value) -> Rt::Value {
        // SAFETY: the caller's contract, which is `call_run`'s at one value.
        unsafe { self.call_run(rt, frame, &[a]) }
    }

    /// # Safety
    /// As `call0`, at two values.
    #[inline]
    unsafe fn call2(&self, rt: &Rt, frame: Rt::Frame<'_>, a: Rt::Value, b: Rt::Value) -> Rt::Value {
        // SAFETY: the caller's contract, which is `call_run`'s at two.
        unsafe { self.call_run(rt, frame, &[a, b]) }
    }

    /// # Safety
    /// As `call0`, at three values.
    #[inline]
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
    /// As `call`, with `WIDTH.ret == 1`.
    #[inline]
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
    /// `WIDTH` is `Width { args: 1, ret: 2 }`, and `a` is this call's own
    /// argument.
    #[inline]
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

/// An argument run of `N` of the runtime's values, one per parameter.
pub struct InRegisters<const N: usize>;

/// An argument run the register forms do not cover: a parameter wider than
/// one of the runtime's values, or more than `REGISTER_FORM` parameters. Such
/// a call is lent the window its arguments already sit in.
pub struct InWindow;

/// An argument run, and the run one more parameter makes of it. `Form::Onto`
/// picks between the two, so the fold over a declaration's parameters needs
/// no bound the parameter types do not already carry.
///
/// A register form passes one register per parameter, and the other half of
/// that contract lives in the interpreter: `prepare::CallForm::of` builds a
/// register shape only where `Width::args` equals the number of parameters.
/// A parameter wider than one value therefore leaves the register forms here,
/// so that both halves say the same thing.
pub trait ArgRun {
    type WithOne: ArgRun;
    type WithPair: ArgRun;
}

impl ArgRun for InRegisters<0> {
    type WithOne = InRegisters<1>;
    type WithPair = InWindow;
}
impl ArgRun for InRegisters<1> {
    type WithOne = InRegisters<2>;
    type WithPair = InWindow;
}
impl ArgRun for InRegisters<2> {
    type WithOne = InRegisters<3>;
    type WithPair = InWindow;
}
impl ArgRun for InRegisters<3> {
    type WithOne = InWindow;
    type WithPair = InWindow;
}
impl ArgRun for InWindow {
    type WithOne = InWindow;
    type WithPair = InWindow;
}

/// The `WithOne` chain above is as long as the widest register form, and
/// `REGISTER_FORM` is that number written once more for `prepare` to read.
/// The assertion is what keeps the two one number.
const _: () = assert!(
    REGISTER_FORM == 3,
    "the ArgRun chain covers REGISTER_FORM parameters and no other number"
);

/// The run a declaration's parameters make, as the form its call takes.
pub trait Parameters<Rt>
where
    Rt: Runtime,
{
    type Run;
}

/// The call form a run of this shape and a result of form `R` take: which
/// `Rt::op_*` a factory names, and which `Rt::fused_*` (RFC-0059 rule 7).
///
/// The choice is a trait impl and not a branch on `Handler::WIDTH` because
/// the monomorphization collector reaches every operation a branch names,
/// whichever way the constant goes: `str`'s `len`, `find` and `concat` take
/// their arguments as pairs since `61da3863`, the arity macro named
/// `CallExtern1`/`CallExtern2` over them all the same, and the bodies that
/// came out — a bounds failure and nothing else, since a pair does not fit a
/// one-value run — are what `asm_probe` refused.
pub trait TakenForm<R>
where
    R: Form,
{
    fn op<Rt, H>(handler: H, shape: Rt::CallShape) -> Rt::Op
    where
        Rt: Runtime,
        H: Handler<Rt>;

    fn fused<Rt, H>(handler: H, shape: Rt::FusedShape) -> Rt::FusedCall
    where
        Rt: Runtime,
        H: Handler<Rt>;
}

macro_rules! taken_form {
    ($run:ty, op = $op:path, fused = $fused:path) => {
        impl TakenForm<One> for $run {
            fn op<Rt, H>(handler: H, shape: Rt::CallShape) -> Rt::Op
            where
                Rt: Runtime,
                H: Handler<Rt>,
            {
                $op(handler, shape)
            }

            fn fused<Rt, H>(handler: H, shape: Rt::FusedShape) -> Rt::FusedCall
            where
                Rt: Runtime,
                H: Handler<Rt>,
            {
                $fused(handler, shape)
            }
        }
    };
}

taken_form!(
    InRegisters<0>,
    op = Rt::op_no_argument,
    fused = Rt::fused_no_argument
);
taken_form!(
    InRegisters<1>,
    op = Rt::op_one_argument,
    fused = Rt::fused_one_argument
);
taken_form!(
    InRegisters<2>,
    op = Rt::op_two_arguments,
    fused = Rt::fused_two_arguments
);
taken_form!(
    InRegisters<3>,
    op = Rt::op_three_arguments,
    fused = no_fused_run
);
taken_form!(InWindow, op = Rt::op_wide, fused = no_fused_run);

/// A result wider than one value is the run of a container's elements, which
/// RFC-0047 §3 admits from a declaration of one parameter of one value and
/// from nowhere else. The missing impls of every other run are that refusal.
impl TakenForm<Pair> for InRegisters<1> {
    fn op<Rt, H>(handler: H, shape: Rt::CallShape) -> Rt::Op
    where
        Rt: Runtime,
        H: Handler<Rt>,
    {
        Rt::op_slice::<H>(handler, shape)
    }

    fn fused<Rt, H>(_: H, _: Rt::FusedShape) -> Rt::FusedCall
    where
        Rt: Runtime,
        H: Handler<Rt>,
    {
        panic!("a fused run holds no call whose result is a run of elements")
    }
}

/// What the module table holds for one declared instance: the handler with
/// its type erased, which `prepare` turns back into a typed operation by
/// handing it the shape it decided for the call site. This is the one `dyn`
/// on the path, and it is taken once, at preparation.
pub trait HandlerFactory<Rt>: Send + Sync
where
    Rt: Runtime,
{
    fn clone_box(&self) -> Box<dyn HandlerFactory<Rt>>;
    fn width(&self) -> Width;
    fn into_op(self: Box<Self>, shape: Rt::CallShape) -> Rt::Op;
    fn into_fused(self: Box<Self>, shape: Rt::FusedShape) -> Rt::FusedCall;
}

impl<Rt> Clone for Box<dyn HandlerFactory<Rt>>
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
/// runtime a future that owns its arguments and outlives the frame the call
/// was made on (RFC-0046).
///
/// The future is boxed, and this trait names `BoxFuture` rather than a
/// future type of the handler's own, because naming an `async` block's type
/// in an associated type needs `impl_trait_in_assoc_type`, which is unstable
/// on the toolchain this repository pins. Storing an async handler's future
/// where it lies waits for that feature: without it `size_of` of the future
/// is not a constant any impl can state.
pub trait AsyncCall<Rt>: Send + Sync + 'static
where
    Rt: Runtime,
{
    const WIDTH: Width;

    /// # Safety
    /// `run` holds `WIDTH.args` of the runtime's values in declaration
    /// order, and any storage a reference the body takes out of them names
    /// is live for as long as the future — which, a spawn's arguments being
    /// owned, it is (RFC-0046).
    unsafe fn call(&self, rt: Rt, run: &[Rt::Value]) -> BoxFuture<'static, Rt::Value>;
}

pub trait AsyncFactory<Rt>: Send + Sync
where
    Rt: Runtime,
{
    fn clone_box(&self) -> Box<dyn AsyncFactory<Rt>>;
    fn width(&self) -> Width;
    fn into_op(self: Box<Self>, shape: Rt::AsyncShape) -> Rt::Op;
}

impl<Rt, H> AsyncFactory<Rt> for H
where
    Rt: Runtime,
    H: AsyncCall<Rt> + Clone,
{
    fn clone_box(&self) -> Box<dyn AsyncFactory<Rt>> {
        Box::new(self.clone())
    }

    fn width(&self) -> Width {
        <H as AsyncCall<Rt>>::WIDTH
    }

    fn into_op(self: Box<Self>, shape: Rt::AsyncShape) -> Rt::Op {
        Rt::async_extern_op::<H>(*self, shape)
    }
}

impl<Rt> Clone for Box<dyn AsyncFactory<Rt>>
where
    Rt: Runtime,
{
    fn clone(&self) -> Self {
        (**self).clone_box()
    }
}

fn no_fused_run<Rt, H>(_: H, _: Rt::FusedShape) -> Rt::FusedCall
where
    Rt: Runtime,
    H: Handler<Rt>,
{
    panic!("a fused run holds no call of this many arguments")
}

/// A Rust closure with the crossing on both sides of it: `A` is the tuple of
/// the declaration's parameter modes, in the order the machine lays a call's
/// arguments (RFC-0052 §7), and `R` its result.
pub struct Glue<Rt, F, A, R> {
    f: F,
    shape: PhantomData<fn() -> (Rt, A, R)>,
}

impl<Rt, F, A, R> Clone for Glue<Rt, F, A, R>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Glue {
            f: self.f.clone(),
            shape: PhantomData,
        }
    }
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

impl<Rt, F, A> Clone for AsyncGlue<Rt, F, A> {
    fn clone(&self) -> Self {
        AsyncGlue {
            f: Arc::clone(&self.f),
            shape: PhantomData,
        }
    }
}

// SAFETY: as `Glue`'s, the closure being behind a shared pointer.
unsafe impl<Rt, F, A> Send for AsyncGlue<Rt, F, A> where F: Send + Sync {}
// SAFETY: as `Send`.
unsafe impl<Rt, F, A> Sync for AsyncGlue<Rt, F, A> where F: Send + Sync {}

/// The run of `InRegisters<0>` widened by each parameter in turn: the fold
/// whose answer `TakenForm` reads.
macro_rules! run_of {
    ($rt:ty, $run:ty) => { $run };
    ($rt:ty, $run:ty, $arg:ident $(, $rest:ident)*) => {
        run_of!($rt, <<$arg as Arg<'static, $rt>>::Form as Form>::Onto<$run> $(, $rest)*)
    };
}

/// One arity: the two `Handler` impls and the two constructors that carry
/// their bounds. A closure is inferred higher-ranked only where the bound is
/// in scope at its own site, which is why the constructors exist and neither
/// `Glue` nor `AsyncGlue` has a `new`.
macro_rules! arity {
    (
        $glue:ident, $async_glue:ident, [$($result:tt)*]
        $(, $arg:ident: $out:ident)*
    ) => {
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
            const WIDTH: Width = Width {
                args: 0 $(+ <$arg as Arg<'static, Rt>>::WIDTH)*,
                ret: <R as Ret<Rt>>::WIDTH,
            };

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

        impl<Rt, $($arg,)*> Parameters<Rt> for ($($arg,)*)
        where
            Rt: Runtime,
            $($arg: for<'a> Arg<'a, Rt>,)*
        {
            type Run = run_of!(Rt, InRegisters<0> $(, $arg)*);
        }

        impl<Rt, F, $($arg,)* R> HandlerFactory<Rt> for Glue<Rt, F, ($($arg,)*), R>
        where
            Rt: Runtime,
            F: Clone + Send + Sync + 'static,
            F: for<'a, 'w> Fn(&'a Rt, Rt::Frame<'w> $(, <$arg as Arg<'a, Rt>>::Out)*)
                -> R::Of,
            $($arg: for<'a> Arg<'a, Rt> + 'static,)*
            R: Ret<Rt> + 'static,
            <($($arg,)*) as Parameters<Rt>>::Run: TakenForm<<R as Ret<Rt>>::Form>,
        {
            fn clone_box(&self) -> Box<dyn HandlerFactory<Rt>> {
                Box::new(self.clone())
            }

            fn width(&self) -> Width {
                <Self as Handler<Rt>>::WIDTH
            }

            fn into_op(self: Box<Self>, shape: Rt::CallShape) -> Rt::Op {
                <<($($arg,)*) as Parameters<Rt>>::Run as TakenForm<
                    <R as Ret<Rt>>::Form,
                >>::op::<Rt, Self>(*self, shape)
            }

            fn into_fused(self: Box<Self>, shape: Rt::FusedShape) -> Rt::FusedCall {
                <<($($arg,)*) as Parameters<Rt>>::Run as TakenForm<
                    <R as Ret<Rt>>::Form,
                >>::fused::<Rt, Self>(*self, shape)
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

        impl<Rt, F, $($arg,)*> AsyncCall<Rt> for AsyncGlue<Rt, F, ($($arg,)*)>
        where
            Rt: Runtime,
            F: Send + Sync + 'static,
            F: for<'a, 'w> Fn(&'a Rt, &'a mut Rt::Frame<'w> $(, <$arg as Arg<'a, Rt>>::Out)*)
                -> BoxFuture<'a, Rt::Value>,
            $($arg: for<'a> Arg<'a, Rt, Form = One> + 'static,)*
        {
            const WIDTH: Width = Width {
                args: 0 $(+ <$arg as Arg<'static, Rt>>::WIDTH)*,
                ret: 1,
            };

            #[allow(unused_variables, unused_mut, unused_assignments)]
            unsafe fn call(&self, rt: Rt, run: &[Rt::Value]) -> BoxFuture<'static, Rt::Value> {
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
// from a declaration of one parameter of one value and from no other: every
// arity but one binds `Form = One` below, and `TakenForm<Pair>` is
// implemented for `InRegisters<1>` alone, so a slice returned anywhere else
// is a compile error.
arity!(glue0, async_glue0, [Ret<Rt, Form = One>]);
arity!(glue1, async_glue1, [Ret<Rt>], A0: a0);
arity!(glue2, async_glue2, [Ret<Rt, Form = One>], A0: a0, A1: a1);
arity!(glue3, async_glue3, [Ret<Rt, Form = One>], A0: a0, A1: a1, A2: a2);
arity!(
    glue4, async_glue4, [Ret<Rt, Form = One>],
    A0: a0, A1: a1, A2: a2, A3: a3
);
arity!(
    glue5, async_glue5, [Ret<Rt, Form = One>],
    A0: a0, A1: a1, A2: a2, A3: a3, A4: a4
);
arity!(
    glue6, async_glue6, [Ret<Rt, Form = One>],
    A0: a0, A1: a1, A2: a2, A3: a3, A4: a4, A5: a5
);
arity!(
    glue7, async_glue7, [Ret<Rt, Form = One>],
    A0: a0, A1: a1, A2: a2, A3: a3, A4: a4, A5: a5, A6: a6
);
arity!(
    glue8, async_glue8, [Ret<Rt, Form = One>],
    A0: a0, A1: a1, A2: a2, A3: a3, A4: a4, A5: a5, A6: a6, A7: a7
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
    Sync(Box<dyn HandlerFactory<R>>),
    Heavy(Box<dyn HandlerFactory<R>>),
    Async(Box<dyn AsyncFactory<R>>),
}

impl<R: Runtime> ExternHandler<R> {
    pub fn sync(handler: impl HandlerFactory<R> + 'static) -> Self {
        Self::Sync(Box::new(handler))
    }

    /// A call the caller waits for is resumed after the frame it ran on is
    /// gone, so neither its arguments nor its result may borrow that frame.
    /// That is why this takes `ValuesOnly` and `sync` does not.
    pub fn heavy(handler: impl ValuesOnly<R> + HandlerFactory<R> + 'static) -> Self {
        Self::Heavy(Box::new(handler))
    }

    pub fn awaited(handler: impl AsyncFactory<R> + 'static) -> Self {
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

/// The operation a host with no register machine runs a call as: the handler
/// behind a closure that takes the argument run as it comes. A host that
/// lays its arguments in registers builds an operation of its own instead
/// and never reaches this one.
pub enum DirectOp<Rt>
where
    Rt: Runtime,
{
    Call(Box<dyn Fn(&Rt, &[Rt::Value], &mut [Rt::Value]) + Send + Sync>),
    Await(Box<dyn Fn(Rt, &[Rt::Value]) -> BoxFuture<'static, Rt::Value> + Send + Sync>),
}

impl<Rt> DirectOp<Rt>
where
    Rt: Runtime,
{
    pub fn of<H>(handler: H) -> Self
    where
        H: Handler<Rt>,
    {
        DirectOp::Call(Box::new(move |rt, run, out| {
            let mut rooted = rt.rooted();
            let frame = Rt::frame_of(&mut rooted);
            // SAFETY: the contract of `DirectOp::call`, which is this
            // closure's only caller.
            unsafe { handler.call(rt, frame, run, out) }
        }))
    }

    pub fn awaiting<H>(handler: H) -> Self
    where
        H: AsyncCall<Rt>,
    {
        DirectOp::Await(Box::new(move |rt, run| {
            // SAFETY: the contract of `DirectOp::call_async`.
            unsafe { handler.call(rt, run) }
        }))
    }

    /// # Safety
    /// As `Handler::call`: `run` is the declaration's whole argument run and
    /// `out` has room for its result.
    pub unsafe fn call(&self, rt: &Rt, run: &[Rt::Value], out: &mut [Rt::Value]) {
        let DirectOp::Call(call) = self else {
            panic!("an awaited handler was called for its value")
        };
        call(rt, run, out)
    }

    /// # Safety
    /// As `Handler::call_run`.
    pub unsafe fn call_run(&self, rt: &Rt, run: &[Rt::Value]) -> Rt::Value {
        let mut out = [Rt::Value::default()];
        // SAFETY: the caller's contract.
        unsafe { self.call(rt, run, &mut out) };
        out[0]
    }

    /// # Safety
    /// As `AsyncCall::call`.
    pub unsafe fn call_async(&self, rt: Rt, run: &[Rt::Value]) -> BoxFuture<'static, Rt::Value> {
        let DirectOp::Await(call) = self else {
            panic!("a synchronous handler was called for a future")
        };
        call(rt, run)
    }
}

/// The ten entries of a host that runs a call where it stands: every form
/// builds the same `DirectOp`, because such a host lays no arguments in
/// registers and so takes every form the same way.
#[macro_export]
macro_rules! direct_call_forms {
    () => {
        $crate::direct_call_forms!(@op op_no_argument);
        $crate::direct_call_forms!(@op op_one_argument);
        $crate::direct_call_forms!(@op op_two_arguments);
        $crate::direct_call_forms!(@op op_three_arguments);
        $crate::direct_call_forms!(@op op_wide);
        $crate::direct_call_forms!(@op op_slice);
        $crate::direct_call_forms!(@fused fused_no_argument);
        $crate::direct_call_forms!(@fused fused_one_argument);
        $crate::direct_call_forms!(@fused fused_two_arguments);

        fn async_extern_op<H>(handler: H, _: Self::AsyncShape) -> Self::Op
        where
            H: $crate::AsyncCall<Self>,
        {
            $crate::DirectOp::awaiting(handler)
        }
    };
    (@op $name:ident) => {
        fn $name<H>(handler: H, _: Self::CallShape) -> Self::Op
        where
            H: $crate::Handler<Self>,
        {
            $crate::DirectOp::of(handler)
        }
    };
    (@fused $name:ident) => {
        fn $name<H>(handler: H, _: Self::FusedShape) -> Self::FusedCall
        where
            H: $crate::Handler<Self>,
        {
            $crate::DirectOp::of(handler)
        }
    };
}
