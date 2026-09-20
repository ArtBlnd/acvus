//! Typed handlers, ready for a runtime to call (RFC-0059).
//!
//! A declaration's Rust body is reached through a closure the proc macro
//! writes, wrapped in a `Glue` whose type parameters say how each parameter
//! comes out of the call's argument run and how the result is written back.
//! Every fact the ABI needs is a constant of those types, summed once in the
//! `Handler` impl and read back through `Handler::width`.

use std::marker::PhantomData;
use std::sync::Arc;

use acvus_mir::ty::{PolyTy, Task, Ty};
use acvus_utils::Interner;
use futures::future::BoxFuture;

use crate::loan::Loan;
use crate::obj::{Cross, Form, FormKind, One, OneValue, Pair, Run};
use crate::runtime::Runtime;

/// One argument of one call site as the host settled it: the type the
/// checker gave it, and the interner that resolves the names in that type.
#[derive(Clone, Copy)]
pub struct ArgAt<'a> {
    pub interner: &'a Interner,
    pub ty: &'a Ty,
}

/// A call site of `n` arguments typed `Unit`. Obligation across artifacts: a
/// projection parameter reaching one panics in
/// `projection::object_fields_at` or `projection::variant_tags_at`, which are
/// the two readers of a settled type.
///
/// Decision not to gate this on `cfg(test)`: the callers are the integration
/// tests of three crates — `acvus-extern`, `acvus-ext` and
/// `acvus-interpreter` — and `cfg(test)` in this crate does not reach them.
#[doc(hidden)]
pub struct SitesNoParameterReads {
    interner: Interner,
    ty: Ty,
}

impl Default for SitesNoParameterReads {
    fn default() -> Self {
        SitesNoParameterReads {
            interner: Interner::new(),
            ty: Ty::Unit,
        }
    }
}

impl SitesNoParameterReads {
    pub fn args(&self, n: usize) -> Vec<ArgAt<'_>> {
        vec![
            ArgAt {
                interner: &self.interner,
                ty: &self.ty,
            };
            n
        ]
    }
}

/// Which crossing a parameter or a result takes (RFC-0040).
pub struct Uniform;
/// The crossing of a `Monomorphize` member instance.
pub struct Specialized;

/// What a parameter's crossing needs from the call site, which the glue
/// holds and every call of that site reads.
///
/// This is not an associated type of `Arg`, and the separate trait is a
/// decision. `Arg` is parameterized by the call's own lifetime, associated
/// type projections are invariant, and the table a glue holds is built once
/// and outlives every call — so passing `&<Self as Arg<'a, Rt>>::Site` from
/// a `'static` table would unify `'a` with `'static` and the borrow checker
/// would then demand that the runtime and the argument run outlive the
/// call. Carrying the datum on a lifetime-free trait is what keeps a site
/// table one type per parameter instead of one per lifetime the parameter
/// is read at.
pub trait Sited<Rt>: Sized
where
    Rt: Runtime,
{
    /// A parameter that needs nothing says `()`, which is zero-sized, so
    /// the per-site glue of a declaration of plain parameters is the
    /// closure and nothing else.
    type Site: Clone + Send + Sync + 'static;

    fn site(at: ArgAt<'_>) -> Self::Site;
}

/// How one Rust parameter takes its argument out of a call's argument run.
/// The mode — by value, by shared reference, by exclusive reference — is
/// written in Rust and read by the macro (RFC-0015); the width is the
/// type's.
pub trait Arg<'a, Rt>: Sited<Rt>
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
    unsafe fn take<'s>(
        rt: &'a Rt,
        run: &'a [Rt::Value],
        site: &'s <Self as Sited<Rt>>::Site,
    ) -> Self::Out;
}

/// A parameter taken by value: the crossing builds the Rust value.
pub struct ByValue<T, C = Uniform>(PhantomData<fn() -> (T, C)>);
/// A parameter taken by reference: the body reads, and at a `Mut` loan writes,
/// the storage the caller lent (RFC-0018).
pub struct ByRef<T, M, C = Uniform>(PhantomData<fn() -> (T, M, C)>);

impl<T, Rt> Sited<Rt> for ByValue<T, Uniform>
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    type Site = ();

    fn site(_: ArgAt<'_>) {}
}

impl<T, Rt> Sited<Rt> for ByValue<T, Specialized>
where
    T: OneValue<Rt, Specialized>,
    Rt: Runtime,
{
    type Site = ();

    fn site(_: ArgAt<'_>) {}
}

impl<T, M, Rt> Sited<Rt> for ByRef<T, M, Uniform>
where
    T: Borrowable<Rt>,
    M: Loan,
    Rt: Runtime,
{
    type Site = ();

    fn site(_: ArgAt<'_>) {}
}

impl<T, M, Rt> Sited<Rt> for ByRef<T, M, Specialized>
where
    T: BorrowableSpecialized<Rt>,
    M: Loan,
    Rt: Runtime,
{
    type Site = ();

    fn site(_: ArgAt<'_>) {}
}

impl<'a, T, Rt> Arg<'a, Rt> for ByValue<T, Uniform>
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    type Out = T;
    type Form = <T as Cross<Rt>>::Form;

    unsafe fn take<'s>(rt: &'a Rt, run: &'a [Rt::Value], _: &'s ()) -> T {
        // SAFETY: the caller's contract, which is `Cross::from_run`'s.
        unsafe { <T as Cross<Rt>>::from_run(rt, run) }
    }
}

impl<'a, T, Rt> Arg<'a, Rt> for ByValue<T, Specialized>
where
    T: OneValue<Rt, Specialized>,
    Rt: Runtime,
{
    type Out = T;
    type Form = One;

    unsafe fn take<'s>(rt: &'a Rt, run: &'a [Rt::Value], _: &'s ()) -> T {
        // SAFETY: as the uniform impl's.
        unsafe { <T as OneValue<Rt, Specialized>>::from_run(rt, run) }
    }
}

impl<'a, T, M, Rt> Arg<'a, Rt> for ByRef<T, M, Uniform>
where
    T: Borrowable<Rt>,
    M: Loan,
    Rt: Runtime,
{
    type Out = M::Of<'a, T>;
    type Form = One;

    unsafe fn take<'s>(rt: &'a Rt, run: &'a [Rt::Value], _: &'s ()) -> M::Of<'a, T> {
        // SAFETY: the caller's contract: a live storage of `T`, exclusively
        // named at a `Mut` loan (RFC-0018).
        unsafe { M::borrow::<T, Uniform, Rt>(rt, &run[0]) }
    }
}

impl<'a, T, M, Rt> Arg<'a, Rt> for ByRef<T, M, Specialized>
where
    T: BorrowableSpecialized<Rt>,
    M: Loan,
    Rt: Runtime,
{
    type Out = M::Of<'a, T>;
    type Form = One;

    unsafe fn take<'s>(rt: &'a Rt, run: &'a [Rt::Value], _: &'s ()) -> M::Of<'a, T> {
        // SAFETY: as the uniform impl's, at the specialized representation.
        unsafe { M::borrow::<T, Specialized, Rt>(rt, &run[0]) }
    }
}

/// A type a parameter may take by reference: one whose values are places the
/// language names. An `Option` is not — `None` is one value and `Some(v)` is
/// `v`'s own value (RFC-0039) — and the missing impl is the refusal.
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no storage of its own type, so a parameter cannot borrow one",
    label = "this parameter is taken by reference",
    note = "a borrowed aggregate crosses as its projection: where `{Self}` is a `#[derive(TyArg)] #[projection]` aggregate, write `{Self}Ref<'_>`, or `{Self}Mut<'_>` for a struct and `{Self}Mut<'_, Rt>` for an enum (RFC-0050 rule 6).",
    note = "an Option has no storage of its own type to borrow: `None` is one value and `Some(v)` is `v`'s own value, so nothing behind a reference is shaped like an `Option<T>`. Take `Option<&T>`, or the option by value.",
    note = "a Rust slice is not one of the language's types: take `Slice<T, Shared, Rt>`, the language's `&[T]` (RFC-0047)."
)]
pub trait Borrowable<Rt>: OneValue<Rt>
where
    Rt: Runtime,
{
}

/// A type a parameter of a `Monomorphize` member may take by reference: one
/// whose specialized crossing writes a `Self` into the storage the caller
/// lends, which is what `OneValue<_, Specialized>::deref` reads back.
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no storage of its own type at a monomorphized member, so a parameter cannot borrow one",
    label = "this parameter of a monomorphized member is taken by reference",
    note = "a Result crosses a member by value: a crossed `Result` is the language's flat variant, not Rust's `Result<Owned, Owned>`, so nothing behind a reference is shaped like a `Result<T, E>` (RFC-0050 rule 8).",
    note = "an Option crosses a member by value: `None` is one value and `Some(v)` is `v`'s own value, so nothing behind a reference is shaped like an `Option<T>`. Take `Option<&T>` (RFC-0039)."
)]
pub trait BorrowableSpecialized<Rt>: OneValue<Rt, Specialized>
where
    Rt: Runtime,
{
}

/// The destination run a call's result is written into, lent for the call's
/// duration by the frame that owns it (RFC-0050 rule 6). It is the caller's
/// own registers where the result stays in the frame, and the heap object's
/// body where it escapes; a handler writes the same components either way.
pub type Out<'a, Rt> = &'a mut [<Rt as Runtime>::Value];

pub trait Ret<Rt>: Sized
where
    Rt: Runtime,
{
    type Of<'a>;
    /// The run the result is written into. A bound that admits only a result
    /// the caller can take away says `Form = One`: the `Pair` a view or a
    /// slice is borrows the caller's frame (RFC-0047 §3), and the `Run<W>` an
    /// aggregate is names the caller's destination.
    type Form: Form;

    /// How many of the runtime's values the result occupies.
    const WIDTH: usize = <Self::Form as Form>::WIDTH;

    fn into_run(value: Self::Of<'_>, rt: &Rt, out: Out<'_, Rt>);
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
into_run_tuple!(A0: 0, A1: 1, A2: 2, A3: 3);
into_run_tuple!(A0: 0, A1: 1, A2: 2, A3: 3, A4: 4);
into_run_tuple!(A0: 0, A1: 1, A2: 2, A3: 3, A4: 4, A5: 5);
into_run_tuple!(A0: 0, A1: 1, A2: 2, A3: 3, A4: 4, A5: 5, A6: 6);
into_run_tuple!(A0: 0, A1: 1, A2: 2, A3: 3, A4: 4, A5: 5, A6: 6, A7: 7);

/// A result crossing as itself, at whichever width its type declares.
pub struct Val<T, C = Uniform>(PhantomData<fn() -> (T, C)>);

impl<T, Rt> Ret<Rt> for Val<T, Uniform>
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    type Of<'a> = T;
    type Form = <T as Cross<Rt>>::ReturnForm;

    fn into_run(value: T, rt: &Rt, out: Out<'_, Rt>) {
        <T as Cross<Rt>>::into_return_run(value, rt, out)
    }
}

impl<T, Rt> Ret<Rt> for Val<T, Specialized>
where
    T: OneValue<Rt, Specialized>,
    Rt: Runtime,
{
    type Of<'a> = T;
    type Form = One;

    fn into_run(value: T, rt: &Rt, out: Out<'_, Rt>) {
        <T as OneValue<Rt, Specialized>>::into_run(value, rt, out)
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
    pub result: FormKind,
}

/// The widest argument run a register form covers: a call of this many of the
/// runtime's values or fewer takes each in a register, and a wider one is lent
/// its window. RFC-0044 stage 2c fixed the cut at three parameters, which with
/// a `&str` or a slice parameter counting two values is four values.
///
/// Four is where the handlers run out, not where the operations do. Of the 248
/// declarations instantiated in the `asm_probe` bench binary, 30 are past the
/// register forms when a pair costs two values; raising the cut to four brings
/// 27 of the 30 in, to five brings 29, to six all 30, and every step above
/// four costs one more `Runtime::op_*_arguments` method that every host
/// implements. `ops/call.rs` asserts the cache-line bound that the operations
/// themselves owe.
pub const REGISTER_FORM: usize = 4;

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

    /// # Safety
    /// As `call0`, at four values.
    #[inline]
    unsafe fn call4(
        &self,
        rt: &Rt,
        frame: Rt::Frame<'_>,
        a: Rt::Value,
        b: Rt::Value,
        c: Rt::Value,
        d: Rt::Value,
    ) -> Rt::Value {
        // SAFETY: the caller's contract, which is `call_run`'s at four.
        unsafe { self.call_run(rt, frame, &[a, b, c, d]) }
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

    /// The pair forms: the caller takes the result away in the two adjacent
    /// registers `acvus-interpreter`'s `assign_slots` placed for it, rather
    /// than lending a place to write it into.
    ///
    /// # Safety
    /// `WIDTH` is `Width { args: k, ret: 2 }` for the `k` this form names,
    /// and the arguments are this call's own, in declaration order.
    #[inline]
    unsafe fn call_pair1(&self, rt: &Rt, frame: Rt::Frame<'_>, a: Rt::Value) -> [Rt::Value; 2] {
        // SAFETY: the caller's contract, which is `call_pair_run`'s at one.
        unsafe { self.call_pair_run(rt, frame, &[a]) }
    }

    /// # Safety
    /// As `call_pair1`, at two values.
    #[inline]
    unsafe fn call_pair2(
        &self,
        rt: &Rt,
        frame: Rt::Frame<'_>,
        a: Rt::Value,
        b: Rt::Value,
    ) -> [Rt::Value; 2] {
        // SAFETY: the caller's contract, which is `call_pair_run`'s at two.
        unsafe { self.call_pair_run(rt, frame, &[a, b]) }
    }

    /// # Safety
    /// As `call_pair1`, at three values.
    #[inline]
    unsafe fn call_pair3(
        &self,
        rt: &Rt,
        frame: Rt::Frame<'_>,
        a: Rt::Value,
        b: Rt::Value,
        c: Rt::Value,
    ) -> [Rt::Value; 2] {
        // SAFETY: the caller's contract, which is `call_pair_run`'s at three.
        unsafe { self.call_pair_run(rt, frame, &[a, b, c]) }
    }

    /// # Safety
    /// As `call_pair1`, at four values.
    #[inline]
    unsafe fn call_pair4(
        &self,
        rt: &Rt,
        frame: Rt::Frame<'_>,
        a: Rt::Value,
        b: Rt::Value,
        c: Rt::Value,
        d: Rt::Value,
    ) -> [Rt::Value; 2] {
        // SAFETY: the caller's contract, which is `call_pair_run`'s at four.
        unsafe { self.call_pair_run(rt, frame, &[a, b, c, d]) }
    }

    /// The aggregate forms: the caller lends the destination run its
    /// placement gave the result — its own registers, or the heap object's
    /// body — and the handler writes `WIDTH.ret` components into it
    /// (RFC-0050 rules 5 and 6). The window form of this family is `call`
    /// itself, which already takes the run and the destination.
    ///
    /// # Safety
    /// `WIDTH` is `Width { args: k, ret: w, result: Components }` for the
    /// `k` this form names, the arguments are this call's own in declaration
    /// order, and `out` is `w` of the runtime's values the caller owns.
    #[inline]
    unsafe fn call_out0(&self, rt: &Rt, frame: Rt::Frame<'_>, out: Out<'_, Rt>) {
        // SAFETY: the caller's contract, which is `call`'s at no arguments.
        unsafe { self.call(rt, frame, &[], out) }
    }

    /// # Safety
    /// As `call_out0`, at one value.
    #[inline]
    unsafe fn call_out1(&self, rt: &Rt, frame: Rt::Frame<'_>, a: Rt::Value, out: Out<'_, Rt>) {
        // SAFETY: the caller's contract, which is `call`'s at one argument.
        unsafe { self.call(rt, frame, &[a], out) }
    }

    /// # Safety
    /// As `call_out0`, at two values.
    #[inline]
    unsafe fn call_out2(
        &self,
        rt: &Rt,
        frame: Rt::Frame<'_>,
        a: Rt::Value,
        b: Rt::Value,
        out: Out<'_, Rt>,
    ) {
        // SAFETY: the caller's contract, which is `call`'s at two arguments.
        unsafe { self.call(rt, frame, &[a, b], out) }
    }

    /// # Safety
    /// As `call_out0`, at three values.
    #[inline]
    unsafe fn call_out3(
        &self,
        rt: &Rt,
        frame: Rt::Frame<'_>,
        a: Rt::Value,
        b: Rt::Value,
        c: Rt::Value,
        out: Out<'_, Rt>,
    ) {
        // SAFETY: the caller's contract, which is `call`'s at three arguments.
        unsafe { self.call(rt, frame, &[a, b, c], out) }
    }

    /// # Safety
    /// As `call_out0`, at four values.
    #[inline]
    unsafe fn call_out4(
        &self,
        rt: &Rt,
        frame: Rt::Frame<'_>,
        a: Rt::Value,
        b: Rt::Value,
        c: Rt::Value,
        d: Rt::Value,
        out: Out<'_, Rt>,
    ) {
        // SAFETY: the caller's contract, which is `call`'s at four arguments.
        unsafe { self.call(rt, frame, &[a, b, c, d], out) }
    }

    /// # Safety
    /// As `call`, with `WIDTH.ret == 2`.
    #[inline]
    unsafe fn call_pair_run(
        &self,
        rt: &Rt,
        frame: Rt::Frame<'_>,
        run: &[Rt::Value],
    ) -> [Rt::Value; 2] {
        let mut out = [Rt::Value::default(); 2];
        // SAFETY: the caller's contract.
        unsafe { self.call(rt, frame, run, &mut out) };
        out
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
/// The fold counts **the runtime's values, not parameters**: a `&str` or a
/// slice parameter is two of them and takes the run two wider, so a
/// declaration whose parameters total `REGISTER_FORM` values or fewer takes a
/// register form naming one register per value. A run wider than that is lent
/// its window. The other half of that contract lives in the interpreter:
/// `prepare::CallForm::of` reads the same `Width::args`, which is the same
/// count of values, so both halves say one thing.
pub trait ArgRun {
    type WithOne: ArgRun;
    type WithPair: ArgRun;
}

impl ArgRun for InRegisters<0> {
    type WithOne = InRegisters<1>;
    type WithPair = InRegisters<2>;
}
impl ArgRun for InRegisters<1> {
    type WithOne = InRegisters<2>;
    type WithPair = InRegisters<3>;
}
impl ArgRun for InRegisters<2> {
    type WithOne = InRegisters<3>;
    type WithPair = InRegisters<4>;
}
impl ArgRun for InRegisters<3> {
    type WithOne = InRegisters<4>;
    type WithPair = InWindow;
}
impl ArgRun for InRegisters<4> {
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
    REGISTER_FORM == 4,
    "the ArgRun chain covers REGISTER_FORM of the runtime's values and no other number"
);

/// The run a declaration's parameters make, as the form its call takes, the
/// site table they make — one `Arg::Site` per parameter — and the tuple the
/// body is handed.
pub trait Parameters<Rt>
where
    Rt: Runtime,
{
    type Run;
    type Sites: Clone + Send + Sync + 'static;
    /// What the Rust body's parameters are, at the call's own lifetime.
    type Out<'a>;

    const ARITY: usize;
    /// How many of the runtime's values the whole argument run is.
    const WIDTH: usize;

    fn sites(args: &[ArgAt<'_>]) -> Self::Sites;

    /// # Safety
    /// As `Arg::take`, for each parameter over its own values of `run`.
    unsafe fn take<'a>(
        rt: &'a Rt,
        run: &'a [Rt::Value],
        sites: &Self::Sites,
    ) -> <Self as Parameters<Rt>>::Out<'a>;
}

/// A parameter list of which every parameter is one of the runtime's values:
/// what a task above `Sync` admits, since a `Pair` borrows the frame the call
/// laid its arguments on (RFC-0047 §3).
pub trait ValueParameters<Rt>: Parameters<Rt>
where
    Rt: Runtime,
{
}

/// A parameter list whose site table is a constant, so a glue over it is
/// complete before any site is known and can be an `Entry`.
pub trait NoSites<Rt>: Parameters<Rt>
where
    Rt: Runtime,
{
    const SITES: Self::Sites;
}

/// A resolved instance's handler as a plain function: the values ABI of
/// `Handler::call` without the `&self` (RFC-0067 Decision 3).
pub type Entry<Rt> = for<'a, 'w> unsafe fn(
    &'a Rt,
    <Rt as Runtime>::Frame<'w>,
    &'a [<Rt as Runtime>::Value],
    &'a mut [<Rt as Runtime>::Value],
);

/// A declaration's entry, as a type: the `fn` item `#[extern_fn]` wrote
/// beside the Rust body, named where the glue's type is named so that the
/// glue itself stays the closure and nothing else.
pub trait AtEntry<Rt>: Send + Sync + 'static
where
    Rt: Runtime,
{
    const ENTRY: Option<Entry<Rt>>;
}

/// The entry of a declaration that has none.
pub struct NoEntry;

impl<Rt> AtEntry<Rt> for NoEntry
where
    Rt: Runtime,
{
    const ENTRY: Option<Entry<Rt>> = None;
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
    ($run:ty, result = $form:ty, op = $op:path, fused = $fused:path) => {
        impl TakenForm<$form> for $run {
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
    result = One,
    op = Rt::op_no_argument,
    fused = Rt::fused_no_argument
);
taken_form!(
    InRegisters<1>,
    result = One,
    op = Rt::op_one_argument,
    fused = Rt::fused_one_argument
);
taken_form!(
    InRegisters<2>,
    result = One,
    op = Rt::op_two_arguments,
    fused = Rt::fused_two_arguments
);
taken_form!(
    InRegisters<3>,
    result = One,
    op = Rt::op_three_arguments,
    fused = no_fused_run
);
taken_form!(
    InRegisters<4>,
    result = One,
    op = Rt::op_four_arguments,
    fused = no_fused_run
);
taken_form!(
    InWindow,
    result = One,
    op = Rt::op_wide,
    fused = no_fused_run
);

taken_form!(
    InRegisters<1>,
    result = Pair,
    op = Rt::op_pair_one_argument,
    fused = no_fused_pair
);
taken_form!(
    InRegisters<2>,
    result = Pair,
    op = Rt::op_pair_two_arguments,
    fused = no_fused_pair
);
taken_form!(
    InRegisters<3>,
    result = Pair,
    op = Rt::op_pair_three_arguments,
    fused = no_fused_pair
);
taken_form!(
    InRegisters<4>,
    result = Pair,
    op = Rt::op_pair_four_arguments,
    fused = no_fused_pair
);
taken_form!(
    InWindow,
    result = Pair,
    op = Rt::op_pair_wide,
    fused = no_fused_pair
);

macro_rules! taken_run_form {
    ($run:ty, op = $op:path) => {
        impl<const W: usize> TakenForm<Run<W>> for $run {
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
                no_fused_run_result(handler, shape)
            }
        }
    };
}

taken_run_form!(InRegisters<0>, op = Rt::op_run_no_argument);
taken_run_form!(InRegisters<1>, op = Rt::op_run_one_argument);
taken_run_form!(InRegisters<2>, op = Rt::op_run_two_arguments);
taken_run_form!(InRegisters<3>, op = Rt::op_run_three_arguments);
taken_run_form!(InRegisters<4>, op = Rt::op_run_four_arguments);
taken_run_form!(InWindow, op = Rt::op_run_wide);

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
    /// How many parameters the declaration has, which is how many settled
    /// types `at_site` reads. `Width::args` counts the runtime's values
    /// instead, and a `&str` or a slice parameter is two of them.
    fn arity(&self) -> usize;
    fn at_site(self: Box<Self>, args: &[ArgAt<'_>]) -> Box<dyn AtSite<Rt>>;
    /// This handler as a plain function (RFC-0067 Decision 3), for the
    /// declarations that have one.
    fn entry(&self) -> Option<Entry<Rt>>;
}

/// One declared instance with its site table filled: the handler of one call
/// site and of no other. `at_site` is the only way to reach one, so an
/// operation cannot hold a handler whose table was never filled.
pub trait AtSite<Rt>: Send + Sync
where
    Rt: Runtime,
{
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
pub trait ValuesOnly<Rt>: HandlerFactory<Rt>
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
    /// As `HandlerFactory::arity`.
    fn arity(&self) -> usize;
    fn at_site(self: Box<Self>, args: &[ArgAt<'_>]) -> Box<dyn AsyncAtSite<Rt>>;
}

/// As `AtSite`, for a declaration whose Rust body is an `async fn`.
pub trait AsyncAtSite<Rt>: Send + Sync
where
    Rt: Runtime,
{
    fn into_op(self: Box<Self>, shape: Rt::AsyncShape) -> Rt::Op;
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

fn no_fused_pair<Rt, H>(_: H, _: Rt::FusedShape) -> Rt::FusedCall
where
    Rt: Runtime,
    H: Handler<Rt>,
{
    panic!(
        "a fused run hands one value from each call to the next and holds no call whose result is a pair"
    )
}

fn no_fused_run_result<Rt, H>(_: H, _: Rt::FusedShape) -> Rt::FusedCall
where
    Rt: Runtime,
    H: Handler<Rt>,
{
    panic!(
        "a fused run hands one value from each call to the next and holds no call whose result is an aggregate's components"
    )
}

/// The site table of a glue the module table holds, which is at no site: one
/// declared instance is reached from every call site the checker settled on
/// it, and `at_site` is where a site is known.
#[derive(Clone, Copy)]
pub struct Unsited;

/// A Rust closure with the crossing on both sides of it: `A` is the tuple of
/// the declaration's parameter modes, in the order the machine lays a call's
/// arguments (RFC-0052 §7), `R` its result, and `S` the site table — one
/// `Arg::Site` per parameter, or `Unsited` before `at_site` has filled it.
///
/// `Handler` is implemented for the sited glue alone, so an operation cannot
/// hold a glue whose table was never filled.
pub struct Glue<Rt, F, A, R, S = Unsited, E = NoEntry> {
    f: F,
    sites: S,
    shape: PhantomData<fn() -> (Rt, A, R, E)>,
}

impl<Rt, F, A, R, S, E> Clone for Glue<Rt, F, A, R, S, E>
where
    F: Clone,
    S: Clone,
{
    fn clone(&self) -> Self {
        Glue {
            f: self.f.clone(),
            sites: self.sites.clone(),
            shape: PhantomData,
        }
    }
}

// SAFETY: a `Glue` holds the closure and the site table and nothing else;
// the `PhantomData` is over `fn() -> _` and carries no value.
unsafe impl<Rt, F, A, R, S, E> Send for Glue<Rt, F, A, R, S, E>
where
    F: Send,
    S: Send,
{
}
// SAFETY: as `Send`.
unsafe impl<Rt, F, A, R, S, E> Sync for Glue<Rt, F, A, R, S, E>
where
    F: Sync,
    S: Sync,
{
}

/// As `Glue`, for a body that awaits. The closure is shared because the
/// future it returns outlives the call that made it, so the call clones the
/// closure into the future rather than borrowing it.
pub struct AsyncGlue<Rt, F, A, S = Unsited> {
    f: Arc<F>,
    sites: S,
    shape: PhantomData<fn() -> (Rt, A)>,
}

impl<Rt, F, A, S> Clone for AsyncGlue<Rt, F, A, S>
where
    S: Clone,
{
    fn clone(&self) -> Self {
        AsyncGlue {
            f: Arc::clone(&self.f),
            sites: self.sites.clone(),
            shape: PhantomData,
        }
    }
}

// SAFETY: as `Glue`'s, the closure being behind a shared pointer.
unsafe impl<Rt, F, A, S> Send for AsyncGlue<Rt, F, A, S>
where
    F: Send + Sync,
    S: Send,
{
}
// SAFETY: as `Send`.
unsafe impl<Rt, F, A, S> Sync for AsyncGlue<Rt, F, A, S>
where
    F: Send + Sync,
    S: Sync,
{
}

/// One per parameter, whatever the parameter's type is: the fold that counts
/// a declaration's arity.
macro_rules! one_per {
    ($arg:ident) => {
        1usize
    };
}

/// The one value a `Site = ()` parameter's table entry has, once per
/// parameter: the fold that builds `NoSites::SITES`.
macro_rules! no_site {
    ($arg:ident) => {
        ()
    };
}

/// The run of `InRegisters<0>` widened by each parameter in turn: the fold
/// whose answer `TakenForm` reads.
macro_rules! run_of {
    ($rt:ty, $run:ty) => { $run };
    ($rt:ty, $run:ty, $arg:ident $(, $rest:ident)*) => {
        run_of!($rt, <<$arg as Arg<'static, $rt>>::Form as Form>::Onto<$run> $(, $rest)*)
    };
}

/// One parameter list: what the fold over its parameters answers, and the
/// crossing that takes each one out of a call's argument run. One more line is
/// one more parameter.
macro_rules! parameters {
    ($($arg:ident: $out:ident: $at:tt),*) => {
        impl<Rt, $($arg,)*> Parameters<Rt> for ($($arg,)*)
        where
            Rt: Runtime,
            $($arg: for<'a> Arg<'a, Rt> + 'static,)*
        {
            type Run = run_of!(Rt, InRegisters<0> $(, $arg)*);
            type Sites = ($(<$arg as Sited<Rt>>::Site,)*);
            type Out<'a> = ($(<$arg as Arg<'a, Rt>>::Out,)*);

            const ARITY: usize = 0 $(+ one_per!($arg))*;
            const WIDTH: usize = 0 $(+ <$arg as Arg<'static, Rt>>::WIDTH)*;

            #[allow(unused_variables)]
            fn sites(args: &[ArgAt<'_>]) -> Self::Sites {
                assert_eq!(
                    args.len(),
                    <Self as Parameters<Rt>>::ARITY,
                    "a call site hands {} settled argument types to a declaration of {} \
                     parameters (RFC-0059 rule 7)",
                    args.len(),
                    <Self as Parameters<Rt>>::ARITY
                );
                ($(<$arg as Sited<Rt>>::site(args[$at]),)*)
            }

            /// Obligation across artifacts: `benches/asm_probe.rs` asserts
            /// that every `Op::run` ends in the tail jump to its successor,
            /// and a call form's body inlines into the operation holding it.
            /// Left out of line, this fold is a `call` in `CallWindow` and
            /// the tail jump is gone.
            #[inline(always)]
            #[allow(unused_variables, unused_mut, unused_assignments)]
            unsafe fn take<'a>(
                rt: &'a Rt,
                run: &'a [Rt::Value],
                sites: &Self::Sites,
            ) -> <Self as Parameters<Rt>>::Out<'a> {
                let mut _at = 0usize;
                $(
                    let _width = <$arg as Arg<'_, Rt>>::WIDTH;
                    // SAFETY: the caller's contract: `run` is this
                    // declaration's whole argument run, so each parameter's
                    // own values are the next `WIDTH` of it.
                    let $out = unsafe {
                        $arg::take(rt, &run[_at.._at + _width], &sites.$at)
                    };
                    _at += _width;
                )*
                ($($out,)*)
            }
        }

        impl<Rt, $($arg,)*> ValueParameters<Rt> for ($($arg,)*)
        where
            Rt: Runtime,
            $($arg: for<'a> Arg<'a, Rt, Form = One> + 'static,)*
        {
        }

        impl<Rt, $($arg,)*> NoSites<Rt> for ($($arg,)*)
        where
            Rt: Runtime,
            $($arg: for<'a> Arg<'a, Rt> + Sited<Rt, Site = ()> + 'static,)*
        {
            const SITES: Self::Sites = ($(no_site!($arg),)*);
        }
    };
}

parameters!();
parameters!(A0: a0: 0);
parameters!(A0: a0: 0, A1: a1: 1);
parameters!(A0: a0: 0, A1: a1: 1, A2: a2: 2);
parameters!(A0: a0: 0, A1: a1: 1, A2: a2: 2, A3: a3: 3);
parameters!(A0: a0: 0, A1: a1: 1, A2: a2: 2, A3: a3: 3, A4: a4: 4);
parameters!(A0: a0: 0, A1: a1: 1, A2: a2: 2, A3: a3: 3, A4: a4: 4, A5: a5: 5);
parameters!(
    A0: a0: 0, A1: a1: 1, A2: a2: 2, A3: a3: 3, A4: a4: 4, A5: a5: 5, A6: a6: 6
);
parameters!(
    A0: a0: 0, A1: a1: 1, A2: a2: 2, A3: a3: 3, A4: a4: 4, A5: a5: 5, A6: a6: 6,
    A7: a7: 7
);

/// The constructor a declaration's glue is built by. It exists, rather than a
/// `Glue::new`, because a closure is inferred higher-ranked only where the
/// `Fn` bound is in scope at its own site.
pub fn glue<Rt, F, A, R>(f: F) -> Glue<Rt, F, A, R>
where
    Rt: Runtime,
    A: Parameters<Rt>,
    F: for<'a, 'w> Fn(&'a Rt, Rt::Frame<'w>, <A as Parameters<Rt>>::Out<'a>) -> R::Of<'a>,
    R: Ret<Rt>,
{
    Glue {
        f,
        sites: Unsited,
        shape: PhantomData,
    }
}

/// As `glue`, with the declaration's body also as a plain function: the
/// entry a resolved instance of it is passed by (RFC-0067 Decision 3).
pub fn glue_at_entry<Rt, F, A, R, E>(f: F) -> Glue<Rt, F, A, R, Unsited, E>
where
    Rt: Runtime,
    A: NoSites<Rt>,
    E: AtEntry<Rt>,
    F: for<'a, 'w> Fn(&'a Rt, Rt::Frame<'w>, <A as Parameters<Rt>>::Out<'a>) -> R::Of<'a>,
    R: Ret<Rt>,
{
    Glue {
        f,
        sites: Unsited,
        shape: PhantomData,
    }
}

/// As `glue`, for a body that awaits.
pub fn async_glue<Rt, F, A>(f: F) -> AsyncGlue<Rt, F, A>
where
    Rt: Runtime,
    A: Parameters<Rt>,
    F: for<'a, 'w> Fn(
        &'a Rt,
        &'a mut Rt::Frame<'w>,
        <A as Parameters<Rt>>::Out<'a>,
    ) -> BoxFuture<'a, Rt::Value>,
{
    AsyncGlue {
        f: Arc::new(f),
        sites: Unsited,
        shape: PhantomData,
    }
}

impl<Rt, F, A, R, E> Handler<Rt> for Glue<Rt, F, A, R, <A as Parameters<Rt>>::Sites, E>
where
    Rt: Runtime,
    E: AtEntry<Rt>,
    A: Parameters<Rt> + 'static,
    F: Clone + Send + Sync + 'static,
    F: for<'a, 'w> Fn(&'a Rt, Rt::Frame<'w>, <A as Parameters<Rt>>::Out<'a>) -> R::Of<'a>,
    R: Ret<Rt> + 'static,
{
    const WIDTH: Width = Width {
        args: <A as Parameters<Rt>>::WIDTH,
        ret: <R as Ret<Rt>>::WIDTH,
        result: <<R as Ret<Rt>>::Form as Form>::KIND,
    };

    unsafe fn call(&self, rt: &Rt, frame: Rt::Frame<'_>, run: &[Rt::Value], out: &mut [Rt::Value]) {
        // SAFETY: the caller's contract, which is `Parameters::take`'s.
        let args = unsafe { <A as Parameters<Rt>>::take(rt, run, &self.sites) };
        <R as Ret<Rt>>::into_run((self.f)(rt, frame, args), rt, out)
    }
}

impl<Rt, F, A, R, E> HandlerFactory<Rt> for Glue<Rt, F, A, R, Unsited, E>
where
    Rt: Runtime,
    E: AtEntry<Rt>,
    A: Parameters<Rt> + 'static,
    F: Clone + Send + Sync + 'static,
    F: for<'a, 'w> Fn(&'a Rt, Rt::Frame<'w>, <A as Parameters<Rt>>::Out<'a>) -> R::Of<'a>,
    R: Ret<Rt> + 'static,
    <A as Parameters<Rt>>::Run: TakenForm<<R as Ret<Rt>>::Form>,
{
    fn clone_box(&self) -> Box<dyn HandlerFactory<Rt>> {
        Box::new(self.clone())
    }

    fn width(&self) -> Width {
        Width {
            args: <A as Parameters<Rt>>::WIDTH,
            ret: <R as Ret<Rt>>::WIDTH,
            result: <<R as Ret<Rt>>::Form as Form>::KIND,
        }
    }

    fn arity(&self) -> usize {
        <A as Parameters<Rt>>::ARITY
    }

    fn at_site(self: Box<Self>, args: &[ArgAt<'_>]) -> Box<dyn AtSite<Rt>> {
        Box::new(self.at(args))
    }

    fn entry(&self) -> Option<Entry<Rt>> {
        <E as AtEntry<Rt>>::ENTRY
    }
}

impl<Rt, F, A, R, E> Glue<Rt, F, A, R, Unsited, E>
where
    Rt: Runtime,
    A: Parameters<Rt>,
{
    pub fn at(self, args: &[ArgAt<'_>]) -> Glue<Rt, F, A, R, <A as Parameters<Rt>>::Sites, E> {
        Glue {
            f: self.f,
            sites: <A as Parameters<Rt>>::sites(args),
            shape: PhantomData,
        }
    }
}

impl<Rt, F, A, R, E> AtSite<Rt> for Glue<Rt, F, A, R, <A as Parameters<Rt>>::Sites, E>
where
    Rt: Runtime,
    E: AtEntry<Rt>,
    A: Parameters<Rt> + 'static,
    F: Clone + Send + Sync + 'static,
    F: for<'a, 'w> Fn(&'a Rt, Rt::Frame<'w>, <A as Parameters<Rt>>::Out<'a>) -> R::Of<'a>,
    R: Ret<Rt> + 'static,
    <A as Parameters<Rt>>::Run: TakenForm<<R as Ret<Rt>>::Form>,
{
    fn into_op(self: Box<Self>, shape: Rt::CallShape) -> Rt::Op {
        <<A as Parameters<Rt>>::Run as TakenForm<<R as Ret<Rt>>::Form>>::op::<Rt, Self>(
            *self, shape,
        )
    }

    fn into_fused(self: Box<Self>, shape: Rt::FusedShape) -> Rt::FusedCall {
        <<A as Parameters<Rt>>::Run as TakenForm<<R as Ret<Rt>>::Form>>::fused::<Rt, Self>(
            *self, shape,
        )
    }
}

impl<Rt, F, A, R, E> ValuesOnly<Rt> for Glue<Rt, F, A, R, Unsited, E>
where
    Rt: Runtime,
    E: AtEntry<Rt>,
    A: ValueParameters<Rt> + 'static,
    F: Clone + Send + Sync + 'static,
    F: for<'a, 'w> Fn(&'a Rt, Rt::Frame<'w>, <A as Parameters<Rt>>::Out<'a>) -> R::Of<'a>,
    R: Ret<Rt, Form = One> + 'static,
    <A as Parameters<Rt>>::Run: TakenForm<<R as Ret<Rt>>::Form>,
{
}

impl<Rt, F, A> AsyncCall<Rt> for AsyncGlue<Rt, F, A, <A as Parameters<Rt>>::Sites>
where
    Rt: Runtime,
    A: ValueParameters<Rt> + 'static,
    F: Send + Sync + 'static,
    F: for<'a, 'w> Fn(
        &'a Rt,
        &'a mut Rt::Frame<'w>,
        <A as Parameters<Rt>>::Out<'a>,
    ) -> BoxFuture<'a, Rt::Value>,
{
    const WIDTH: Width = Width {
        args: <A as Parameters<Rt>>::WIDTH,
        ret: 1,
        result: FormKind::Value,
    };

    unsafe fn call(&self, rt: Rt, run: &[Rt::Value]) -> BoxFuture<'static, Rt::Value> {
        let held: Vec<Rt::Value> = run.to_vec();
        let f = Arc::clone(&self.f);
        let sites = self.sites.clone();
        Box::pin(async move {
            let mut rooted = rt.rooted();
            let mut frame = Rt::frame_of(&mut rooted);
            // SAFETY: as the synchronous impl's, over the run the future owns.
            let args = unsafe { <A as Parameters<Rt>>::take(&rt, &held, &sites) };
            f(&rt, &mut frame, args).await
        })
    }
}

impl<Rt, F, A> AsyncFactory<Rt> for AsyncGlue<Rt, F, A>
where
    Rt: Runtime,
    A: ValueParameters<Rt> + 'static,
    F: Send + Sync + 'static,
    F: for<'a, 'w> Fn(
        &'a Rt,
        &'a mut Rt::Frame<'w>,
        <A as Parameters<Rt>>::Out<'a>,
    ) -> BoxFuture<'a, Rt::Value>,
{
    fn clone_box(&self) -> Box<dyn AsyncFactory<Rt>> {
        Box::new(self.clone())
    }

    fn width(&self) -> Width {
        Width {
            args: <A as Parameters<Rt>>::WIDTH,
            ret: 1,
            result: FormKind::Value,
        }
    }

    fn arity(&self) -> usize {
        <A as Parameters<Rt>>::ARITY
    }

    fn at_site(self: Box<Self>, args: &[ArgAt<'_>]) -> Box<dyn AsyncAtSite<Rt>> {
        Box::new(self.at(args))
    }
}

impl<Rt, F, A> AsyncGlue<Rt, F, A>
where
    Rt: Runtime,
    A: Parameters<Rt>,
{
    pub fn at(self, args: &[ArgAt<'_>]) -> AsyncGlue<Rt, F, A, <A as Parameters<Rt>>::Sites> {
        AsyncGlue {
            f: self.f,
            sites: <A as Parameters<Rt>>::sites(args),
            shape: PhantomData,
        }
    }
}

impl<Rt, F, A> AsyncAtSite<Rt> for AsyncGlue<Rt, F, A, <A as Parameters<Rt>>::Sites>
where
    Rt: Runtime,
    A: ValueParameters<Rt> + 'static,
    F: Send + Sync + 'static,
    F: for<'a, 'w> Fn(
        &'a Rt,
        &'a mut Rt::Frame<'w>,
        <A as Parameters<Rt>>::Out<'a>,
    ) -> BoxFuture<'a, Rt::Value>,
{
    fn into_op(self: Box<Self>, shape: Rt::AsyncShape) -> Rt::Op {
        Rt::async_extern_op(*self, shape)
    }
}

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

    /// This instance's entry (RFC-0067 Decision 3). A `Heavy` or `Async`
    /// handler has none: the glue that runs one suspends the caller, and an
    /// entry is called where it stands.
    pub fn entry(&self) -> Option<Entry<R>> {
        match self {
            Self::Sync(f) => f.entry(),
            Self::Heavy(_) | Self::Async(_) => None,
        }
    }
}

pub struct DeclaredInstance<R: Runtime> {
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
    pub concrete: Vec<DeclaredInstance<R>>,
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
    pub fn add_concrete(&mut self, more: Vec<DeclaredInstance<R>>) {
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

/// The entries of a host that runs a call where it stands: every form
/// builds the same `DirectOp`, because such a host lays no arguments in
/// registers and so takes every form the same way.
#[macro_export]
macro_rules! direct_call_forms {
    () => {
        $crate::direct_call_forms!(@op op_no_argument);
        $crate::direct_call_forms!(@op op_one_argument);
        $crate::direct_call_forms!(@op op_two_arguments);
        $crate::direct_call_forms!(@op op_three_arguments);
        $crate::direct_call_forms!(@op op_four_arguments);
        $crate::direct_call_forms!(@op op_wide);
        $crate::direct_call_forms!(@op op_pair_one_argument);
        $crate::direct_call_forms!(@op op_pair_two_arguments);
        $crate::direct_call_forms!(@op op_pair_three_arguments);
        $crate::direct_call_forms!(@op op_pair_four_arguments);
        $crate::direct_call_forms!(@op op_pair_wide);
        $crate::direct_call_forms!(@op op_run_no_argument);
        $crate::direct_call_forms!(@op op_run_one_argument);
        $crate::direct_call_forms!(@op op_run_two_arguments);
        $crate::direct_call_forms!(@op op_run_three_arguments);
        $crate::direct_call_forms!(@op op_run_four_arguments);
        $crate::direct_call_forms!(@op op_run_wide);
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
