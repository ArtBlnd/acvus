//! Typed handlers, ready for a runtime to call (RFC-0059).
//!
//! A declaration's Rust body is reached through a closure the proc macro
//! writes, wrapped in a `Glue` whose type parameters say how each parameter
//! comes out of the call's argument run and how the result is written back.
//! Every fact the ABI needs is a constant of those types, summed once in the
//! `Handler` impl and read back through `Handler::width`.

use std::marker::PhantomData;
use std::sync::Arc;

use acvus_mir::laws::{Laws, Postcondition};
use acvus_mir::ty::{EffectVarBound, PolyTy, RequirementSig, Task, Ty};
use acvus_utils::{Interner, QualifiedRef};
use futures::future::BoxFuture;

use crate::ctx::Ctx;
use crate::instance::InstanceRun;
use crate::instance::{Instance, Signature};
use crate::loan::Loan;
use crate::obj::{
    Cross, Form, FormKind, Nothing, One, OneRegister, OneValue, OptionOf, Returned,
    SurvivesSuspension,
};
use crate::runtime::Runtime;

/// Which instance of a required signature a call site runs: its position
/// among that signature's own instances, as `Instances` numbers them. This
/// is what `prepare` hands `InstanceEntries::glue` to build an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequiredInstance(pub usize);

pub trait InstanceEntries<Rt>: Send + Sync
where
    Rt: Runtime,
{
    /// The glue of the instance the checker settled on.
    fn glue(&self, signature: QualifiedRef, instance: RequiredInstance) -> InstanceRun;
}

pub struct NoInstances;

impl<Rt> InstanceEntries<Rt> for NoInstances
where
    Rt: Runtime,
{
    fn glue(&self, _: QualifiedRef, _: RequiredInstance) -> InstanceRun {
        panic!(
            "this call site was built with no registry, so a required instance cannot be resolved (RFC-0070 rule 2)"
        )
    }
}

#[derive(Clone, Copy)]
pub struct ArgAt<'a> {
    pub interner: &'a Interner,
    pub ty: &'a Ty,
}

/// One call site as `prepare` hands it to a handler: the settled type of
/// each argument, and the word of the entry `prepare` chose for each
/// requirement the callee states, in the declaration's order (RFC-0070 rule 2).
///
/// The fields are private because `Required::site` makes a callable
/// `Instance` out of a `requires` word: only `new`, which is `unsafe`,
/// puts a word there, and `of_args` puts none.
pub struct CallSite<'a, Rt>
where
    Rt: Runtime,
{
    pub(crate) args: &'a [ArgAt<'a>],
    pub(crate) requires: &'a [Rt::Value],
}

impl<'a, Rt> CallSite<'a, Rt>
where
    Rt: Runtime,
{
    /// A site whose callee states no requirement.
    pub fn of_args(args: &'a [ArgAt<'a>]) -> Self {
        CallSite {
            args,
            requires: &[],
        }
    }

    /// A site whose callee states requirements, with the checker's answer
    /// for each.
    ///
    /// # Safety
    /// The site is handed only to the declaration whose requirements
    /// `requires` answers: `requires[n]` was made by
    /// `Runtime::instance_value` from an entry that outlives every handler
    /// sited here, and that entry is an instance of the `n`th requirement's
    /// signature at the type its variable is filled with at this site.
    pub unsafe fn new(args: &'a [ArgAt<'a>], requires: &'a [Rt::Value]) -> Self {
        CallSite { args, requires }
    }
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

/// Which crossing a parameter or a result takes (RFC-0041).
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

    /// How many of the call's settled argument types this parameter takes.
    /// A required instance takes none: the site table is where its word is.
    const ARGUMENTS: usize = 1;

    fn site(site: &CallSite<'_, Rt>, at: usize) -> Self::Site;
}

/// How one Rust parameter takes its argument out of a call's argument run.
/// The mode — by value, by shared reference, by exclusive reference — is
/// written in Rust and read by the macro (RFC-0023 rule 5); the width is the
/// type's. `'a` is the call and `'w` the lifetime of the call's `Ctx`: a
/// carrier is handed out at the first, a required instance at the second
/// (RFC-0079 rule 6).
pub trait Arg<'a, 'w, Rt>: Sited<Rt>
where
    Rt: Runtime,
{
    /// What the closure's parameter is.
    type Out;
    /// The run this parameter takes out of the call's argument run. A bound
    /// that admits only a parameter which survives the caller suspending
    /// says `Form = One`: the `Pair` a slice is borrows the caller's frame
    /// (RFC-0047 rule 6).
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

    fn site(_: &CallSite<'_, Rt>, _: usize) {}
}

impl<T, Rt> Sited<Rt> for ByValue<T, Specialized>
where
    T: OneValue<Rt, Specialized>,
    Rt: Runtime,
{
    type Site = ();

    fn site(_: &CallSite<'_, Rt>, _: usize) {}
}

impl<T, M, Rt> Sited<Rt> for ByRef<T, M, Uniform>
where
    T: Borrowable<Rt>,
    M: Loan,
    Rt: Runtime,
{
    type Site = ();

    fn site(_: &CallSite<'_, Rt>, _: usize) {}
}

impl<T, M, Rt> Sited<Rt> for ByRef<T, M, Specialized>
where
    T: BorrowableSpecialized<Rt>,
    M: Loan,
    Rt: Runtime,
{
    type Site = ();

    fn site(_: &CallSite<'_, Rt>, _: usize) {}
}

/// A parameter that is the declaration's `NTH` required instance: the call
/// site holds the checker's answer for it, and the run carries nothing.
pub struct Required<S, I, T, const NTH: usize>(PhantomData<fn() -> (S, I, T)>);

impl<S, I, T, Rt, const NTH: usize> Sited<Rt> for Required<S, I, T, NTH>
where
    S: Signature<Rt>,
    I: Send + Sync + 'static,
    T: Send + Sync + 'static,
    Rt: Runtime,
{
    type Site = Instance<'static, S, I, Rt, T>;

    const ARGUMENTS: usize = 0;

    fn site(site: &CallSite<'_, Rt>, _: usize) -> Instance<'static, S, I, Rt, T> {
        // SAFETY: `CallSite::new`'s contract, the one way a word reaches
        // `requires`: the word is the entry chosen for this site's `NTH`
        // requirement, whose signature is `S` and whose type is what the
        // requirement's variable is filled with here, and the entry
        // outlives every handler sited here.
        unsafe { Instance::at(site.requires[NTH]) }
    }
}

impl<'a, 'w, S, I, T, Rt, const NTH: usize> Arg<'a, 'w, Rt> for Required<S, I, T, NTH>
where
    S: Signature<Rt>,
    I: Send + Sync + 'static,
    T: Send + Sync + 'static,
    Rt: Runtime,
{
    type Out = Instance<'w, S, I, Rt, T>;
    type Form = Nothing;

    unsafe fn take<'s>(
        _: &'a Rt,
        _: &'a [Rt::Value],
        site: &'s Instance<'static, S, I, Rt, T>,
    ) -> Instance<'w, S, I, Rt, T> {
        *site
    }
}

impl<'a, 'w, T, Rt> Arg<'a, 'w, Rt> for ByValue<T, Uniform>
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    type Out = T::At<'a>;
    type Form = <T as Cross<Rt>>::Form;

    unsafe fn take<'s>(rt: &'a Rt, run: &'a [Rt::Value], _: &'s ()) -> T::At<'a> {
        // SAFETY: the caller's contract, which is `Cross::from_run`'s, and
        // what the value names is live for the call (RFC-0018).
        unsafe { crate::brand::<T>(<T as Cross<Rt>>::from_run(rt, run)) }
    }
}

impl<'a, 'w, T, Rt> Arg<'a, 'w, Rt> for ByValue<T, Specialized>
where
    T: OneValue<Rt, Specialized>,
    Rt: Runtime,
{
    type Out = T::At<'a>;
    type Form = One;

    unsafe fn take<'s>(rt: &'a Rt, run: &'a [Rt::Value], _: &'s ()) -> T::At<'a> {
        // SAFETY: as the uniform impl's.
        unsafe { crate::brand::<T>(<T as OneValue<Rt, Specialized>>::from_run(rt, run)) }
    }
}

impl<'a, 'w, T, M, Rt> Arg<'a, 'w, Rt> for ByRef<T, M, Uniform>
where
    T: Borrowable<Rt>,
    M: Loan,
    Rt: Runtime,
{
    type Out = M::Of<'a, T::At<'a>>;
    type Form = One;

    unsafe fn take<'s>(rt: &'a Rt, run: &'a [Rt::Value], _: &'s ()) -> M::Of<'a, T::At<'a>> {
        // SAFETY: the caller's contract: a live storage of `T`, exclusively
        // named at a `Mut` loan (RFC-0018).
        unsafe { M::brand::<T>(M::borrow::<T, Uniform, Rt>(rt, &run[0])) }
    }
}

impl<'a, 'w, T, M, Rt> Arg<'a, 'w, Rt> for ByRef<T, M, Specialized>
where
    T: BorrowableSpecialized<Rt>,
    M: Loan,
    Rt: Runtime,
{
    type Out = M::Of<'a, T::At<'a>>;
    type Form = One;

    unsafe fn take<'s>(rt: &'a Rt, run: &'a [Rt::Value], _: &'s ()) -> M::Of<'a, T::At<'a>> {
        // SAFETY: as the uniform impl's, at the specialized representation.
        unsafe { M::brand::<T>(M::borrow::<T, Specialized, Rt>(rt, &run[0])) }
    }
}

/// A type a parameter may take by reference: one whose values are places the
/// language names. An `Option` is not — `None` is one value and `Some(v)` is
/// `v`'s own value (RFC-0039) — and the missing impl is the refusal.
///
/// The in-place read is this trait's and not `OneValue`'s: a type whose
/// storage holds no `Self`, such as a derived struct kept as an object or a
/// `Vec<i64>` kept as a `Vec<Owned<Rt>>`, has no impl, so no reader reaches
/// its storage as a `Self`.
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
    /// # Safety
    /// `reference` names a live storage of `Self`.
    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a Self;

    /// # Safety
    /// As `deref`, and the reference is the only live name of the storage.
    #[allow(clippy::mut_from_ref)]
    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut Self;
}

/// A type a parameter of a `Monomorphize` member may take by reference: one
/// whose specialized crossing writes a `Self` into the storage the caller
/// lends, which is what `deref` reads back.
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
    /// # Safety
    /// As `Borrowable::deref`, at the specialized representation.
    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a Self;

    /// # Safety
    /// As `Borrowable::deref_mut`, at the specialized representation.
    #[allow(clippy::mut_from_ref)]
    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut Self;
}

/// A representation that lends a `T` in place: `Uniform` where `T` is
/// `Borrowable`, `Specialized` where it is `BorrowableSpecialized`. A reader
/// written once over the representation, as `Loan::borrow` and the
/// `Restore*` impls are, asks this and so asks each representation's own
/// bound.
pub trait Lends<T, Rt>
where
    Rt: Runtime,
{
    /// # Safety
    /// As `Borrowable::deref`.
    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a T;

    /// # Safety
    /// As `Borrowable::deref_mut`.
    #[allow(clippy::mut_from_ref)]
    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut T;
}

impl<T, Rt> Lends<T, Rt> for Uniform
where
    T: Borrowable<Rt>,
    Rt: Runtime,
{
    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a T {
        // SAFETY: the caller's contract.
        unsafe { <T as Borrowable<Rt>>::deref(rt, reference) }
    }

    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut T {
        // SAFETY: the caller's contract.
        unsafe { <T as Borrowable<Rt>>::deref_mut(rt, reference) }
    }
}

impl<T, Rt> Lends<T, Rt> for Specialized
where
    T: BorrowableSpecialized<Rt>,
    Rt: Runtime,
{
    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a T {
        // SAFETY: the caller's contract.
        unsafe { <T as BorrowableSpecialized<Rt>>::deref(rt, reference) }
    }

    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut T {
        // SAFETY: the caller's contract.
        unsafe { <T as BorrowableSpecialized<Rt>>::deref_mut(rt, reference) }
    }
}

/// The destination run a call's result is written into, lent for the call's
/// duration by the frame that owns it (RFC-0050 rule 5). It is the caller's
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
    /// slice is borrows the caller's frame (RFC-0047 rule 3), and the `Run<W>` an
    /// aggregate is names the caller's destination.
    type Form: Returned;

    /// How many of the runtime's values the result occupies.
    const WIDTH: usize = <Self::Form as Form>::WIDTH;

    fn into_run(
        value: Self::Of<'_>,
        rt: &Rt,
        out: Out<'_, Rt>,
    ) -> <Self::Form as Returned>::Verdict;
}

/// The arguments of a closure call, written into the callee's parameter
/// registers — the run the window a handler was lent begins with (RFC-0052
/// rule 7). Each member crosses at its own width, as a result does through `Ret`.
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

/// A result that is a borrow of a parameter the caller lent, at the carrier
/// type the declaration names for it (RFC-0047 rule 3, RFC-0068 rule 4): `Ref`,
/// `Slice`, and an `Option` of one. The handler returns Rust's borrow with
/// Rust's lifetime; the crossing writes the word or pair.
pub trait LentBack<Rt>: Sized
where
    Rt: Runtime,
{
    type Of<'a>;
    type Form: Returned;

    fn into_run(
        value: Self::Of<'_>,
        rt: &Rt,
        out: Out<'_, Rt>,
    ) -> <Self::Form as Returned>::Verdict;
}

impl<L, Rt> LentBack<Rt> for Option<L>
where
    L: LentBack<Rt, Form = One>,
    Rt: Runtime,
{
    type Of<'a> = Option<L::Of<'a>>;
    type Form = OptionOf<One>;

    fn into_run(value: Option<L::Of<'_>>, rt: &Rt, out: Out<'_, Rt>) -> bool {
        let Some(lent) = value else {
            return false;
        };
        L::into_run(lent, rt, out);
        true
    }
}

/// The macro emits this where `Val` would stand for an owned result, for a
/// result written as a Rust borrow.
pub struct RetLent<L>(PhantomData<fn() -> L>);

impl<L, Rt> Ret<Rt> for RetLent<L>
where
    L: LentBack<Rt>,
    Rt: Runtime,
{
    type Of<'a> = L::Of<'a>;
    type Form = L::Form;

    fn into_run(value: L::Of<'_>, rt: &Rt, out: Out<'_, Rt>) -> <L::Form as Returned>::Verdict {
        L::into_run(value, rt, out)
    }
}

/// A result crossing as itself, at whichever width its type declares.
pub struct Val<T, C = Uniform>(PhantomData<fn() -> (T, C)>);

impl<T, Rt> Ret<Rt> for Val<T, Uniform>
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    type Of<'a> = T::At<'a>;
    type Form = <T as Cross<Rt>>::ReturnForm;

    fn into_run(
        value: T::At<'_>,
        rt: &Rt,
        out: Out<'_, Rt>,
    ) -> <<T as Cross<Rt>>::ReturnForm as Returned>::Verdict {
        // SAFETY: the value is erased here, before safe code sees it again.
        <T as Cross<Rt>>::into_return_run(unsafe { crate::unbrand::<T>(value) }, rt, out)
    }
}

impl<T, Rt> Ret<Rt> for Val<T, Specialized>
where
    T: OneValue<Rt, Specialized>,
    Rt: Runtime,
{
    type Of<'a> = T::At<'a>;
    /// The specialized crossing is the value and nothing beside it, so there
    /// is no run for a verdict to be returned beside and no `OptionOf` here
    /// (RFC-0041).
    type Form = One;

    fn into_run(value: T::At<'_>, rt: &Rt, out: Out<'_, Rt>) {
        // SAFETY: as the uniform impl's.
        <T as OneValue<Rt, Specialized>>::into_run(unsafe { crate::unbrand::<T>(value) }, rt, out)
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
    /// `Returned::ABSENT` of the result form, which is the one fact about
    /// the result a caller holding no type of the handler cannot project.
    pub absent: bool,
}

/// The widest argument run a register form covers: a call of this many of the
/// runtime's values or fewer takes each in a register, and a wider one is lent
/// its window. RFC-0044 rule 3 fixed the cut at three parameters, which with
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
    /// The run this call's arguments are, as a type: which of the six forms,
    /// and at a register form the constant length of the array the entry
    /// takes.
    type Args: ArgRun;
    /// The run its result is written into, at that form's own width.
    type Ret: Returned;

    const WIDTH: Width;

    /// `frame` is the window above the calling frame, which a handler that
    /// calls a closure calls it in and a handler that calls none ignores
    /// (RFC-0052 rule 7).
    ///
    /// # Safety
    /// Every reference the handler takes out of `run` names storage live for
    /// the call (RFC-0018). The two lengths `WIDTH` states are the two runs'
    /// types, so neither is the caller's to get right.
    unsafe fn call(
        &self,
        ctx: &mut Ctx<'_, Rt>,
        run: <Self::Args as ArgRun>::Run<'_, Rt>,
        out: <Self::Ret as Returned>::Out<'_, Rt>,
    ) -> <Self::Ret as Returned>::Verdict;
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

    /// The run a call of this form is handed: the array a register form's
    /// width names, or the window itself.
    type Run<'a, Rt>: Copy
    where
        Rt: Runtime;

    fn as_slice<'a, Rt>(run: Self::Run<'a, Rt>) -> &'a [Rt::Value]
    where
        Rt: Runtime + 'a;

    /// # Safety
    /// `run.len()` is this form's width — the dyn crossing (`DirectOp`).
    unsafe fn from_slice<'a, Rt>(run: &'a [Rt::Value]) -> Self::Run<'a, Rt>
    where
        Rt: Runtime + 'a;

    /// The type-level match: hands `f` to the one method of `forms` this
    /// form names.
    fn select<Rt, F, H>(forms: F, f: H) -> F::Out
    where
        Rt: Runtime,
        F: CallForms<Rt>,
        H: Handler<Rt, Args = Self>;
}

/// The six argument runs, as methods, for a caller that builds one thing per
/// form. `ArgRun::select` is what picks the method, so a caller writes six
/// arms and no match.
pub trait CallForms<Rt>
where
    Rt: Runtime,
{
    type Out;

    fn registers0<H>(self, f: H) -> Self::Out
    where
        H: Handler<Rt, Args = InRegisters<0>>;
    fn registers1<H>(self, f: H) -> Self::Out
    where
        H: Handler<Rt, Args = InRegisters<1>>;
    fn registers2<H>(self, f: H) -> Self::Out
    where
        H: Handler<Rt, Args = InRegisters<2>>;
    fn registers3<H>(self, f: H) -> Self::Out
    where
        H: Handler<Rt, Args = InRegisters<3>>;
    fn registers4<H>(self, f: H) -> Self::Out
    where
        H: Handler<Rt, Args = InRegisters<4>>;
    fn window<H>(self, f: H) -> Self::Out
    where
        H: Handler<Rt, Args = InWindow>;
}

/// One register form: the next two forms a parameter widens it to, and the
/// array of its own length that a call of it is handed.
macro_rules! in_registers {
    ($n:literal, $with_one:ty, $with_pair:ty, $select:ident) => {
        impl ArgRun for InRegisters<$n> {
            type WithOne = $with_one;
            type WithPair = $with_pair;

            type Run<'a, Rt>
                = &'a [Rt::Value; $n]
            where
                Rt: Runtime;

            fn as_slice<'a, Rt>(run: &'a [Rt::Value; $n]) -> &'a [Rt::Value]
            where
                Rt: Runtime + 'a,
            {
                run
            }

            unsafe fn from_slice<'a, Rt>(run: &'a [Rt::Value]) -> &'a [Rt::Value; $n]
            where
                Rt: Runtime + 'a,
            {
                debug_assert_eq!(run.len(), $n, "an argument run is this form's own width");
                // SAFETY: the caller's contract: `run` is `$n` long, and a
                // `[T; N]` is `N` `T`s with no other requirement.
                unsafe { &*(run.as_ptr() as *const [Rt::Value; $n]) }
            }

            fn select<Rt, F, H>(forms: F, f: H) -> F::Out
            where
                Rt: Runtime,
                F: CallForms<Rt>,
                H: Handler<Rt, Args = Self>,
            {
                forms.$select(f)
            }
        }
    };
}

in_registers!(0, InRegisters<1>, InRegisters<2>, registers0);
in_registers!(1, InRegisters<2>, InRegisters<3>, registers1);
in_registers!(2, InRegisters<3>, InRegisters<4>, registers2);
in_registers!(3, InRegisters<4>, InWindow, registers3);
in_registers!(4, InWindow, InWindow, registers4);

impl ArgRun for InWindow {
    type WithOne = InWindow;
    type WithPair = InWindow;

    type Run<'a, Rt>
        = &'a [Rt::Value]
    where
        Rt: Runtime;

    fn as_slice<'a, Rt>(run: &'a [Rt::Value]) -> &'a [Rt::Value]
    where
        Rt: Runtime + 'a,
    {
        run
    }

    unsafe fn from_slice<'a, Rt>(run: &'a [Rt::Value]) -> &'a [Rt::Value]
    where
        Rt: Runtime + 'a,
    {
        run
    }

    fn select<Rt, F, H>(forms: F, f: H) -> F::Out
    where
        Rt: Runtime,
        F: CallForms<Rt>,
        H: Handler<Rt, Args = Self>,
    {
        forms.window(f)
    }
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
    type Run: ArgRun;
    type Sites: Clone + Send + Sync + 'static;
    /// What the Rust body's parameters are, at the call's own lifetime and
    /// at its `Ctx`'s.
    type Out<'a, 'w>;

    const ARITY: usize;
    /// How many of the runtime's values the whole argument run is.
    const WIDTH: usize;

    fn sites(site: &CallSite<'_, Rt>) -> Self::Sites;

    /// # Safety
    /// As `Arg::take`, for each parameter over its own values of `run`.
    unsafe fn take<'a, 'w>(
        rt: &'a Rt,
        run: &'a [Rt::Value],
        sites: &Self::Sites,
    ) -> <Self as Parameters<Rt>>::Out<'a, 'w>;
}

/// A parameter list every parameter of which survives the caller
/// suspending: what a task above `Sync` admits.
pub trait ValueParameters<Rt>: Parameters<Rt>
where
    Rt: Runtime,
{
}

/// A declaration's mono glue, as a type: the `fn` item `#[extern_fn]` wrote
/// beside the Rust body, named where the glue's type is named so that the
/// glue itself stays the closure and nothing else.
pub trait AtInstance<Rt>: Send + Sync + 'static
where
    Rt: Runtime,
{
    fn run() -> Option<InstanceRun>;
}

pub struct NoInstance;

impl<Rt> AtInstance<Rt> for NoInstance
where
    Rt: Runtime,
{
    fn run() -> Option<InstanceRun> {
        None
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
    /// How many parameters the declaration has, which is how many settled
    /// types `at_site` reads. `Width::args` counts the runtime's values
    /// instead, and a `&str` or a slice parameter is two of them.
    fn arity(&self) -> usize;
    fn at_site(self: Box<Self>, site: &CallSite<'_, Rt>) -> Box<dyn AtSite<Rt>>;
    /// This handler as the plain function an `Instance` value names
    /// (RFC-0067 rule 8), for the declarations that have one.
    fn instance(&self) -> Option<InstanceRun>;
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
/// resumes (RFC-0047 rule 6, RFC-0023 rule 6).
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
    fn at_site(self: Box<Self>, site: &CallSite<'_, Rt>) -> Box<dyn AsyncAtSite<Rt>>;
    /// As `HandlerFactory::instance`.
    fn instance(&self) -> Option<InstanceRun>;
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

/// The site table of a glue the module table holds, which is at no site: one
/// declared instance is reached from every call site the checker settled on
/// it, and `at_site` is where a site is known.
#[derive(Clone, Copy)]
pub struct Unsited;

/// A Rust closure with the crossing on both sides of it: `A` is the tuple of
/// the declaration's parameter modes, in the order the machine lays a call's
/// arguments (RFC-0052 rule 7), `R` its result, and `S` the site table — one
/// `Arg::Site` per parameter, or `Unsited` before `at_site` has filled it.
///
/// `Handler` is implemented for the sited glue alone, so an operation cannot
/// hold a glue whose table was never filled.
pub struct Glue<Rt, F, A, R, S = Unsited, E = NoInstance> {
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
pub struct AsyncGlue<Rt, F, A, S = Unsited, E = NoInstance> {
    f: Arc<F>,
    sites: S,
    shape: PhantomData<fn() -> (Rt, A, E)>,
}

impl<Rt, F, A, S, E> Clone for AsyncGlue<Rt, F, A, S, E>
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
unsafe impl<Rt, F, A, S, E> Send for AsyncGlue<Rt, F, A, S, E>
where
    F: Send + Sync,
    S: Send,
{
}
// SAFETY: as `Send`.
unsafe impl<Rt, F, A, S, E> Sync for AsyncGlue<Rt, F, A, S, E>
where
    F: Send + Sync,
    S: Sync,
{
}

/// The run of `InRegisters<0>` widened by each parameter in turn: the fold
/// whose answer is `Parameters::Run`, and `Handler::WIDTH.args` its width.
macro_rules! run_of {
    ($rt:ty, $run:ty) => { $run };
    ($rt:ty, $run:ty, $arg:ident $(, $rest:ident)*) => {
        run_of!($rt, <<$arg as Arg<'static, 'static, $rt>>::Form as Form>::Onto<$run> $(, $rest)*)
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
            $($arg: for<'a, 'w> Arg<'a, 'w, Rt> + 'static,)*
        {
            type Run = run_of!(Rt, InRegisters<0> $(, $arg)*);
            type Sites = ($(<$arg as Sited<Rt>>::Site,)*);
            type Out<'a, 'w> = ($(<$arg as Arg<'a, 'w, Rt>>::Out,)*);

            const ARITY: usize = 0 $(+ <$arg as Sited<Rt>>::ARGUMENTS)*;
            const WIDTH: usize = 0 $(+ <$arg as Arg<'static, 'static, Rt>>::WIDTH)*;

            #[allow(unused_variables, unused_mut, unused_assignments)]
            fn sites(site: &CallSite<'_, Rt>) -> Self::Sites {
                assert_eq!(
                    site.args.len(),
                    <Self as Parameters<Rt>>::ARITY,
                    "a call site hands {} settled argument types to a declaration of {} \
                     parameters (RFC-0059 rule 7)",
                    site.args.len(),
                    <Self as Parameters<Rt>>::ARITY
                );
                let mut _at = 0usize;
                $(
                    let $out = <$arg as Sited<Rt>>::site(site, _at);
                    _at += <$arg as Sited<Rt>>::ARGUMENTS;
                )*
                ($($out,)*)
            }

            /// Obligation across artifacts: `benches/asm_probe.rs` asserts
            /// that every `Op::run` ends in the tail jump to its successor,
            /// and a call form's body inlines into the operation holding it.
            /// Left out of line, this fold is a `call` in `CallWindow` and
            /// the tail jump is gone.
            #[inline(always)]
            #[allow(unused_variables, unused_mut, unused_assignments)]
            unsafe fn take<'a, 'w>(
                rt: &'a Rt,
                run: &'a [Rt::Value],
                sites: &Self::Sites,
            ) -> <Self as Parameters<Rt>>::Out<'a, 'w> {
                let mut _at = 0usize;
                $(
                    let _width = <$arg as Arg<'a, 'w, Rt>>::WIDTH;
                    // SAFETY: the caller's contract: `run` is this
                    // declaration's whole argument run, so each parameter's
                    // own values are the next `WIDTH` of it.
                    let $out = unsafe {
                        <$arg as Arg<'a, 'w, Rt>>::take(rt, &run[_at.._at + _width], &sites.$at)
                    };
                    _at += _width;
                )*
                ($($out,)*)
            }
        }

        impl<Rt, $($arg,)*> ValueParameters<Rt> for ($($arg,)*)
        where
            Rt: Runtime,
            $($arg: for<'a, 'w> Arg<'a, 'w, Rt, Form: SurvivesSuspension> + 'static,)*
        {
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
    F: for<'a, 'w> Fn(&'a mut Ctx<'w, Rt>, <A as Parameters<Rt>>::Out<'a, 'w>) -> R::Of<'a>,
    R: Ret<Rt>,
{
    Glue {
        f,
        sites: Unsited,
        shape: PhantomData,
    }
}

/// As `glue`, with the declaration's body also as the plain function a
/// resolved instance of it is called through (RFC-0067 rule 8).
pub fn glue_at_instance<Rt, F, A, R, E>(f: F) -> Glue<Rt, F, A, R, Unsited, E>
where
    Rt: Runtime,
    A: Parameters<Rt>,
    E: AtInstance<Rt>,
    F: for<'a, 'w> Fn(&'a mut Ctx<'w, Rt>, <A as Parameters<Rt>>::Out<'a, 'w>) -> R::Of<'a>,
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
        &'a mut Ctx<'w, Rt>,
        <A as Parameters<Rt>>::Out<'a, 'w>,
    ) -> BoxFuture<'a, Rt::Value>,
{
    AsyncGlue {
        f: Arc::new(f),
        sites: Unsited,
        shape: PhantomData,
    }
}

/// As `glue_at_instance`, for a body that awaits.
pub fn async_glue_at_instance<Rt, F, A, E>(f: F) -> AsyncGlue<Rt, F, A, Unsited, E>
where
    Rt: Runtime,
    A: Parameters<Rt>,
    E: AtInstance<Rt>,
    F: for<'a, 'w> Fn(
        &'a mut Ctx<'w, Rt>,
        <A as Parameters<Rt>>::Out<'a, 'w>,
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
    E: AtInstance<Rt>,
    A: Parameters<Rt> + 'static,
    F: Clone + Send + Sync + 'static,
    F: for<'a, 'w> Fn(&'a mut Ctx<'w, Rt>, <A as Parameters<Rt>>::Out<'a, 'w>) -> R::Of<'a>,
    R: Ret<Rt> + 'static,
{
    type Args = <A as Parameters<Rt>>::Run;
    type Ret = <R as Ret<Rt>>::Form;

    const WIDTH: Width = Width {
        args: <A as Parameters<Rt>>::WIDTH,
        ret: <R as Ret<Rt>>::WIDTH,
        result: <<R as Ret<Rt>>::Form as Form>::KIND,
        absent: <<R as Ret<Rt>>::Form as Returned>::ABSENT,
    };

    #[inline]
    unsafe fn call(
        &self,
        ctx: &mut Ctx<'_, Rt>,
        run: <<A as Parameters<Rt>>::Run as ArgRun>::Run<'_, Rt>,
        out: <<R as Ret<Rt>>::Form as Returned>::Out<'_, Rt>,
    ) -> <<R as Ret<Rt>>::Form as Returned>::Verdict {
        let rt = ctx.rt;
        let run = <<A as Parameters<Rt>>::Run as ArgRun>::as_slice::<Rt>(run);
        // SAFETY: the caller's contract, which is `Parameters::take`'s.
        let args = unsafe { <A as Parameters<Rt>>::take(rt, run, &self.sites) };
        let out = <<R as Ret<Rt>>::Form as Returned>::as_mut_slice::<Rt>(out);
        <R as Ret<Rt>>::into_run((self.f)(ctx, args), rt, out)
    }
}

impl<Rt, F, A, R, E> HandlerFactory<Rt> for Glue<Rt, F, A, R, Unsited, E>
where
    Rt: Runtime,
    E: AtInstance<Rt>,
    A: Parameters<Rt> + 'static,
    F: Clone + Send + Sync + 'static,
    F: for<'a, 'w> Fn(&'a mut Ctx<'w, Rt>, <A as Parameters<Rt>>::Out<'a, 'w>) -> R::Of<'a>,
    R: Ret<Rt> + 'static,
{
    fn clone_box(&self) -> Box<dyn HandlerFactory<Rt>> {
        Box::new(self.clone())
    }

    fn width(&self) -> Width {
        Width {
            args: <A as Parameters<Rt>>::WIDTH,
            ret: <R as Ret<Rt>>::WIDTH,
            result: <<R as Ret<Rt>>::Form as Form>::KIND,
            absent: <<R as Ret<Rt>>::Form as Returned>::ABSENT,
        }
    }

    fn arity(&self) -> usize {
        <A as Parameters<Rt>>::ARITY
    }

    fn at_site(self: Box<Self>, site: &CallSite<'_, Rt>) -> Box<dyn AtSite<Rt>> {
        Box::new(self.at(site))
    }

    fn instance(&self) -> Option<InstanceRun> {
        <E as AtInstance<Rt>>::run()
    }
}

impl<Rt, F, A, R, E> Glue<Rt, F, A, R, Unsited, E>
where
    Rt: Runtime,
    A: Parameters<Rt>,
{
    pub fn at(self, site: &CallSite<'_, Rt>) -> Glue<Rt, F, A, R, <A as Parameters<Rt>>::Sites, E> {
        Glue {
            f: self.f,
            sites: <A as Parameters<Rt>>::sites(site),
            shape: PhantomData,
        }
    }
}

impl<Rt, F, A, R, E> AtSite<Rt> for Glue<Rt, F, A, R, <A as Parameters<Rt>>::Sites, E>
where
    Rt: Runtime,
    E: AtInstance<Rt>,
    A: Parameters<Rt> + 'static,
    F: Clone + Send + Sync + 'static,
    F: for<'a, 'w> Fn(&'a mut Ctx<'w, Rt>, <A as Parameters<Rt>>::Out<'a, 'w>) -> R::Of<'a>,
    R: Ret<Rt> + 'static,
{
    fn into_op(self: Box<Self>, shape: Rt::CallShape) -> Rt::Op {
        Rt::op(*self, shape)
    }

    fn into_fused(self: Box<Self>, shape: Rt::FusedShape) -> Rt::FusedCall {
        Rt::fused(*self, shape)
    }
}

impl<Rt, F, A, R, E> ValuesOnly<Rt> for Glue<Rt, F, A, R, Unsited, E>
where
    Rt: Runtime,
    E: AtInstance<Rt>,
    A: ValueParameters<Rt> + 'static,
    F: Clone + Send + Sync + 'static,
    F: for<'a, 'w> Fn(&'a mut Ctx<'w, Rt>, <A as Parameters<Rt>>::Out<'a, 'w>) -> R::Of<'a>,
    R: Ret<Rt, Form: OneRegister> + 'static,
{
}

impl<Rt, F, A, E> AsyncCall<Rt> for AsyncGlue<Rt, F, A, <A as Parameters<Rt>>::Sites, E>
where
    Rt: Runtime,
    E: AtInstance<Rt>,
    A: ValueParameters<Rt> + 'static,
    F: Send + Sync + 'static,
    F: for<'a, 'w> Fn(
        &'a mut Ctx<'w, Rt>,
        <A as Parameters<Rt>>::Out<'a, 'w>,
    ) -> BoxFuture<'a, Rt::Value>,
{
    const WIDTH: Width = Width {
        args: <A as Parameters<Rt>>::WIDTH,
        ret: 1,
        result: FormKind::Value,
        absent: false,
    };

    unsafe fn call(&self, rt: Rt, run: &[Rt::Value]) -> BoxFuture<'static, Rt::Value> {
        let held: Vec<Rt::Value> = run.to_vec();
        let f = Arc::clone(&self.f);
        let sites = self.sites.clone();
        Box::pin(async move {
            let mut rooted = rt.rooted();
            // SAFETY: the `Ctx` is lent to the handler's body alone, and
            // safe code reaches no second `Ctx` to exchange it with:
            // `ctx_of`, `Ctx::new` and `Ctx::frame_mut` are `unsafe`.
            let ctx = unsafe { Rt::ctx_of(&mut rooted) };
            // SAFETY: as the synchronous impl's, over the run the future owns.
            let args = unsafe { <A as Parameters<Rt>>::take(ctx.rt, &held, &sites) };
            f(ctx, args).await
        })
    }
}

impl<Rt, F, A, E> AsyncFactory<Rt> for AsyncGlue<Rt, F, A, Unsited, E>
where
    Rt: Runtime,
    E: AtInstance<Rt>,
    A: ValueParameters<Rt> + 'static,
    F: Send + Sync + 'static,
    F: for<'a, 'w> Fn(
        &'a mut Ctx<'w, Rt>,
        <A as Parameters<Rt>>::Out<'a, 'w>,
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
            absent: false,
        }
    }

    fn arity(&self) -> usize {
        <A as Parameters<Rt>>::ARITY
    }

    fn at_site(self: Box<Self>, site: &CallSite<'_, Rt>) -> Box<dyn AsyncAtSite<Rt>> {
        Box::new(self.at(site))
    }

    fn instance(&self) -> Option<InstanceRun> {
        <E as AtInstance<Rt>>::run()
    }
}

impl<Rt, F, A, E> AsyncGlue<Rt, F, A, Unsited, E>
where
    Rt: Runtime,
    A: Parameters<Rt>,
{
    pub fn at(
        self,
        site: &CallSite<'_, Rt>,
    ) -> AsyncGlue<Rt, F, A, <A as Parameters<Rt>>::Sites, E> {
        AsyncGlue {
            f: self.f,
            sites: <A as Parameters<Rt>>::sites(site),
            shape: PhantomData,
        }
    }
}

impl<Rt, F, A, E> AsyncAtSite<Rt> for AsyncGlue<Rt, F, A, <A as Parameters<Rt>>::Sites, E>
where
    Rt: Runtime,
    E: AtInstance<Rt>,
    A: ValueParameters<Rt> + 'static,
    F: Send + Sync + 'static,
    F: for<'a, 'w> Fn(
        &'a mut Ctx<'w, Rt>,
        <A as Parameters<Rt>>::Out<'a, 'w>,
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

    /// This instance's entry (RFC-0070 rule 2), at the task its
    /// declaration named.
    ///
    /// `Heavy` has none, and that is a proof rather than a gap. Reaching a
    /// `Heavy` body through an entry is either running it in the caller's
    /// frame, which is `Task::Sync` and drops the whole reason the
    /// declaration said `heavy`, or offloading it — which needs the
    /// runtime's executor, and an entry's arguments do not carry one.
    pub fn instance(&self) -> Option<InstanceRun> {
        match self {
            Self::Sync(f) => f.instance(),
            Self::Async(f) => f.instance(),
            Self::Heavy(_) => None,
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
    /// Written at this instance's own variables (RFC-0070 rule 3): the solver
    /// opens one decision per entry once it settles on this candidate. A
    /// declaration that is no signature's instance reaches the solver
    /// through `FnKind::Extern::requires` instead, and its instances carry
    /// none.
    pub requires: Vec<RequirementSig>,
    /// The bound of each effect variable of `signature`, by position, carried
    /// as `requires` is: a signature's instance states its own (RFC-0011
    /// rule 5), and any other declaration states them on
    /// `FnDecl::effect_bounds`, so its instances carry none.
    pub effect_bounds: Vec<EffectVarBound>,
}

impl<R> DeclaredInstance<R>
where
    R: Runtime,
{
    pub fn signature_under(
        &self,
        laws: Laws,
        ensures: Vec<Postcondition>,
    ) -> acvus_mir::ty::InstanceSig {
        acvus_mir::ty::InstanceSig {
            ty: self.signature.clone(),
            admits: self.admits,
            task: self.handler.task(),
            requires: self.requires.clone(),
            effect_bounds: self.effect_bounds.clone(),
            laws,
            ensures,
        }
    }
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

    pub fn signatures(
        &self,
        laws: &Laws,
        ensures: &[Postcondition],
    ) -> acvus_mir::ty::Instances {
        acvus_mir::ty::Instances {
            concrete: self
                .concrete
                .iter()
                .map(|i| i.signature_under(laws.clone(), ensures.to_vec()))
                .collect(),
            generic: self.generic.as_ref().map(|_| acvus_mir::ty::GenericSig {
                laws: laws.clone(),
                ensures: ensures.to_vec(),
            }),
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
            // SAFETY: the `Ctx` is lent to the handler's body alone, and
            // safe code reaches no second `Ctx` to exchange it with:
            // `ctx_of`, `Ctx::new` and `Ctx::frame_mut` are `unsafe`.
            let ctx = unsafe { Rt::ctx_of(&mut rooted) };
            // SAFETY: the contract of `DirectOp::call`, which is this
            // closure's only caller: the two runs are the declaration's own
            // widths, which is what each form's `from_slice` asks.
            unsafe {
                let run = <H::Args as ArgRun>::from_slice::<Rt>(run);
                let verdict = handler.call(ctx, run, <H::Ret as Returned>::from_slice::<Rt>(out));
                <H::Ret as Returned>::land_in(rt, verdict, <H::Ret as Returned>::from_slice(out));
            }
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
        fn op<H>(handler: H, _: Self::CallShape) -> Self::Op
        where
            H: $crate::Handler<Self>,
        {
            $crate::DirectOp::of(handler)
        }

        fn fused<H>(handler: H, _: Self::FusedShape) -> Self::FusedCall
        where
            H: $crate::Handler<Self>,
        {
            $crate::DirectOp::of(handler)
        }

        fn async_extern_op<H>(handler: H, _: Self::AsyncShape) -> Self::Op
        where
            H: $crate::AsyncCall<Self>,
        {
            $crate::DirectOp::awaiting(handler)
        }
    };
}
