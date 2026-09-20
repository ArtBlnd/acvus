//! A requirement is a bound, and the instance it names is a pointer the
//! call site resolved (RFC-0067 Decisions 1 and 4).
//!
//! A handler requires a shared signature of one of its type variables by
//! writing `T: Instance<sig::S<T, Rt>>`, and calls it as `T::call(&x, rt,
//! frame, rest)`. Nothing is looked up while the call runs: the entry is
//! resolved once, when the glue is given its site.

use crate::handler::{ArgAt, Entry};
use crate::runtime::Runtime;
use crate::ty_arg::{Var, kind};

/// A bounded type variable's carrier: the value one argument of one call
/// site passed, with the entry of each instance the declaration required
/// beside it. `#[extern_fn]` writes one per bounded variable.
pub trait Carrier<Rt>: Var<kind::Type> + Sized
where
    Rt: Runtime,
{
    /// One entry per `Instance` bound. The index is the bound's position in
    /// the declaration's `where` clause; `#[extern_fn]` writes both halves,
    /// the array here and the `requires` list it puts on the `FnDecl`.
    type Entries: Clone + Copy + Send + Sync + 'static;

    fn entries(at: ArgAt<'_, Rt>) -> Self::Entries;

    fn of(value: Rt::Value, entries: &Self::Entries) -> Self;

    /// The value where it lies: `call_entry` takes this reference's address
    /// as the storage the instance's `&T` parameter names.
    fn value(&self) -> &Rt::Value;
}

/// A shared signature as a Rust caller of one of its instances sees it:
/// the shape of a call and nothing about a receiver beyond how the first
/// parameter takes it. `extern_signature!` writes the impl, so that a
/// handler which requires a signature restates none of its modes and none
/// of its widths.
pub trait Signature<Rt>: Send + Sync + 'static
where
    Rt: Runtime,
{
    /// What the signature's first parameter stands at: the variable a bound
    /// names, which RFC-0019 makes the one an instance is matched by.
    type This: Carrier<Rt>;
    /// The first parameter's mode: `&'a This`, `&'a mut This`, or `This`.
    /// The mode reaches a requiring handler through this projection alone,
    /// so the handler's own `I::call` is where a wrong mode is refused;
    /// `acvus-extern-macro/tests/compile_fail/instance_wrong_mode.rs` is
    /// that refusal, executed.
    type Recv<'a>;
    /// The arguments after the first.
    type Rest<'a>;
    type Ret;

    /// The carrier a receiver in any of the three modes stands at, which is
    /// where the entry of this signature lies.
    fn as_this<'a>(recv: &'a Self::Recv<'_>) -> &'a Self::This;

    /// # Safety
    /// `entry` is the entry of this signature's instance at the acvus type
    /// `this` holds, which is what `Carrier::entries` resolved at the site
    /// `this` came from.
    unsafe fn call_entry(
        entry: Entry<Rt>,
        rt: &Rt,
        frame: &mut Rt::Frame<'_>,
        this: Self::Recv<'_>,
        rest: Self::Rest<'_>,
    ) -> Self::Ret;
}

/// One of the runtime's values holding what `value` crosses as. The run an
/// entry is called with is built out of these, one per parameter after the
/// first.
pub fn one_value<Rt, T>(rt: &Rt, value: T) -> Rt::Value
where
    Rt: Runtime,
    T: crate::obj::Cross<Rt, Form = crate::obj::One>,
{
    let mut out = [<Rt::Value as Default>::default()];
    value.into_run(rt, &mut out);
    out[0]
}

/// A type with an instance of the shared signature `S` (RFC-0067 Decision
/// 1). `call` is the handle the marker bound `HasInstance<S>` never had,
/// and for want of which that bound was deleted.
///
/// What a call takes and gives is `S`'s and not this trait's, so a handler
/// that requires a signature writes the requirement and nothing more: the
/// modes, the widths and the result are read off the signature's own
/// declaration wherever the bound is used.
pub trait Instance<S, Rt>: Sized
where
    Rt: Runtime,
    S: Signature<Rt, This = Self>,
{
    fn call(
        this: <S as Signature<Rt>>::Recv<'_>,
        rt: &Rt,
        frame: &mut Rt::Frame<'_>,
        rest: <S as Signature<Rt>>::Rest<'_>,
    ) -> <S as Signature<Rt>>::Ret;
}
