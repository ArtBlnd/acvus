//! `Ref<'a, T, M, Rt>`: the acvus types `&T` and `&mut T` in an extern
//! declaration, carrying the reference value itself (RFC-0018, RFC-0028). A
//! Rust parameter `&T` / `&mut T` declares the same type and is read at entry;
//! a carrier is for a body that keeps the reference, returns it, or takes it
//! inside a lambda or an iterator. `'a` is the call it was handed to, so a
//! handler cannot keep it past that call (RFC-0079 rule 6).
//!
//! A `Ref<'a, T, M, Rt>` is made in one place: the crossing, as
//! `OneValue::materialize` of a value the checker typed `&T` / `&mut T`. That
//! is the whole ground of the `T` it carries, and it is why the type has no
//! constructor a handler can call. A constructor from `Rt::Value` — a `new`,
//! a `lend` over a handler's own storage — would put a `T` on a word that
//! nothing checked, and a `Ref<T>` made over a `U` reads a `U` as a `T` at
//! the first `with` (RFC-0068 rule 1). A reference a handler makes to its own
//! value is Rust's `&T`, with Rust's lifetime; it is not this type.
//!
//! `with` is the one operation, and Rust proves it safe: the borrow it
//! hands `f` is `&'a T` for the `'a` of `&'a self`, or `&'a mut T` for
//! `&'a mut self`, so it cannot outlive the `Ref`, two exclusive borrows
//! cannot be live at once, and a body that tries to carry one out of the
//! closure does not compile. What the borrow reads is a `T` in live storage by the
//! premise above — the crossing made the `Ref`, and the loans analysis keeps
//! what a reference names alive for every use of it (RFC-0018), an
//! exclusive loan being the only live name. The storage the language keeps
//! for a `T` is a Rust `T` only where `T: Borrowable<Rt>`, which `with` asks
//! for: a `Vec<i64>` is kept as a `Vec<Owned<Rt>>`. `with` adds nothing
//! else to that premise and takes nothing from it; its shape is what keeps
//! the proof inside Rust.
//!
//! Nothing else is here on purpose. `map` and `try_map` made a `Ref<U>` from
//! a Rust `&U` borrowed inside `with`: a reference value over a part the
//! analysis never named, with the lifetime dropped — what `with` gives as a
//! Rust borrow is enough. `as_slice` and `elements` handed out the storage
//! as `[T]` or `[Rt::Value]`: a `with` over `&Vec<T>` is the same view with
//! Rust's lifetime on it. `into_value` existed to feed the constructors that
//! are gone.

use std::marker::PhantomData;

use acvus_mir::ty::{PolyTy, TypeArg};
use acvus_utils::Interner;

use crate::handler::Borrowable;
use crate::loan::{Loan, Mut, Shared};
use crate::obj::TransparentOver;
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg, Var, kind};

pub struct Ref<'a, T, M, Rt>(Rt::Value, PhantomData<(&'a (), T, M)>)
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime;

// SAFETY: `At<'b>` changes the brand alone.
unsafe impl<'a, T, M, Rt> crate::Branded for Ref<'a, T, M, Rt>
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    type At<'b> = Ref<'b, T, M, Rt>;
}

impl<'b, T, Rt> Ref<'b, T, Shared, Rt>
where
    T: Borrowable<Rt>,
    Rt: Runtime,
{
    /// Read through the reference, at the storage `T`'s own crossing wrote,
    /// with what `T` holds at the read's lifetime.
    pub fn with<'a, R>(&'a self, rt: &Rt, f: impl FnOnce(&'a T::At<'a>) -> R) -> R {
        // SAFETY: the module's head: the storage holds a `T` and is live,
        // and what it holds is live for as long as this reference.
        f(unsafe {
            crate::brand_ref::<T>(<Shared as Loan>::borrow::<T, crate::Uniform, Rt>(rt, &self.0))
        })
    }
}

impl<'b, T, Rt> Ref<'b, T, Mut, Rt>
where
    T: Borrowable<Rt>,
    Rt: Runtime,
{
    /// Read and write through the reference.
    pub fn with<'a, R>(&'a mut self, rt: &Rt, f: impl FnOnce(&'a mut T::At<'a>) -> R) -> R {
        // SAFETY: as the shared `with`'s, and an exclusive loan is the only
        // live name of its storage, which `&'a mut self` keeps.
        f(unsafe {
            crate::brand_mut::<T>(<Mut as Loan>::borrow::<T, crate::Uniform, Rt>(rt, &self.0))
        })
    }
}

impl<T, M, Rt> Var<kind::Type> for Ref<'static, T, M, Rt>
where
    T: Var<kind::Type>,
    M: Loan,
    Rt: Runtime,
{
}

// SAFETY: the referent is its own canonical form's, and the brand is at
// `'static`.
unsafe impl<'a, T, M, Rt> crate::Canonical<kind::Type> for Ref<'a, T, M, Rt>
where
    T: Var<kind::Type>,
    M: Loan,
    Rt: Runtime,
{
    type Canon = Ref<'static, T::Canon, M, Rt>;
}

// SAFETY: a `Ref` is one `Rt::Value` at every `T` and `M`; the referent is
// a box of its own, read through its own `Canonical`.
unsafe impl<'a, Mk, T, M, Rt> crate::UniformPayload<Mk> for Ref<'a, T, M, Rt>
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
}

impl<T, M, Rt> TyArg for Ref<'static, T, M, Rt>
where
    T: TyArg + Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Ref(
            M::MUTABILITY,
            Box::new(TypeArg::uniform(T::poly_ty(i, vars))),
        )
    }
}

impl<T, M, Rt> crate::Cross<Rt> for Ref<'static, T, M, Rt>
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    type Form = crate::One;
    type ReturnForm = crate::One;

    unsafe fn from_run(rt: crate::Crossing<'_, Rt>, run: &[Rt::Value]) -> Self {
        // SAFETY: the caller's contract, at one value.
        unsafe { <Self as crate::OneValue<Rt>>::from_run(rt, run) }
    }

    fn into_run(self, rt: crate::Crossing<'_, Rt>, out: &mut [Rt::Value]) {
        <Self as crate::OneValue<Rt>>::into_run(self, rt, out)
    }

    fn into_return_run(self, rt: crate::Crossing<'_, Rt>, out: &mut [Rt::Value]) {
        <Self as crate::OneValue<Rt>>::into_run(self, rt, out)
    }
}

/// A closure parameter declared `&T` is passed as Rust's `&T`: the borrow
/// lives for the call by Rust's rule, and the reference word the crossing
/// makes from it is kept no longer than the call by the runtime's
/// (`Runtime::reference`, RFC-0018).
impl<T, Rt> crate::Passed<Rt> for Ref<'static, T, Shared, Rt>
where
    T: TransparentOver<Rt> + Sync,
    Rt: Runtime,
{
    type As<'a> = &'a T;

    fn cross(rt: crate::Crossing<'_, Rt>, passed: &T) -> Rt::Value {
        // SAFETY: `passed` is live for the call, and the closure keeps the
        // reference no longer than that.
        unsafe { rt.reference(value_of(passed)) }
    }

    unsafe fn restore<'a>(rt: crate::Crossing<'_, Rt>, word: Rt::Value) -> &'a T {
        // SAFETY: the caller's contract: the storage the word names is live
        // for `'a`, which the caller took from the receiver it lent, and it
        // holds a `T`, which `TransparentOver` lays out as the runtime's
        // value. The word itself is a copy and does not bound `'a`.
        unsafe {
            &*(<Rt::Value as Borrowable<Rt>>::deref(rt.rt(), &word) as *const Rt::Value).cast::<T>()
        }
    }
}

impl<T, Rt> crate::Passed<Rt> for Ref<'static, T, Mut, Rt>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    type As<'a> = &'a mut T;

    fn cross(rt: crate::Crossing<'_, Rt>, passed: &mut T) -> Rt::Value {
        // SAFETY: as the shared form's, and `&mut T` is the only live name.
        unsafe { rt.reference(value_of(passed)) }
    }

    unsafe fn restore<'a>(rt: crate::Crossing<'_, Rt>, word: Rt::Value) -> &'a mut T {
        // SAFETY: as the shared form's, and the receiver was lent
        // exclusively for `'a`.
        unsafe {
            &mut *(<Rt::Value as Borrowable<Rt>>::deref_mut(rt.rt(), &word) as *mut Rt::Value)
                .cast::<T>()
        }
    }
}

/// A result declared `&T` / `&mut T` is returned as Rust's borrow of a
/// parameter the caller lent (RFC-0047 rule 3), and crosses as one reference
/// word.
impl<T, Rt> crate::LentBack<Rt> for Ref<'static, T, Shared, Rt>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    type Of<'a> = &'a T;
    type Form = crate::One;

    fn into_run(value: &T, rt: crate::Crossing<'_, Rt>, out: &mut [Rt::Value]) {
        // SAFETY: the storage is the caller's, lent for this call and
        // outliving it; RFC-0018 keeps the reference within the caller.
        out[0] = unsafe { rt.reference(value_of(value)) };
    }
}

impl<T, Rt> crate::LentBack<Rt> for Ref<'static, T, Mut, Rt>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    type Of<'a> = &'a mut T;
    type Form = crate::One;

    fn into_run(value: &mut T, rt: crate::Crossing<'_, Rt>, out: &mut [Rt::Value]) {
        // SAFETY: as the shared form's, exclusively.
        out[0] = unsafe { rt.reference(value_of(value)) };
    }
}

/// `T: TransparentOver<Rt>` is the promise that a `T` is one `Rt::Value` in
/// its layout, so its address is the value's.
fn value_of<T, Rt>(item: &T) -> &Rt::Value
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    // SAFETY: `TransparentOver`'s contract.
    unsafe { &*(item as *const T).cast::<Rt::Value>() }
}

impl<T, M, Rt> crate::OneValue<Rt> for Ref<'static, T, M, Rt>
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    fn erase(self, _: crate::Crossing<'_, Rt>) -> Rt::Value {
        self.0
    }

    unsafe fn materialize(_: crate::Crossing<'_, Rt>, value: Rt::Value) -> Self {
        Self(value, PhantomData)
    }
}
