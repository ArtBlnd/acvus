//! `Ref<T, M, Rt>`: the acvus types `&T` and `&mut T` in an extern
//! declaration, carrying the reference value itself (RFC-0018, RFC-0028). A
//! Rust parameter `&T` / `&mut T` declares the same type and is read at entry;
//! a carrier is for a body that keeps the reference, returns it, or takes it
//! inside a lambda or an iterator. Its region is the caller's.
//!
//! A `Ref<T, M, Rt>` is made in one place: the crossing, as
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

pub struct Ref<T, M, Rt>(Rt::Value, PhantomData<(T, M)>)
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime;

impl<T, Rt> Ref<T, Shared, Rt>
where
    T: Borrowable<Rt>,
    Rt: Runtime,
{
    /// Read through the reference, at the storage `T`'s own crossing wrote.
    pub fn with<'a, R>(&'a self, rt: &Rt, f: impl FnOnce(&'a T) -> R) -> R {
        // SAFETY: the module's head: the storage holds a `T` and is live.
        f(unsafe { <Shared as Loan>::borrow::<T, crate::Uniform, Rt>(rt, &self.0) })
    }
}

impl<T, Rt> Ref<T, Mut, Rt>
where
    T: Borrowable<Rt>,
    Rt: Runtime,
{
    /// Read and write through the reference.
    pub fn with<'a, R>(&'a mut self, rt: &Rt, f: impl FnOnce(&'a mut T) -> R) -> R {
        // SAFETY: as the shared `with`'s, and an exclusive loan is the only
        // live name of its storage, which `&'a mut self` keeps.
        f(unsafe { <Mut as Loan>::borrow::<T, crate::Uniform, Rt>(rt, &self.0) })
    }
}

impl<T, M, Rt> Var<kind::Type> for Ref<T, M, Rt>
where
    T: Var<kind::Type>,
    M: Loan,
    Rt: Runtime,
{
}

impl<T, M, Rt> TyArg for Ref<T, M, Rt>
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

impl<T, M, Rt> crate::Cross<Rt> for Ref<T, M, Rt>
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    type Form = crate::One;
    type ReturnForm = crate::One;

    unsafe fn from_run(rt: &Rt, run: &[Rt::Value]) -> Self {
        // SAFETY: the caller's contract, at one value.
        unsafe { <Self as crate::OneValue<Rt>>::from_run(rt, run) }
    }

    fn into_run(self, rt: &Rt, out: &mut [Rt::Value]) {
        <Self as crate::OneValue<Rt>>::into_run(self, rt, out)
    }

    fn into_return_run(self, rt: &Rt, out: &mut [Rt::Value]) {
        <Self as crate::OneValue<Rt>>::into_run(self, rt, out)
    }
}

/// A closure parameter declared `&T` is passed as Rust's `&T`: the borrow
/// lives for the call by Rust's rule, and the reference word the crossing
/// makes from it is kept no longer than the call by the runtime's
/// (`Runtime::reference`, RFC-0018).
impl<T, Rt> crate::Passed<Rt> for Ref<T, Shared, Rt>
where
    T: TransparentOver<Rt> + Sync,
    Rt: Runtime,
{
    type As<'a> = &'a T;

    fn cross(rt: &Rt, passed: &T) -> Rt::Value {
        // SAFETY: `passed` is live for the call, and the closure keeps the
        // reference no longer than that.
        unsafe { rt.reference(value_of(passed)) }
    }

    unsafe fn restore<'a>(rt: &Rt, word: Rt::Value) -> &'a T {
        // SAFETY: the caller's contract: the storage the word names is live
        // for `'a`, which the caller took from the receiver it lent, and it
        // holds a `T`. The word itself is a copy and does not bound `'a`.
        unsafe { &*(<Shared as Loan>::borrow::<T, crate::Uniform, Rt>(rt, &word) as *const T) }
    }
}

impl<T, Rt> crate::Passed<Rt> for Ref<T, Mut, Rt>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    type As<'a> = &'a mut T;

    fn cross(rt: &Rt, passed: &mut T) -> Rt::Value {
        // SAFETY: as the shared form's, and `&mut T` is the only live name.
        unsafe { rt.reference(value_of(passed)) }
    }

    unsafe fn restore<'a>(rt: &Rt, word: Rt::Value) -> &'a mut T {
        // SAFETY: as the shared form's, and the receiver was lent
        // exclusively for `'a`.
        unsafe { &mut *(<Mut as Loan>::borrow::<T, crate::Uniform, Rt>(rt, &word) as *mut T) }
    }
}

/// A result declared `&T` / `&mut T` is returned as Rust's borrow of a
/// parameter the caller lent (RFC-0047 rule 3), and crosses as one reference
/// word.
impl<T, Rt> crate::LentBack<Rt> for Ref<T, Shared, Rt>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    type Of<'a> = &'a T;
    type Form = crate::One;

    fn into_run(value: &T, rt: &Rt, out: &mut [Rt::Value]) {
        // SAFETY: the storage is the caller's, lent for this call and
        // outliving it; RFC-0018 keeps the reference within the caller.
        out[0] = unsafe { rt.reference(value_of(value)) };
    }
}

impl<T, Rt> crate::LentBack<Rt> for Ref<T, Mut, Rt>
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    type Of<'a> = &'a mut T;
    type Form = crate::One;

    fn into_run(value: &mut T, rt: &Rt, out: &mut [Rt::Value]) {
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

impl<T, M, Rt> crate::OneValue<Rt> for Ref<T, M, Rt>
where
    T: Send + Sync + 'static,
    M: Loan,
    Rt: Runtime,
{
    fn erase(self, _: &Rt) -> Rt::Value {
        self.0
    }

    unsafe fn materialize(_: &Rt, value: Rt::Value) -> Self {
        Self(value, PhantomData)
    }
}
