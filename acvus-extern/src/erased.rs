//! `Erased<R, T>`: a runtime value that was erased from a `T`, read and
//! edited as a `T` in place.

use std::fmt;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::Deref;

use acvus_mir::ty::{Poly, PolyTy, TypeArg};
use acvus_utils::Interner;

use crate::canonical::{Canonical, same_layout};
use crate::obj::{FromValue, InPlaceElement, Inline, OneValue, Stored, TransparentOver};
use crate::owned::{Owned, Release};
use crate::runtime::{HoldsNoValues, Runtime};
use crate::ty_arg::{PolyVars, TyArg, Var, kind};

/// Every Rust store that owns a runtime value holds one of these: a
/// container's elements, an `Erased`, an iterator stage's captured closure.
/// A store that holds `R::Value` instead owns nothing and must be borrowing.
///
/// `T` is only in the `PhantomData`. At a runtime that makes values no trait
/// impl here exists or answers by `T`: what reads `T` is an inherent method
/// bounded on that method, and `TyArg` holds at `TypesOnly` alone. That is
/// the first ground of the third layer in `Canonical`'s `# Safety`, on which
/// every read of a box at another `Erased`'s name rests (RFC-0076).
#[repr(transparent)]
pub struct Erased<R, T>(ManuallyDrop<R::Value>, PhantomData<fn() -> T>)
where
    R: Runtime;

impl<R, T> Erased<R, T>
where
    R: Runtime,
{
    pub(crate) fn holding(value: R::Value) -> Self {
        Self(ManuallyDrop::new(value), PhantomData)
    }

    pub(crate) fn held_mut(&mut self) -> &mut R::Value {
        &mut self.0
    }

    /// As `Owned::from_value`'s.
    #[doc(hidden)]
    #[inline(always)]
    pub fn into_value(self) -> R::Value {
        let mut held = ManuallyDrop::new(self);
        // SAFETY: `held` is a `ManuallyDrop`, so `Erased::drop` does not
        // run, and this is the only read of the inner value.
        unsafe { ManuallyDrop::take(&mut held.0) }
    }

    #[inline(always)]
    pub fn release(self) {
        self.into_value().release();
    }

    pub fn new(rt: &R, value: T) -> Self
    where
        T: Stored<R>,
    {
        Self::holding(value.erase(rt))
    }

    /// `FromValue::from_value` at a named `T`, with the debug check that
    /// `value` records `erase::<T::Payload>`. Inherent, not that trait's
    /// impl: the check reads `T` through `T: Stored<R>`, and a trait impl on
    /// `Erased` bounded on `T` is what RFC-0076 rule 1 forbids, so the impl
    /// holds for every `T` and does not look.
    ///
    /// # Safety
    /// `FromValue`'s contract: `value` was erased from `T`.
    pub unsafe fn from_value_of(rt: &R, value: R::Value) -> Self
    where
        T: Stored<R>,
    {
        crate::debug_assert_erased_from!(rt, &value, <T as Stored<R>>::Payload);
        Self::holding(value)
    }

    /// The bound is `Stored`, not `Cross`, and there is no check here: a
    /// type converted on the way in (a derived struct, stored as an `Obj`)
    /// has no `T` in storage to read, so it is refused at the type. An
    /// extension type is `Stored` at its payload, whose bytes are the `T`'s,
    /// so an `Erased<R, X>` reads an `X` in place like any stored type.
    pub fn as_ref<'a>(&'a self, rt: &'a R) -> &'a T
    where
        T: Stored<R>,
    {
        // SAFETY: `new` erased the value from a `T`, and `T: Stored<R>`
        // makes that the runtime's own `erase::<T::Payload>`.
        T::from_payload(unsafe { rt.value_as_ref::<T::Payload>(&self.0) })
    }

    pub fn as_mut<'a>(&'a mut self, rt: &'a R) -> &'a mut T
    where
        T: Stored<R>,
    {
        // SAFETY: as in `as_ref`; `&mut self` is the exclusive name.
        T::from_payload_mut(unsafe { rt.value_as_mut::<T::Payload>(&mut self.0) })
    }

    pub fn into_inner(self, rt: &R) -> T
    where
        T: Stored<R>,
    {
        // SAFETY: `new` erased the value from a `T`.
        unsafe { T::materialize(rt, self.into_value()) }
    }

    pub fn get(&self) -> T
    where
        T: Stored<R> + Inline,
    {
        *self.get_ref()
    }

    pub fn get_ref(&self) -> &T
    where
        T: Stored<R> + Inline,
    {
        // SAFETY: `new` erased the value from a `T`, and an `Inline` `T`
        // lives in the word, which needs no runtime state to read.
        unsafe { R::inline_ref::<T>(&self.0) }
    }

    pub fn get_mut(&mut self) -> &mut T
    where
        T: Stored<R> + Inline,
    {
        // SAFETY: as in `get_ref`; `&mut self` is the exclusive name.
        unsafe { R::inline_mut::<T>(&mut self.0) }
    }
}

impl<R, T> Drop for Erased<R, T>
where
    R: Runtime,
{
    #[inline(always)]
    fn drop(&mut self) {
        // SAFETY: `drop` runs once, and `into_value` — the only other
        // reader — forgets the holder before reading.
        unsafe { ManuallyDrop::take(&mut self.0) }.release();
    }
}

impl<R, T> Deref for Erased<R, T>
where
    R: Runtime,
{
    type Target = R::Value;

    #[inline(always)]
    fn deref(&self) -> &R::Value {
        &self.0
    }
}

impl<R, T> fmt::Debug for Erased<R, T>
where
    R: Runtime,
    R::Value: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

// SAFETY: `R::Value` is `Send + Sync` through the `Release` bound the
// `Runtime` contract carries; the release obligation adds no capability,
// and `PhantomData<fn() -> T>` is `Send + Sync` for every `T`.
unsafe impl<R, T> Send for Erased<R, T> where R: Runtime {}
// SAFETY: as `Send`.
unsafe impl<R, T> Sync for Erased<R, T> where R: Runtime {}

crate::cross_one_value!(Erased<__Rt, T>, T: 'static);

impl<Rep, R, T> OneValue<R, Rep> for Erased<R, T>
where
    R: Runtime,
    T: 'static,
{
    const STORED_AS_VALUE: bool = true;

    fn erase(self, _: &R) -> R::Value {
        self.into_value()
    }

    unsafe fn materialize(_: &R, value: R::Value) -> Self {
        Self::holding(value)
    }
}

impl<R, T> Stored<R> for Erased<R, T>
where
    R: Runtime,
    T: 'static,
{
    crate::stored_as_canonical!();
}

impl<R, T> crate::Borrowable<R> for Erased<R, T>
where
    R: Runtime,
    T: 'static,
{
    unsafe fn deref<'a>(rt: &R, reference: &'a R::Value) -> &'a Self {
        // SAFETY: the caller's contract. The storage a reference names is
        // read by the host's own `R::Value` reading of it and not by
        // `Runtime::deref::<R::Value>`, whose contract is a storage erased
        // *from* a `R::Value`; `Self` is `repr(transparent)` over it.
        let slot = unsafe { <R::Value as crate::Borrowable<R>>::deref(rt, reference) };
        // SAFETY: as above.
        unsafe { &*(slot as *const R::Value).cast::<Self>() }
    }

    unsafe fn deref_mut<'a>(rt: &R, reference: &'a R::Value) -> &'a mut Self {
        // SAFETY: as `deref`, with the caller's exclusive loan.
        let slot = unsafe { <R::Value as crate::Borrowable<R>>::deref_mut(rt, reference) };
        // SAFETY: as above.
        unsafe { &mut *(slot as *mut R::Value).cast::<Self>() }
    }
}

impl<R, T> crate::obj::sealed::Sealed for Erased<R, T> where R: Runtime {}

impl<R, T> InPlaceElement<R> for Erased<R, T>
where
    R: Runtime,
    T: 'static,
{
    fn in_place(values: &Vec<Owned<R>>) -> &Vec<Self> {
        same_layout!(Vec<Owned<R>>, Vec<Self>);
        // SAFETY: `Owned<R>` is this type's canonical form, and the two
        // `Vec`s differ in nothing else (`Canonical`), with the layout
        // checked above.
        unsafe { &*(values as *const Vec<Owned<R>>).cast::<Vec<Self>>() }
    }

    fn in_place_mut(values: &mut Vec<Owned<R>>) -> &mut Vec<Self> {
        same_layout!(Vec<Owned<R>>, Vec<Self>);
        // SAFETY: as `in_place`; `&mut` is the exclusive name.
        unsafe { &mut *(values as *mut Vec<Owned<R>>).cast::<Vec<Self>>() }
    }
}

// SAFETY: `Erased<R, T>` is `#[repr(transparent)]` with
// `ManuallyDrop<R::Value>` — itself `repr(transparent)` over `R::Value` —
// as its one non-zero-sized field.
unsafe impl<R, T> TransparentOver<R> for Erased<R, T>
where
    R: Runtime,
    T: 'static,
{
}

// SAFETY: the contract is `from_value`'s caller's, who names `T` the type
// the value was erased from; the holder reads nothing of it.
unsafe impl<R, T> FromValue<R> for Erased<R, T>
where
    R: Runtime,
{
    unsafe fn from_value(_: &R, value: R::Value) -> Self {
        Self::holding(value)
    }
}

impl<R, T> Var<kind::Type> for Erased<R, T>
where
    R: Runtime,
    T: 'static,
{
}

// SAFETY: `Owned<R>` is `Erased<R, Never>`, which differs from `Self` in
// `T` alone.
unsafe impl<R, T> Canonical<kind::Type> for Erased<R, T>
where
    R: Runtime,
    T: 'static,
{
    type Canon = Owned<R>;
}

// SAFETY: the runtime value is one `R::Value` at every `T`, and `R` is
// bounded by `Runtime` here.
unsafe impl<M, R, T> crate::UniformPayload<M> for Erased<R, T> where R: Runtime {}

impl<R, T> TyArg for Erased<R, T>
where
    R: HoldsNoValues,
    T: TyArg,
{
    fn poly_ty(interner: &Interner, vars: &PolyVars) -> PolyTy {
        T::poly_ty(interner, vars)
    }

    fn held(interner: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        TypeArg::uniform(T::poly_ty(interner, vars))
    }
}
