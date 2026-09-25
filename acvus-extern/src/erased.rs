//! `Erased<R, T>`: a runtime value that was erased from a `T`, read and
//! edited as a `T` in place.

use std::fmt;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;

use acvus_mir::ty::{Poly, PolyTy, TypeArg};
use acvus_utils::Interner;

use crate::canonical::{Canonical, same_layout};
use crate::crossing::{Crossing, Holding};
use crate::obj::{InPlaceElement, Inline, OneValue, Stored, TransparentOver};
use crate::owned::{Owned, Release};
use crate::runtime::{HoldsNoValues, Runtime};
use crate::ty_arg::{PolyVars, TyArg, Var, kind};

/// Every Rust store that owns a runtime value holds one of these: a
/// container's elements, an `Erased`, an iterator stage's captured closure.
/// A store that holds `R::Value` instead owns nothing and must be borrowing.
///
/// `T` is only in the `PhantomData`. At a runtime that makes values no trait
/// impl here answers by `T`, and only `Within`'s exists by it: what reads
/// `T` is an inherent method bounded on that method, and `TyArg` holds at
/// `TypesOnly` alone. That is the first ground of the third layer in
/// `Canonical`'s `# Safety`, on which every read of a box at another
/// `Erased`'s name rests (RFC-0076).
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

    /// `Erased` is `repr(transparent)` over the runtime's value, which a
    /// runtime reads a run or a place of holders at.
    #[doc(hidden)]
    #[inline(always)]
    pub fn over_value() -> crate::repr::SameLayout<R::Value, Self> {
        // SAFETY: `Erased<R, T>` is `repr(transparent)` with
        // `ManuallyDrop<R::Value>`, itself `repr(transparent)` over
        // `R::Value`, as its one non-zero-sized field.
        unsafe { same_layout!(R::Value, Self) }
    }

    pub(crate) fn held_mut(&mut self) -> &mut R::Value {
        &mut self.0
    }

    pub(crate) fn word(&self) -> &R::Value {
        &self.0
    }

    pub(crate) fn take(self) -> R::Value {
        let mut held = ManuallyDrop::new(self);
        // SAFETY: `held` is a `ManuallyDrop`, so `Erased::drop` does not
        // run, and this is the only read of the inner value.
        unsafe { ManuallyDrop::take(&mut held.0) }
    }

    /// The held value, moved out of its holder, which owes its release to
    /// the caller from then on.
    #[doc(hidden)]
    #[inline(always)]
    pub fn into_value(self, _: Holding<'_, R>) -> R::Value {
        self.take()
    }

    #[inline(always)]
    pub fn release(self) {
        self.take().release();
    }

    pub fn new(rt: &R, value: T) -> Self
    where
        T: Stored<R>,
    {
        // SAFETY: `T` is the type this holder names, so the value crosses at
        // the type it was erased from.
        Self::holding(value.erase(unsafe { Crossing::new(rt) }))
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
        // makes that the runtime's own `erase::<T::Payload>`; the payload is
        // named as the `T` this holder names.
        T::from_payload(unsafe { Holding::new() }, unsafe {
            rt.value_as_ref::<T::Payload>(&self.0)
        })
    }

    /// The held `T`, read and written in place for as long as the result
    /// lives; dropping it ends the loan (`Runtime::loan_ended`).
    pub fn as_mut<'a>(&'a mut self, rt: &'a R) -> StoredMut<'a, R, T>
    where
        T: Stored<R>,
    {
        StoredMut { holder: self, rt }
    }

    pub fn into_inner(self, rt: &R) -> T
    where
        T: Stored<R>,
    {
        // SAFETY: `new` erased the value from a `T`, and it crosses back at
        // that `T`.
        unsafe { T::materialize(Crossing::new(rt), self.take()) }
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

    /// As `as_mut`, for an `Inline` `T`, with no runtime in hand.
    pub fn get_mut(&mut self) -> InlineMut<'_, R, T>
    where
        T: Stored<R> + Inline,
    {
        InlineMut { holder: self }
    }
}

/// `Erased::as_mut`'s loan of the held `T`.
pub struct StoredMut<'a, R, T>
where
    R: Runtime,
    T: Stored<R>,
{
    holder: &'a mut Erased<R, T>,
    rt: &'a R,
}

impl<R, T> std::ops::Deref for StoredMut<'_, R, T>
where
    R: Runtime,
    T: Stored<R>,
{
    type Target = T;

    fn deref(&self) -> &T {
        self.holder.as_ref(self.rt)
    }
}

impl<R, T> std::ops::DerefMut for StoredMut<'_, R, T>
where
    R: Runtime,
    T: Stored<R>,
{
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as in `Erased::as_ref`; the loan's `&mut Erased` is the
        // exclusive name, and `Drop` below ends the loan.
        T::from_payload_mut(unsafe { Holding::new() }, unsafe {
            self.rt.value_as_mut::<T::Payload>(&mut self.holder.0)
        })
    }
}

impl<R, T> Drop for StoredMut<'_, R, T>
where
    R: Runtime,
    T: Stored<R>,
{
    fn drop(&mut self) {
        R::loan_ended(&mut self.holder.0);
    }
}

/// `Erased::get_mut`'s loan of the held `Inline` `T`.
pub struct InlineMut<'a, R, T>
where
    R: Runtime,
    T: Stored<R> + Inline,
{
    holder: &'a mut Erased<R, T>,
}

impl<R, T> std::ops::Deref for InlineMut<'_, R, T>
where
    R: Runtime,
    T: Stored<R> + Inline,
{
    type Target = T;

    fn deref(&self) -> &T {
        self.holder.get_ref()
    }
}

impl<R, T> std::ops::DerefMut for InlineMut<'_, R, T>
where
    R: Runtime,
    T: Stored<R> + Inline,
{
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as in `Erased::get_ref`; the loan's `&mut Erased` is the
        // exclusive name, and `Drop` below hands the word to `loan_ended`.
        unsafe { R::inline_mut::<T>(&mut self.holder.0) }
    }
}

impl<R, T> Drop for InlineMut<'_, R, T>
where
    R: Runtime,
    T: Stored<R> + Inline,
{
    fn drop(&mut self) {
        R::loan_ended(&mut self.holder.0);
    }
}

impl<R, T> Drop for Erased<R, T>
where
    R: Runtime,
{
    #[inline(always)]
    fn drop(&mut self) {
        // SAFETY: `drop` runs once, and `take` — the only other reader —
        // forgets the holder before reading.
        unsafe { ManuallyDrop::take(&mut self.0) }.release();
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

// SAFETY: the value was erased from a `T` that holds no carrier at any
// lifetime, so nothing it holds was lent by a call. `Never` has no value. A
// `T` that holds a carrier has no impl here, and an `Erased` at it reaches no
// handler (RFC-0079 rule 6).
unsafe impl<'s, R, T> crate::Within<'s> for Erased<R, T>
where
    R: Runtime,
    fn() -> T: carrier::HoldsNone,
{
}

/// The one impl on `Erased` whose existence depends on `T` is `Within`'s
/// (RFC-0076 rule 1): `Within` has no item that reads, writes or lays out a
/// value. `tests/erased_impls.rs` refuses any other impl that bounds `T`.
mod carrier {
    /// `fn() -> T` where `T` is `Within` at every lifetime or is `Never`.
    /// `Never` is named through `fn() -> !` here because an impl on its alias
    /// conflicts (E0119) with every other impl of the trait; `T` is under
    /// `fn() ->` so that the two impls are disjoint, `!` not being `Within`.
    pub trait HoldsNone {}

    impl<T> HoldsNone for fn() -> T where T: for<'s> crate::Within<'s> {}

    impl HoldsNone for fn() -> ! {}
}
crate::cross_one_value!(Erased<__Rt, T>, T);

// SAFETY: an `Erased<R, T>` holds the word the crossing made for the `T` the
// checker settled, so `erase` hands that word back and `materialize` holds the
// word it is handed, unread; nothing else crosses, and the capability is not
// used.
unsafe impl<Rep, R, T> OneValue<R, Rep> for Erased<R, T>
where
    R: Runtime,
{
    const STORED_AS_VALUE: bool = true;

    fn erase(self, _: crate::Crossing<'_, R>) -> R::Value {
        self.take()
    }

    unsafe fn materialize(_: crate::Crossing<'_, R>, value: R::Value) -> Self {
        Self::holding(value)
    }
}

// SAFETY: the body is `stored_as_canonical!`'s, which names the payload as a
// `Self` through `Canonical`'s layers and reads nothing else; the capability is
// not kept.
unsafe impl<R, T> Stored<R> for Erased<R, T>
where
    R: Runtime,
{
    crate::stored_as_canonical!(R);
}

impl<R, T> crate::Borrowable<R> for Erased<R, T>
where
    R: Runtime,
{
    /// The borrow is the runtime's value itself, which a write replaces
    /// whole; `get_mut` and `as_mut`, which view the held `T` in place, end
    /// their own loans.
    const LENDS_A_WORD: bool = false;

    unsafe fn deref<'a>(rt: &R, reference: &'a R::Value) -> &'a Self {
        // SAFETY: the caller's contract. The storage a reference names is
        // read by the host's own `R::Value` reading of it and not by
        // `Runtime::deref::<R::Value>`, whose contract is a storage erased
        // *from* a `R::Value`; `Self` is `repr(transparent)` over it.
        let slot = unsafe { <R::Value as crate::Borrowable<R>>::deref(rt, reference) };
        // SAFETY: a shared name of the value as its holder releases nothing,
        // and `T` is a `PhantomData`'s.
        unsafe { Self::over_value().cast_ref(slot) }
    }

    unsafe fn deref_mut<'a>(rt: &R, reference: &'a R::Value) -> &'a mut Self {
        // SAFETY: as `deref`, with the caller's exclusive loan.
        let slot = unsafe { <R::Value as crate::Borrowable<R>>::deref_mut(rt, reference) };
        // SAFETY: as `deref`'s. A write through the holder replaces the
        // value whole, as a write of the value would, and the holder is not
        // dropped here, so it releases nothing.
        unsafe { Self::over_value().cast_mut(slot) }
    }
}

impl<R, T> crate::obj::sealed::Sealed for Erased<R, T> where R: Runtime {}

// SAFETY: the storage is named as a `Vec<Self>` only because `Owned<R>` is this
// type's canonical form, with the layout checked in the body; the capability is
// not used.
unsafe impl<R, T> InPlaceElement<R> for Erased<R, T>
where
    R: Runtime,
{
    fn in_place<'v>(_: Holding<'_, R>, values: &'v Vec<Owned<R>>) -> &'v Vec<Self> {
        // SAFETY: `Owned<R>` is this type's canonical form, and the two
        // `Vec`s differ in nothing else (`Canonical`).
        let layout = unsafe { same_layout!(Vec<Owned<R>>, Vec<Self>) };
        // SAFETY: an `Erased<R, T>` differs from an `Owned<R>` in `T` alone,
        // which a `PhantomData` holds, so the elements keep their holders'
        // promises.
        unsafe { layout.cast_ref(values) }
    }

    fn in_place_mut<'v>(_: Holding<'_, R>, values: &'v mut Vec<Owned<R>>) -> &'v mut Vec<Self> {
        // SAFETY: as `in_place`'s.
        let layout = unsafe { same_layout!(Vec<Owned<R>>, Vec<Self>) };
        // SAFETY: as `in_place`'s, both ways; `&mut` is the exclusive name.
        unsafe { layout.cast_mut(values) }
    }
}

// SAFETY: `Erased<R, T>` is `#[repr(transparent)]` with
// `ManuallyDrop<R::Value>` — itself `repr(transparent)` over `R::Value` —
// as its one non-zero-sized field.
unsafe impl<R, T> TransparentOver<R> for Erased<R, T>
where
    R: Runtime,
{
}

impl<R, T> Var<kind::Type> for Erased<R, T>
where
    R: Runtime,
{
}

// SAFETY: `Owned<R>` is `Erased<R, Never>`, which differs from `Self` in
// `T` alone.
unsafe impl<R, T> Canonical<kind::Type> for Erased<R, T>
where
    R: Runtime,
{
    type Canon = Owned<R>;
}

// SAFETY: the runtime value is one `R::Value` at every `T`, and `R` is
// bounded by `Runtime` here.
unsafe impl<M, R, T> crate::UniformPayload<M> for Erased<R, T> where R: Runtime {}

/// `T` holds no carrier, as it does where `Erased` is `Within`: a carrier's
/// marker here would declare a parameter the glue cannot hand a handler
/// (RFC-0079 rule 6).
impl<R, T> TyArg for Erased<R, T>
where
    R: HoldsNoValues,
    T: TyArg + for<'s> crate::Within<'s>,
{
    fn poly_ty(interner: &Interner, vars: &PolyVars) -> PolyTy {
        T::poly_ty(interner, vars)
    }

    fn held(interner: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        TypeArg::uniform(T::poly_ty(interner, vars))
    }
}
