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
/// impl here answers by `T`, and only `Branded`'s exists by it: what reads
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

    pub fn as_mut<'a>(&'a mut self, rt: &'a R) -> &'a mut T
    where
        T: Stored<R>,
    {
        // SAFETY: as in `as_ref`; `&mut self` is the exclusive name.
        T::from_payload_mut(unsafe { Holding::new() }, unsafe {
            rt.value_as_mut::<T::Payload>(&mut self.0)
        })
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

// SAFETY: `At<'a>` is `Self`, which holds where `T` is `Unbranded`: the
// value was erased from a `T` that names no lifetime, so nothing it holds
// was lent at a brand. `Never` has no value. A `T` that names a lifetime,
// a carrier or a type holding one, has no brand here, and an `Erased` at it
// reaches no handler (RFC-0079 rule 6).
unsafe impl<R, T> crate::Branded for Erased<R, T>
where
    R: Runtime,
    T: 'static,
    fn() -> T: brand::FromUnbranded,
{
    type At<'a> = Self;
}

/// The one impl on `Erased` whose existence depends on `T` is `Branded`'s
/// (RFC-0076 rule 1): `Branded` has no item that reads, writes or lays out a
/// value. An impl of a trait that requires `Branded` states `Self: Branded`
/// or `Self: Unbranded` and no other bound on `T`; `tests/erased_impls.rs`
/// refuses any other.
mod brand {
    /// `fn() -> T` where `T` is `Unbranded` or is `Never`. `Never` is named
    /// through `fn() -> !` here because an impl on its alias conflicts
    /// (E0119) with every other impl of the trait; `T` is under `fn() ->`
    /// so that the two impls are disjoint, `!` not being `Branded`.
    pub trait FromUnbranded {}

    impl<T> FromUnbranded for fn() -> T where T: crate::Unbranded {}

    impl FromUnbranded for fn() -> ! {}
}
crate::cross_one_value!(Erased<__Rt, T>, [T: 'static] where Self: crate::Branded,);

// SAFETY: an `Erased<R, T>` holds the word the crossing made for the `T` the
// checker settled, so `erase` hands that word back and `materialize` holds the
// word it is handed, unread; nothing else crosses, and the capability is not
// used.
unsafe impl<Rep, R, T> OneValue<R, Rep> for Erased<R, T>
where
    R: Runtime,
    T: 'static,
    Self: crate::Branded,
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
    T: 'static,
    Self: crate::Unbranded,
{
    crate::stored_as_canonical!(R);
}

impl<R, T> crate::Borrowable<R> for Erased<R, T>
where
    R: Runtime,
    T: 'static,
    Self: crate::Branded,
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

// SAFETY: the storage is named as a `Vec<Self>` only because `Owned<R>` is this
// type's canonical form, with the layout checked in the body; the capability is
// not used.
unsafe impl<R, T> InPlaceElement<R> for Erased<R, T>
where
    R: Runtime,
    T: 'static,
    Self: crate::Branded,
{
    fn in_place<'v>(_: Holding<'_, R>, values: &'v Vec<Owned<R>>) -> &'v Vec<Self> {
        same_layout!(Vec<Owned<R>>, Vec<Self>);
        // SAFETY: `Owned<R>` is this type's canonical form, and the two
        // `Vec`s differ in nothing else (`Canonical`), with the layout
        // checked above.
        unsafe { &*(values as *const Vec<Owned<R>>).cast::<Vec<Self>>() }
    }

    fn in_place_mut<'v>(_: Holding<'_, R>, values: &'v mut Vec<Owned<R>>) -> &'v mut Vec<Self> {
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
    Self: crate::Unbranded,
{
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

/// `T` is `Unbranded`, as it is where `Erased` has a brand: `Branded::At`
/// keeps `T` (RFC-0076 rule 1), so a carrier's marker there would hand a
/// handler a value it could keep past its brand (RFC-0079 rule 6).
impl<R, T> TyArg for Erased<R, T>
where
    R: HoldsNoValues,
    T: TyArg + crate::Unbranded,
{
    fn poly_ty(interner: &Interner, vars: &PolyVars) -> PolyTy {
        T::poly_ty(interner, vars)
    }

    fn held(interner: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        TypeArg::uniform(T::poly_ty(interner, vars))
    }
}
