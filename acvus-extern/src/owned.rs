//! Ownership at the boundary (RFC-0048 §1, §7).
//!
//! The ABI — handler signatures, `Ref`, `Elements`, the glue's window —
//! passes `R::Value` and owes nothing. `Owned<R>` is the only holder that
//! owes a release, and the conversion between the two happens at the
//! glue the macro emits, never inside a handler body.

use std::fmt;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};

use crate::runtime::Runtime;

pub trait Release: Copy + Send + Sync + 'static {
    fn release(self);
}

/// Every Rust store that owns a runtime value holds one of these: a
/// container's elements, an `Erased`, an iterator stage's captured
/// closure. A store that holds `R::Value` instead owns nothing and must
/// be borrowing.
#[repr(transparent)]
pub struct Owned<R>(ManuallyDrop<R::Value>)
where
    R: Runtime;

impl<R> Owned<R>
where
    R: Runtime,
{
    #[inline(always)]
    pub fn from_value(value: R::Value) -> Self {
        Self(ManuallyDrop::new(value))
    }

    #[inline(always)]
    pub fn into_value(self) -> R::Value {
        let mut held = ManuallyDrop::new(self);
        // SAFETY: `held` is a `ManuallyDrop`, so `Owned::drop` does not
        // run, and this is the only read of the inner value.
        unsafe { ManuallyDrop::take(&mut held.0) }
    }

    #[inline(always)]
    pub fn borrow_value(&self) -> R::Value {
        *self.0
    }

    #[inline(always)]
    pub fn release(self) {
        self.into_value().release();
    }
}

impl<R> Drop for Owned<R>
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

impl<R> Deref for Owned<R>
where
    R: Runtime,
{
    type Target = R::Value;

    #[inline(always)]
    fn deref(&self) -> &R::Value {
        &self.0
    }
}

impl<R> DerefMut for Owned<R>
where
    R: Runtime,
{
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut R::Value {
        &mut self.0
    }
}

impl<R> Default for Owned<R>
where
    R: Runtime,
{
    fn default() -> Self {
        Self::from_value(R::Value::default())
    }
}

impl<R> fmt::Debug for Owned<R>
where
    R: Runtime,
    R::Value: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

// SAFETY: `R::Value` is `Send + Sync` through the `Release` bound the
// `Runtime` contract carries; the release obligation adds no capability.
unsafe impl<R> Send for Owned<R> where R: Runtime {}
// SAFETY: as `Send`.
unsafe impl<R> Sync for Owned<R> where R: Runtime {}

impl<R> crate::Cross<R> for Owned<R>
where
    R: Runtime,
{
    const STORED_AS_VALUE: bool = true;

    fn erase(self, _: &R) -> R::Value {
        self.into_value()
    }

    unsafe fn materialize(_: &R, value: R::Value) -> Self {
        Self::from_value(value)
    }

    unsafe fn deref<'a>(rt: &R, reference: &'a R::Value) -> &'a Self {
        // SAFETY: the caller's contract: a live storage of a runtime value,
        // and `Self` is `repr(transparent)` over it.
        unsafe { &*(rt.deref::<R::Value>(reference) as *const R::Value).cast::<Self>() }
    }

    unsafe fn deref_mut<'a>(rt: &R, reference: &'a R::Value) -> &'a mut Self {
        // SAFETY: as `deref`, with the caller's exclusive loan.
        unsafe { &mut *(rt.deref_mut::<R::Value>(reference) as *mut R::Value).cast::<Self>() }
    }
}

impl<R> crate::CrossSpecialized<R> for Owned<R>
where
    R: Runtime,
{
    fn erase(self, _: &R) -> R::Value {
        self.into_value()
    }

    unsafe fn materialize(_: &R, value: R::Value) -> Self {
        Self::from_value(value)
    }
}

impl<R> crate::Stored<R> for Owned<R> where R: Runtime {}

// SAFETY: `Owned<R>` is `#[repr(transparent)]` with `ManuallyDrop<R::Value>`
// — itself `repr(transparent)` over `R::Value` — as its one field.
unsafe impl<R> crate::TransparentOver<R> for Owned<R> where R: Runtime {}

impl<R> crate::FromValue<R> for Owned<R>
where
    R: Runtime,
{
    fn from_value(_: &R, value: R::Value) -> Self {
        Self::from_value(value)
    }
}
