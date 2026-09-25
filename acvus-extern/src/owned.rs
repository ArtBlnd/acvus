//! Ownership at the boundary (RFC-0048 rules 1 and 7).
//!
//! The ABI — handler signatures, `Ref`, `Elements`, the glue's window —
//! passes `R::Value` and owes nothing. `Owned<R>` is the only holder that
//! owes a release, and the conversion between the two happens at the
//! glue the macro emits, never inside a handler body.

use std::ops::Deref;

use crate::crossing::{Crossing, Holding};
use crate::erased::Erased;
use crate::runtime::Runtime;

pub trait Release: Copy + Send + Sync + 'static {
    fn release(self);
}

/// Rust's `!`, named through `fn() -> !` because a stable crate cannot
/// write `!` as a type; it becomes `pub type Never = !;` when `never_type`
/// stabilizes. `Owned` alone names it, and nothing implements a trait on
/// it: an impl on this alias conflicts (E0119) with every other crate's
/// impl of the same trait. The language's `!` is `Bottom`.
pub type Never = <fn() -> ! as never::HasOutput>::Output;

mod never {
    pub trait HasOutput {
        type Output;
    }

    impl<T> HasOutput for fn() -> T {
        type Output = T;
    }
}

/// A runtime value erased from no Rust type: the canonical form of every
/// `Erased` (`Canonical`), and a type variable's run-time instantiation.
pub type Owned<R> = Erased<R, Never>;

impl<R> Owned<R>
where
    R: Runtime,
{
    /// The runtime's and the glue's: a handler never names an `Owned`
    /// (RFC-0068 rule 4). `Owned` names no type its value was erased from,
    /// so this claims only that the holder owns the word.
    ///
    /// # Safety
    /// No other holder owns `value`: it was moved out of the one that did,
    /// or it owns nothing. `R::Value` is `Copy`, so a word read out of a
    /// live holder (`Owned`'s `Deref`) and made an `Owned` here would be
    /// released twice, and a word kept past the call it was lent to would
    /// come back as a value holding a loan that ended (RFC-0079 rule 8).
    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn from_value(_: Holding<'_, R>, value: R::Value) -> Self {
        Self::holding(value)
    }

    /// `value` erased into a holder of its own: `erase` consumes it, so no
    /// other holder owns the word.
    #[inline(always)]
    pub fn erased<T, Rep>(rt: Crossing<'_, R>, value: T) -> Self
    where
        T: crate::OneValue<R, Rep>,
    {
        Self::holding(value.erase(rt))
    }

    /// A holder of the runtime's default value, which owns nothing. Written
    /// on `Owned` alone and not as `Default`: an `Erased<R, T>` names the
    /// type its value was erased from, and a default value was erased from
    /// none.
    #[inline(always)]
    pub fn vacant(_: Holding<'_, R>) -> Self {
        Self::holding(R::Value::default())
    }

    /// The held value, to replace in place. On `Owned` alone and not as
    /// `DerefMut`, for `vacant`'s reason: a value written into an
    /// `Erased<R, T>` would have to be one erased from a `T`.
    ///
    /// # Safety
    /// A word written through the result is owned by no other holder, as
    /// `from_value`'s, and the word it replaces is released or moved out
    /// by the caller.
    #[inline(always)]
    pub unsafe fn value_mut(&mut self, _: Holding<'_, R>) -> &mut R::Value {
        self.held_mut()
    }
}

/// The word an `Owned` holds, read in place. `Owned` is what the glue fills
/// a handler's type variable with, and a receiver bound
/// `I: Deref<Target = Rt::Value>` is how a handler names that the storage
/// it lends an instance call holds one word (RFC-0068 rule 2). Only `Owned`
/// has it: an `Erased<R, T>` is read as its `T`, by `as_ref`, `get` and
/// `get_ref`.
impl<R> Deref for Owned<R>
where
    R: Runtime,
{
    type Target = R::Value;

    #[inline(always)]
    fn deref(&self) -> &R::Value {
        self.word()
    }
}

#[doc(hidden)]
/// The runtime's values behind a run of `Owned`s, for a caller that fills a
/// destination it owns (RFC-0050 rule 5). `Owned<R>` is `#[repr(transparent)]`
/// over `R::Value`, so the two slices have one layout.
///
/// # Safety
/// Every slot of `values` owns nothing — it is `Owned::vacant()` or was
/// taken out — so a write through the result releases no live value.
pub unsafe fn lend_run<'v, R>(_: Holding<'_, R>, values: &'v mut [Owned<R>]) -> &'v mut [R::Value]
where
    R: Runtime,
{
    // SAFETY: the caller's contract: no slot owns a value, so a write
    // through the result releases none, and whatever it writes, a holder
    // then owns.
    unsafe { Owned::<R>::over_value().flip().cast_slice_mut(values) }
}
