//! Ownership at the boundary (RFC-0048 rules 1 and 7).
//!
//! The ABI — handler signatures, `Ref`, `Elements`, the glue's window —
//! passes `R::Value` and owes nothing. `Owned<R>` is the only holder that
//! owes a release, and the conversion between the two happens at the
//! glue the macro emits, never inside a handler body.

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
    /// (RFC-0068 rule 4). Only `Owned` is made from a bare value without
    /// `unsafe`: any other `Erased` names the type its value was erased
    /// from, and that is `FromValue::from_value`'s claim.
    #[doc(hidden)]
    #[inline(always)]
    pub fn from_value(value: R::Value) -> Self {
        Self::holding(value)
    }

    /// A holder of the runtime's default value, which owns nothing. Written
    /// on `Owned` alone and not as `Default`: an `Erased<R, T>` names the
    /// type its value was erased from, and a default value was erased from
    /// none.
    #[inline(always)]
    pub fn vacant() -> Self {
        Self::holding(R::Value::default())
    }

    /// The held value, to replace in place. On `Owned` alone and not as
    /// `DerefMut`, for `vacant`'s reason: a value written into an
    /// `Erased<R, T>` would have to be one erased from a `T`.
    #[inline(always)]
    pub fn value_mut(&mut self) -> &mut R::Value {
        self.held_mut()
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
pub unsafe fn lend_run<R>(values: &mut [Owned<R>]) -> &mut [R::Value]
where
    R: Runtime,
{
    let len = values.len();
    // SAFETY: the `repr(transparent)` stated above, and the caller's contract
    // for what the slots hold.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<R::Value>(), len) }
}
