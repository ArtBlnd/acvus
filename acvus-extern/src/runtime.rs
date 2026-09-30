//! The runtime contract: what every executor implements and every extern fn
//! is written against, generic and dispatched statically (runtime contract).
//!
//! No method here takes a Rust value and returns a runtime value, or reads a
//! runtime value, except under a `Settled`, which only an entry makes: a value
//! is made and read only by the crossing, at the type the checker settled.
//!
//! The runtime makes and reads values by kind: an opaque value, a primitive,
//! text, an option, a result and a tuple. A type's crossing (`Crosses`) is
//! its own: it composes the kinds over its parts' crossings, so the runtime
//! never sees a type's structure, only the kind's.

use std::ops::{Deref, DerefMut};
use std::time::Duration;

use acvus_utils::Interner;

use crate::boundary::entry::Settled;
use crate::boundary::seal::Crossings;
use crate::boundary::signed::{SignedS1, SignedS2, SignedS3, Structural};
use crate::boundary::uniform::{Primitive, Release};

/// A type that crosses at the runtime `R`: one of the language's types, or an
/// extern type the derive declares. Each crossing is its type's own and is
/// sealed: this crate writes the language's, and the derive an extern
/// type's. `give` makes the runtime's value of `self`, and `take` moves one
/// back out, each by its kind's operations over its parts' crossings.
///
/// A type that crosses may carry lifetimes; a value that holds a loan holds it
/// as a `Loaned`, never as a Rust reference.
///
/// A type neither declares does not cross, nor does a sequence, which is the
/// runtime's own and is lent as a slice:
///
/// ```compile_fail,E0277
/// use acvus_extern::context::Returning;
/// use acvus_extern::done::Done;
/// use acvus_extern::runtime::Runtime;
///
/// struct Undeclared;
///
/// fn finish<R: Runtime>(cx: Returning<'_, R>) -> Done<R> {
///     cx.finish(Undeclared)
/// }
/// ```
///
/// ```compile_fail,E0277
/// use acvus_extern::context::Returning;
/// use acvus_extern::done::Done;
/// use acvus_extern::runtime::Runtime;
///
/// fn finish<R: Runtime>(cx: Returning<'_, R>) -> Done<R> {
///     cx.finish(vec![1i64])
/// }
/// ```
pub trait Crosses<R: Runtime>: Sized + Send + Sync + Crossings {
    fn give(self, rt: &R, at: Settled<'_>) -> R::Value;
    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self;
}

/// An extern type, held by the runtime as the Rust value itself: the one kind
/// lent in place (`Runtime::shared`, `Runtime::exclusive`), since only its
/// storage is a `Self`. Only the derive declares one.
pub trait Opaque: Send + Sync + Structural + Crossings {}

/// A lend is shared or exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutability {
    Shared,
    Exclusive,
}

/// What a reference value is made from: a part of storage the call lent,
/// borrowed from the guard that lends it (`context::Returning::finish_lent`);
/// a writable one only from an exclusive guard. The part is the whole lent
/// value, a range or an element of it, a field, or storage the lent value
/// owns, such as a `String`'s text (runtime contract §2, References).
pub enum RefTarget<'a, P: ?Sized> {
    Shared(&'a P),
    Exclusive(&'a mut P),
}

pub trait Runtime: Sized + Send + Sync + 'static + SignedS1 + SignedS2 + SignedS3 {
    type Value: Release + Send + Sync + 'static;

    /// A shared lend of an opaque value's storage; its borrow is the lend.
    /// Every guard is `Send`, as what it lends is, so an awaited body holds
    /// one across a wait (macros §1).
    type Shared<'a, T: Opaque + 'a>: Deref<Target = T> + Send;
    /// An exclusive lend of an opaque value's storage; its drop is the loan's
    /// end, and what the drop does is a constant of `T` (K13).
    type Exclusive<'a, T: Opaque + 'a>: DerefMut<Target = T> + Send;

    /// Each kind's value made, and moved back out at its settled type. A take
    /// consumes the value: its storage is released, and the parts it hands
    /// back are the caller's.
    fn give_opaque<T: Opaque>(&self, value: T, at: Settled<'_>) -> Self::Value;
    fn take_opaque<T: Opaque>(&self, v: Self::Value, at: Settled<'_>) -> T;

    fn give_primitive<P: Primitive>(&self, value: P, at: Settled<'_>) -> Self::Value;
    fn take_primitive<P: Primitive>(&self, v: Self::Value, at: Settled<'_>) -> P;

    fn give_string(&self, value: String, at: Settled<'_>) -> Self::Value;
    fn take_string(&self, v: Self::Value, at: Settled<'_>) -> String;

    fn give_option(&self, value: Option<Self::Value>, at: Settled<'_>) -> Self::Value;
    fn take_option(&self, v: Self::Value, at: Settled<'_>) -> Option<Self::Value>;

    fn give_result(&self, value: Result<Self::Value, Self::Value>, at: Settled<'_>) -> Self::Value;
    fn take_result(&self, v: Self::Value, at: Settled<'_>) -> Result<Self::Value, Self::Value>;

    fn give_tuple<const N: usize>(&self, parts: [Self::Value; N], at: Settled<'_>) -> Self::Value;
    fn take_tuple<const N: usize>(&self, v: Self::Value, at: Settled<'_>) -> [Self::Value; N];

    /// The guards lend an opaque value's storage from the value's place
    /// (runtime contract §2, Guards).
    fn shared<'a, T: Opaque + 'a>(&'a self, v: &'a Self::Value, at: Settled<'a>) -> Self::Shared<'a, T>;
    fn exclusive<'a, T: Opaque + 'a>(&'a self, v: &'a mut Self::Value, at: Settled<'a>) -> Self::Exclusive<'a, T>;

    /// The reference value made from a part of storage the call lent; a
    /// borrowed result is one (crossing §4).
    fn reference<'a, P: ?Sized>(&'a self, target: RefTarget<'a, P>, at: Settled<'a>) -> Self::Value;
    /// What a reference value names, read back for the loan `'l` it was made
    /// under: by S2 it names its target until that loan ends (runtime
    /// contract §2, Typed reads).
    fn target<'l, P: ?Sized>(&'l self, v: Self::Value, at: Settled<'l>) -> &'l P;
    fn target_mut<'l, P: ?Sized>(&'l self, v: Self::Value, at: Settled<'l>) -> &'l mut P;

    fn interner(&self) -> &Interner;
    fn sleep(&self, d: Duration) -> impl Future<Output = ()> + Send;
}
