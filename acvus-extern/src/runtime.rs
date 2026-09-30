//! The runtime contract: what every executor implements and every extern fn
//! is written against, generic and dispatched statically (runtime contract).
//!
//! No method here takes a Rust value and returns a runtime value, or reads a
//! runtime value, except under a `Settled`, which only an entry makes: a value
//! is made and read only by the crossing, at the type the checker settled.

use std::ops::{Deref, DerefMut};
use std::time::Duration;

use acvus_utils::Interner;

use crate::boundary::entry::Settled;
use crate::boundary::signed::{SignedS1, SignedS2, SignedS3, Structural};
use crate::boundary::uniform::Release;

/// A type a checker-settled site may be read at. It may carry lifetimes; a
/// value that holds a loan holds it as a `Loaned`, never as a Rust reference.
/// It is `Send + Sync`, as the runtime value it becomes is. Every `Structural`
/// type that is `Send + Sync` crosses, by the one implementation below, so no
/// author implements it (safety §1): an author signs only `Structural`,
/// through the derive.
pub trait Crosses: Sized + Send + Sync + Structural {}

impl<T> Crosses for T where T: Structural + Send + Sync {}

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

    /// A shared lend of storage; its borrow is the lend. Every guard is
    /// `Send`, as what it lends is (`Crosses`), so an awaited body holds one
    /// across a wait (macros §1).
    type Shared<'a, T: Crosses + 'a>: Deref<Target = T> + Send;
    /// An exclusive lend of storage; its drop is the loan's end, and what the
    /// drop does is a constant of `T` (K13).
    type Exclusive<'a, T: Crosses + 'a>: DerefMut<Target = T> + Send;
    /// A program's array lent as a slice.
    type SharedSlice<'a, T: Crosses + 'a>: Deref<Target = [T]> + Send;
    type ExclusiveSlice<'a, T: Crosses + 'a>: DerefMut<Target = [T]> + Send;
    /// A `String`'s bytes lent as text.
    type SharedStr<'a>: Deref<Target = str> + Send;

    /// Moves a value of an opaque type out at its settled type, releasing
    /// its storage. It is reached only through `Uniform`'s take, where a
    /// handler needs the value itself; a value only dropped is released and
    /// never taken (runtime contract §1).
    fn take<T: Crosses>(&self, v: Self::Value, at: Settled<'_>) -> T;
    /// Makes the runtime value of an opaque type at its settled type.
    fn give<T: Crosses>(&self, value: T, at: Settled<'_>) -> Self::Value;

    /// The guards lend storage from the value's place, so a value held in the
    /// place itself, as a scalar in a word, is lent in place (runtime contract
    /// §2, Guards).
    fn shared<'a, T: Crosses + 'a>(&'a self, v: &'a Self::Value, at: Settled<'a>) -> Self::Shared<'a, T>;
    fn exclusive<'a, T: Crosses + 'a>(&'a self, v: &'a mut Self::Value, at: Settled<'a>) -> Self::Exclusive<'a, T>;
    fn shared_slice<'a, T: Crosses + 'a>(&'a self, v: &'a Self::Value, at: Settled<'a>) -> Self::SharedSlice<'a, T>;
    fn exclusive_slice<'a, T: Crosses + 'a>(
        &'a self,
        v: &'a mut Self::Value,
        at: Settled<'a>,
    ) -> Self::ExclusiveSlice<'a, T>;
    fn shared_str<'a>(&'a self, v: &'a Self::Value, at: Settled<'a>) -> Self::SharedStr<'a>;

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
