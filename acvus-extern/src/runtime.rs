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
pub trait Crosses: Sized + Structural {}

/// A lend is shared or exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutability {
    Shared,
    Exclusive,
}

/// What a reference value is made from; a writable one only from an exclusive
/// borrow of its target.
pub enum RefTarget<'a, V> {
    Shared(&'a V),
    Exclusive(&'a mut V),
}

/// A tuple's body: its parts, each an owning holder of a runtime value.
pub trait TupleBody<V>: Send + Sync + 'static {
    fn arity(&self) -> u32;
    fn take_part(&mut self, i: u32) -> V;
}

/// An object's body: its fields in the settled order, each an owning holder.
pub trait ObjectBody<V>: Send + Sync + 'static {
    fn len(&self) -> u32;
    fn take_field(&mut self, i: u32) -> V;
}

/// A variant's body: its tag and its payload as an owning holder.
pub trait VariantBody<V>: Send + Sync + 'static {
    fn tag(&self) -> u32;
    fn take_payload(&mut self) -> Option<V>;
}

pub trait Runtime: Sized + Send + Sync + 'static + SignedS1 + SignedS2 + SignedS3 {
    type Value: Release + Send + Sync + 'static;
    type Tuple: TupleBody<Self::Value>;
    type Object: ObjectBody<Self::Value>;
    type Variant: VariantBody<Self::Value>;

    /// A shared lend of storage; its borrow is the lend.
    type Shared<'a, T: Crosses + 'a>: Deref<Target = T>;
    /// An exclusive lend of storage; its drop is the loan's end, and what the
    /// drop does is a constant of `T` (K13).
    type Exclusive<'a, T: Crosses + 'a>: DerefMut<Target = T>;
    /// A program's array lent as a slice.
    type SharedSlice<'a, T: Crosses + 'a>: Deref<Target = [T]>;
    type ExclusiveSlice<'a, T: Crosses + 'a>: DerefMut<Target = [T]>;
    /// A `String`'s bytes lent as text.
    type SharedStr<'a>: Deref<Target = str>;

    /// Moves a value out at its settled type.
    fn take<T: Crosses>(&self, v: Self::Value, at: Settled<'_>) -> T;
    /// Makes the runtime value of a result at its settled type.
    fn give<T: Crosses>(&self, value: T, at: Settled<'_>) -> Self::Value;

    fn shared<'a, T: Crosses + 'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::Shared<'a, T>;
    fn exclusive<'a, T: Crosses + 'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::Exclusive<'a, T>;
    fn shared_slice<'a, T: Crosses + 'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::SharedSlice<'a, T>;
    fn exclusive_slice<'a, T: Crosses + 'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::ExclusiveSlice<'a, T>;
    fn shared_str<'a>(&'a self, v: Self::Value, at: Settled<'a>) -> Self::SharedStr<'a>;

    fn take_tuple(&self, v: Self::Value, at: Settled<'_>) -> Self::Tuple;
    fn take_object(&self, v: Self::Value, at: Settled<'_>) -> Self::Object;
    fn take_variant(&self, v: Self::Value, at: Settled<'_>) -> Self::Variant;

    /// The reference value a borrowed result is, made from the guard it
    /// borrows (crossing §4).
    fn lent_shared<'a, T: Crosses + 'a>(&'a self, guard: Self::Shared<'a, T>) -> Self::Value;
    fn lent_exclusive<'a, T: Crosses + 'a>(&'a self, guard: Self::Exclusive<'a, T>) -> Self::Value;

    /// A reference value made from its target.
    fn reference(&self, target: RefTarget<'_, Self::Value>) -> Self::Value;

    fn interner(&self) -> &Interner;
    fn sleep(&self, d: Duration) -> impl Future<Output = ()> + Send;
}
