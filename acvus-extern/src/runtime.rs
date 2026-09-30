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

/// What a reference value is made from: a part of storage the call lent,
/// borrowed from the guard that lends it (`context::Returning::finish_lent`);
/// a writable one only from an exclusive guard. The part is the whole lent
/// value, a range or an element of it, a field, or storage the lent value
/// owns, such as a `String`'s text (runtime contract §2, References).
pub enum RefTarget<'a, P: ?Sized> {
    Shared(&'a P),
    Exclusive(&'a mut P),
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

    /// Makes an aggregate at its settled type from its parts' values, which
    /// it then owns (runtime contract §2, Aggregates).
    fn make_tuple(&self, parts: impl ExactSizeIterator<Item = Self::Value>, at: Settled<'_>) -> Self::Value;
    fn make_object(&self, fields: impl ExactSizeIterator<Item = Self::Value>, at: Settled<'_>) -> Self::Value;
    fn make_variant(&self, tag: u32, payload: Option<Self::Value>, at: Settled<'_>) -> Self::Value;

    /// The `i`-th part of an aggregate the call lent, its storage lent in
    /// place: the aggregate still owns it, and it is read only through a
    /// guard, so a projection is a guard at each leaf. A variant's payload is
    /// its part 0.
    fn lent_part<'a>(&'a self, v: Self::Value, i: u32, at: Settled<'a>) -> Self::Value;
    /// The tag of a variant the call lent.
    fn lent_tag(&self, v: Self::Value, at: Settled<'_>) -> u32;

    /// The reference value made from a part of storage the call lent; a
    /// borrowed result is one (crossing §4).
    fn reference<'a, P: ?Sized>(&'a self, target: RefTarget<'a, P>, at: Settled<'a>) -> Self::Value;

    fn interner(&self) -> &Interner;
    fn sleep(&self, d: Duration) -> impl Future<Output = ()> + Send;
}
