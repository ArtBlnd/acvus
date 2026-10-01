//! The runtime contract: what every executor implements and every extern fn
//! is written against, generic and dispatched statically (runtime contract).
//!
//! No operation here takes a Rust value and returns a runtime value, or reads a
//! runtime value, except under a `Settled`, which only an entry makes: a value
//! is made and read only by a crossing, at the type the checker settled or
//! held at its own type (`Uniform::new`), never erased.
//!
//! The runtime makes, takes and lends values by kind
//! (`OpaqueKind`, `PrimitiveKind`, `StringKind`, `OptionKind`, `ResultKind`,
//! `TupleKind`, `RefKind`), each kind its own associated type, so a runtime
//! implements each kind for its own representation. A type's crossing
//! (`Crosses`) is its own: it composes its kind's operations over its parts'
//! crossings, and the runtime never sees a type's structure, only the kind's.

use std::time::Duration;

use acvus_utils::Interner;

use crate::boundary::entry::Settled;
use crate::boundary::seal::Crossing;
use crate::boundary::signed::{SignedS1, SignedS2, SignedS3, Structural};
use crate::boundary::uniform::{Primitive, Release};

/// A type that crosses at the runtime `R`: one of the language's types, or an
/// extern type the derive declares. Each crossing is its type's own and is
/// sealed: this crate writes the language's, and the derive an extern type's.
///
/// `give` makes the runtime's value of `self` and `take` moves one back out.
/// `open` lends a held value's place as the type's projection, and
/// `open_ref` the storage a reference value names: `&T` for a type the
/// runtime holds as itself, and for an option, a result or a tuple the same
/// structure over its parts' projections (crossing §2).
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
pub trait Crosses<R: Runtime>: Sized + Send + Sync + Crossing {
    type Open<'a>
    where
        Self: 'a,
        R: 'a;
    type OpenMut<'a>
    where
        Self: 'a,
        R: 'a;

    fn give(self, rt: &R, at: Settled<'_>) -> R::Value;
    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Self;

    fn open<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> Self::Open<'a>
    where
        Self: 'a;
    fn open_mut<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
    where
        Self: 'a;

    /// The storage the reference value `reference` names, lent as the view.
    fn open_ref<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::Open<'a>
    where
        Self: 'a;
    fn open_ref_mut<'a>(rt: &'a R, reference: R::Value, at: Settled<'a>) -> Self::OpenMut<'a>
    where
        Self: 'a;
}

/// An extern type, held by the runtime as the Rust value itself. Only the
/// derive declares one, `#[derive(Opaque)]`, reading the definition: a field
/// naming a parameter is `Structural`, so one holding it under interior
/// mutability, or a Rust reference a loan would be kept as, is refused:
///
/// ```compile_fail,E0277
/// #[derive(acvus_extern::Opaque)]
/// pub struct Interior<T> { cell: std::cell::Cell<T> }
/// ```
///
/// ```compile_fail,E0277
/// #[derive(acvus_extern::Opaque)]
/// pub struct Borrowing<'a> { text: &'a str }
/// ```
///
/// A type parameter is unbounded, parametric, and no parameter is a const:
///
/// ```compile_fail
/// #[derive(acvus_extern::Opaque)]
/// pub struct Bounded<T: Clone> { item: T }
/// ```
///
/// ```compile_fail
/// #[derive(acvus_extern::Opaque)]
/// pub struct Fixed<const N: usize> { items: [u8; N] }
/// ```
pub trait Opaque: Send + Sync + Structural + Crossing + Referent {}

/// What a reference value is made to from Rust storage: storage whose Rust
/// type is the type the reference is read back at. A type the runtime holds as
/// itself (an extern type, a primitive, a `String`), text, a type variable's
/// value and a run of them are; an option, a result and a tuple, which the
/// runtime holds in its own representation, are not, so no Rust storage of
/// one is lent as a reference. Sealed: only this crate and the derive declare
/// one.
pub trait Referent: Crossing {}

/// What a reference value is made from: a Rust borrow, at a lifetime within
/// the call, of storage that lives until the loan it is labelled with ends
/// (`context::Returning::finish_borrowed`); a writable one only from an
/// exclusive borrow. The storage is what an argument lent, whole or any part
/// of it, or what a kept loan holds (runtime contract §2, References).
pub enum RefTarget<'a, P: ?Sized> {
    Shared(&'a P),
    Exclusive(&'a mut P),
}

pub trait Runtime: Sized + Send + Sync + 'static + SignedS1 + SignedS2 + SignedS3 {
    type Value: Release + Send + Sync + 'static;

    type Opaque: OpaqueKind<Self>;
    type Primitive: PrimitiveKind<Self>;
    type String: StringKind<Self>;
    type Option: OptionKind<Self>;
    type Result: ResultKind<Self>;
    type Tuple: TupleKind<Self>;
    type Ref: RefKind<Self>;

    fn interner(&self) -> &Interner;
    fn sleep(&self, d: Duration) -> impl Future<Output = ()> + Send;
}

/// An extern type's value, holding the Rust value itself, whose storage is lent
/// as a reference to it.
pub trait OpaqueKind<R: Runtime> {
    fn give<T: Opaque>(rt: &R, value: T, at: Settled<'_>) -> R::Value;
    /// Moves the value out, releasing its storage.
    fn take<T: Opaque>(rt: &R, v: R::Value, at: Settled<'_>) -> T;

    fn shared<'a, T: Opaque + 'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> &'a T;
    fn exclusive<'a, T: Opaque + 'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> &'a mut T;
}

/// A primitive's value, the primitive in the runtime's own representation.
pub trait PrimitiveKind<R: Runtime> {
    fn give<P: Primitive>(rt: &R, value: P, at: Settled<'_>) -> R::Value;
    fn take<P: Primitive>(rt: &R, v: R::Value, at: Settled<'_>) -> P;

    fn shared<'a, P: Primitive>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> &'a P;
    fn exclusive<'a, P: Primitive>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> &'a mut P;
}

/// Text, a Rust `String`.
pub trait StringKind<R: Runtime> {
    fn give(rt: &R, value: String, at: Settled<'_>) -> R::Value;
    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> String;

    fn shared<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> &'a String;
    fn exclusive<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> &'a mut String;
}

/// An option over its part's value. Its release releases the part. Lent, it
/// is its part, if present, in place.
pub trait OptionKind<R: Runtime> {
    fn give(rt: &R, value: Option<R::Value>, at: Settled<'_>) -> R::Value;
    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Option<R::Value>;
    fn shared<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> Option<&'a R::Value>;
    fn exclusive<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> Option<&'a mut R::Value>;
}

/// A result over its ok part's or its error part's value.
pub trait ResultKind<R: Runtime> {
    fn give(rt: &R, value: Result<R::Value, R::Value>, at: Settled<'_>) -> R::Value;
    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Result<R::Value, R::Value>;
    fn shared<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> Result<&'a R::Value, &'a R::Value>;
    fn exclusive<'a>(
        rt: &'a R,
        place: &'a mut R::Value,
        at: Settled<'a>,
    ) -> Result<&'a mut R::Value, &'a mut R::Value>;
}

/// A tuple over its `N` parts' values.
pub trait TupleKind<R: Runtime> {
    fn give<const N: usize>(rt: &R, parts: [R::Value; N], at: Settled<'_>) -> R::Value;
    fn take<const N: usize>(rt: &R, v: R::Value, at: Settled<'_>) -> [R::Value; N];
    fn shared<'a, const N: usize>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> &'a [R::Value; N];
    fn exclusive<'a, const N: usize>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> &'a mut [R::Value; N];
}

/// A reference value, naming storage the call lent.
pub trait RefKind<R: Runtime> {
    /// The reference value made from a part of storage the call lent; a
    /// borrowed result is one (crossing §4).
    fn reference<'a, P: ?Sized>(rt: &'a R, target: RefTarget<'a, P>, at: Settled<'a>) -> R::Value;
    /// What a reference value names, read back for the loan `'l` it was made
    /// under: by S2 it names its target until that loan ends (runtime
    /// contract §2, Typed reads).
    fn target<'l, P: ?Sized>(rt: &'l R, v: R::Value, at: Settled<'l>) -> &'l P;
    fn target_mut<'l, P: ?Sized>(rt: &'l R, v: R::Value, at: Settled<'l>) -> &'l mut P;
    /// The runtime's value a reference value names, where its target is
    /// held in the runtime's own representation (an option, a result, a
    /// tuple, a type variable's value, a function value), read back as that
    /// value's place for the loan `'l`, as `target` reads storage of a Rust
    /// type.
    fn place<'l>(rt: &'l R, v: R::Value, at: Settled<'l>) -> &'l R::Value;
    fn place_mut<'l>(rt: &'l R, v: R::Value, at: Settled<'l>) -> &'l mut R::Value;
}
