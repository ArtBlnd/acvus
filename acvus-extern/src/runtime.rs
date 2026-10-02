//! The runtime contract: what every executor implements and every extern fn
//! is written against, generic and dispatched statically (runtime contract).
//!
//! No operation here takes a Rust value and returns a runtime value, or reads a
//! runtime value, except under a `Settled`, which only an entry makes: a value
//! is made and read only by a crossing, at the type the checker settled or
//! held at its own type (`Uniform::new`), never erased.
//!
//! The runtime makes, takes and views values by kind (`OpaqueKind`,
//! `PrimitiveKind`, `StringKind`, `OptionKind`, `ResultKind`, `TupleKind`),
//! makes and reads references to values' places (`RefKind`), and makes and
//! reads views of runs and of text (`ViewKind`), each its own associated type, so a runtime implements each
//! for its own representation. `R` is generic, so every operation is
//! inlined at a call: a value held in place is read in a register, one held
//! apart by one pointer step.
//!
//! What comes into a call is opened as the Rust type its place writes
//! (`Open`): a value, a view of a place, a box, a reference value, or a
//! container of these. What goes out is a runtime value (`Give`): a box
//! (`Uniform`), a reference value (`Ref`, `Mut`), or a container of these. A
//! reference value always names a value's place: the runtime makes one only
//! from a box's place, so its lifetime is the runtime's borrow.

use std::marker::PhantomData;
use std::time::Duration;

use acvus_utils::Interner;

use crate::boundary::entry::Settled;
use crate::boundary::seal::Crossing;
use crate::boundary::signed::{SignedS1, SignedS2, SignedS3, Structural};
use crate::boundary::uniform::{Primitive, Release};
use crate::decl::{DeclTy, DeclType};

/// What comes into a call, opened as the Rust type its place writes, at the
/// loan `'l`: a value taken, a view of the place a reference names, a box, a
/// reference value kept, or a container of these (crossing §2).
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a form a value comes in as",
    label = "no declared type",
    note = "a value comes in as itself, as a view (`&X`, `&mut X`) of a place, as a box `Uniform<R, X>`, as a reference value `Ref`, `Mut`, or as an `Option`, a `Result` or a tuple of these",
    note = "a reference to an option, a result or a tuple is lent as its parts' references, the program's `as_ref`"
)]
pub trait Open<'l, R: Runtime>: Sized + Crossing {
    /// Its declared type.
    fn decl(names: &Interner) -> DeclTy;
    #[doc(hidden)]
    fn open(rt: &'l R, v: R::Value, at: Settled<'l>) -> Self;
}

/// What goes out of a call, a runtime value at the loan `'l`: a box, a
/// reference value, or a container of these (crossing §2). A Rust value goes
/// out only boxed, `Uniform::new`, and a reference only as a reference value
/// the runtime made, `Ref::of`:
///
/// ```compile_fail,E0277
/// use acvus_extern::executor::Executor;
/// use acvus_extern::done::Done;
/// use acvus_extern::runtime::Runtime;
///
/// fn finish<R: Runtime>(cx: Executor<'_, R>) -> Done<R> {
///     cx.finish(3i64)
/// }
/// ```
///
/// A type neither declares does not cross:
///
/// ```compile_fail,E0277
/// use acvus_extern::executor::Executor;
/// use acvus_extern::done::Done;
/// use acvus_extern::runtime::Runtime;
///
/// struct Undeclared;
///
/// fn finish<R: Runtime>(cx: Executor<'_, R>) -> Done<R> {
///     cx.finish(Undeclared)
/// }
/// ```
///
/// ```compile_fail,E0277
/// use acvus_extern::executor::Executor;
/// use acvus_extern::done::Done;
/// use acvus_extern::runtime::Runtime;
///
/// fn finish<R: Runtime>(cx: Executor<'_, R>) -> Done<R> {
///     cx.finish(vec![1i64])
/// }
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a form a value goes out as",
    label = "not a runtime value",
    note = "a value goes out as a box `Uniform<R, X>` (`Uniform::new`), a reference value `Ref`, `Mut` (`Ref::of`), or an `Option`, a `Result` or a tuple of these"
)]
pub trait Give<'l, R: Runtime>: Sized + Crossing {
    /// Its declared type.
    fn decl(names: &Interner) -> DeclTy;
    #[doc(hidden)]
    fn give(self, rt: &'l R, at: Settled<'l>) -> R::Value;
}

/// What a reference value names, viewed as this Rust type: a value's place
/// viewed in place by its kind (a primitive, text, an extern-defined value, a
/// box whose values have no Rust view of their own), or a view of a run or of
/// text (a slice of boxes, `str`).
pub trait Referent<R: Runtime>: Crossing + DeclType {
    #[doc(hidden)]
    fn view<'l>(rt: &'l R, reference: R::Value, at: Settled<'l>) -> &'l Self
    where
        Self: 'l;
    #[doc(hidden)]
    fn view_mut<'l>(rt: &'l R, reference: R::Value, at: Settled<'l>) -> &'l mut Self
    where
        Self: 'l;
}

/// A part of a function type, named with no loan of its own (crossing §2):
/// at the call's loan `'l` it is given as `Give` and opened as `Open`. A value
/// type is given boxed and opened as itself; `RefKey<X>` is given as
/// `Ref<'l, R, X>` and opened as `&'l X`; `Func<A, U, E>` is a function value,
/// which holds what the call lent (compiler: types §5); an extern-defined
/// type with lifetimes is at `'l` whatever lifetime it is written with.
pub trait Key<'l, R: Runtime>: DeclType {
    type Give: Give<'l, R>;
    type Open: Open<'l, R>;
}

/// A Rust type written in a signature at `'l`, named as a key: how a generic
/// function's signature, written in Rust, is stated in its keys. A key names
/// a declared type, so a box at a value type and the value type name one key:
/// `[Uniform<R, i64>]` and `[i64]` are both `[i64]`.
pub trait Place<'l, R: Runtime> {
    type Key: ?Sized;
}

/// How a declared type, written as its key, is viewed where a reference
/// names it: a value type as itself, a type variable as its box, a run of
/// values as the slice of its elements as held, text as `str`.
pub trait Viewed<R: Runtime>: DeclType {
    type View: ?Sized;
}

/// How a value of a declared type, written as its key, is held in a
/// container: a value type boxed, a type variable as its box.
pub trait Repr<R: Runtime>: Viewed<R> {
    type Held;
}

/// The declared type `&X` as a function type's part.
pub struct RefKey<X: ?Sized>(PhantomData<fn() -> *const X>);

/// The declared type `&mut X` as a function type's part.
pub struct MutKey<X: ?Sized>(PhantomData<fn() -> *const X>);

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
pub trait Opaque: Send + Sync + Structural + Crossing {}

pub trait Runtime: Sized + Send + Sync + 'static + SignedS1 + SignedS2 + SignedS3 {
    type Value: Release + Send + Sync + 'static;

    type Opaque: OpaqueKind<Self>;
    type Primitive: PrimitiveKind<Self>;
    type String: StringKind<Self>;
    type Option: OptionKind<Self>;
    type Result: ResultKind<Self>;
    type Tuple: TupleKind<Self>;
    type View: ViewKind<Self>;
    type Ref: RefKind<Self>;

    fn interner(&self) -> &Interner;
    fn sleep(&self, d: Duration) -> impl Future<Output = ()> + Send;
}

/// An extern type's value, holding the Rust value itself, viewed in place.
pub trait OpaqueKind<R: Runtime> {
    fn give<T: Opaque>(rt: &R, value: T, at: Settled<'_>) -> R::Value;
    /// Moves the value out, releasing its storage.
    fn take<T: Opaque>(rt: &R, v: R::Value, at: Settled<'_>) -> T;

    fn shared<'a, T: Opaque + 'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> &'a T;
    fn exclusive<'a, T: Opaque + 'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> &'a mut T;
}

/// A primitive's value, the primitive in the runtime's own representation,
/// viewed in place as the Rust primitive.
pub trait PrimitiveKind<R: Runtime> {
    fn give<P: Primitive>(rt: &R, value: P, at: Settled<'_>) -> R::Value;
    fn take<P: Primitive>(rt: &R, v: R::Value, at: Settled<'_>) -> P;

    fn shared<'a, P: Primitive>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> &'a P;
    fn exclusive<'a, P: Primitive>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> &'a mut P;
}

/// Text, a Rust `String`, viewed in place.
pub trait StringKind<R: Runtime> {
    fn give(rt: &R, value: String, at: Settled<'_>) -> R::Value;
    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> String;

    fn shared<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> &'a String;
    fn exclusive<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> &'a mut String;
}

/// An option over its part's value. Its release releases the part.
pub trait OptionKind<R: Runtime> {
    fn give(rt: &R, value: Option<R::Value>, at: Settled<'_>) -> R::Value;
    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Option<R::Value>;
}

/// A result over its ok part's or its error part's value.
pub trait ResultKind<R: Runtime> {
    fn give(rt: &R, value: Result<R::Value, R::Value>, at: Settled<'_>) -> R::Value;
    fn take(rt: &R, v: R::Value, at: Settled<'_>) -> Result<R::Value, R::Value>;
}

/// A tuple over its `N` parts' values.
pub trait TupleKind<R: Runtime> {
    fn give<const N: usize>(rt: &R, parts: [R::Value; N], at: Settled<'_>) -> R::Value;
    fn take<const N: usize>(rt: &R, v: R::Value, at: Settled<'_>) -> [R::Value; N];
}

/// A view: a run of values' places or a range of text, a reference value the
/// program lends, or made from a `String`'s place and a range, or narrowed
/// from another view (runtime contract §2, Views). A view is read back as the
/// run of its places, or as `str`, for the loan `'l`.
pub trait ViewKind<R: Runtime> {
    fn text<'a>(rt: &'a R, base: &'a R::Value, from: usize, to: usize, at: Settled<'a>) -> R::Value;
    fn text_mut<'a>(rt: &'a R, base: &'a mut R::Value, from: usize, to: usize, at: Settled<'a>) -> R::Value;
    fn narrow<'a>(rt: &'a R, view: R::Value, from: usize, to: usize, at: Settled<'a>) -> R::Value;

    fn places<'l>(rt: &'l R, view: R::Value, at: Settled<'l>) -> &'l [R::Value];
    fn places_mut<'l>(rt: &'l R, view: R::Value, at: Settled<'l>) -> &'l mut [R::Value];
    fn str<'l>(rt: &'l R, view: R::Value, at: Settled<'l>) -> &'l str;
    fn str_mut<'l>(rt: &'l R, view: R::Value, at: Settled<'l>) -> &'l mut str;
}

/// A reference value: made by the runtime from a value's place the call
/// borrows, a writable one only from an exclusive borrow, and read back as
/// that place for the loan `'l`. By S2 it names its place until the
/// checker's loan ends (runtime contract §2, References).
pub trait RefKind<R: Runtime> {
    fn of<'a>(rt: &'a R, place: &'a R::Value, at: Settled<'a>) -> R::Value;
    fn of_mut<'a>(rt: &'a R, place: &'a mut R::Value, at: Settled<'a>) -> R::Value;
    fn place<'l>(rt: &'l R, v: R::Value, at: Settled<'l>) -> &'l R::Value;
    fn place_mut<'l>(rt: &'l R, v: R::Value, at: Settled<'l>) -> &'l mut R::Value;
}
