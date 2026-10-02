//! The kinds, each a runtime's associated type. A safe kind is implementable
//! with no unsafe operation, and a misuse a safe runtime detects inside one
//! panics. An unsafe kind, Opaque or Ref, reads a value back on the compiler's
//! premise; the extern crate provides one implementation of each over
//! `Stores`.
//!
//! A value is given to the runtime by `make` and taken back whole by `take`,
//! which ends the runtime's hold of it. A view is lent in continuation-passing
//! style: the runtime calls `lent` with the view, so the view lives no longer
//! than that call and the runtime keeps whatever guard it needs around it.
//!
//! Making a value is safe. Every function that consumes one is `unsafe`,
//! since a `Value` carries neither its type nor its ownership: `take` rests on
//! the value having been made by the kind's `make` and on the caller owning
//! it, `view` on its having been made so and being live, and `view_mut` also
//! on the caller holding it exclusively while `lent` runs. A safe runtime
//! checks these premises and panics; an optimized one may not check them.

#![allow(unsafe_code)]

use crate::runtime::Runtime;

/// The kind of `Int`, `Float`, `Char`, `Bool` and `Unit`, one implementation
/// per scalar type `S`.
pub trait ScalarKind<R, S>
where
    R: Runtime,
{
    fn make(rt: &R, scalar: S) -> R::Value;

    /// # Safety
    ///
    /// `value` was made by `make` at `S`; the caller owns it and uses
    /// neither it nor a copy of it again.
    unsafe fn take(rt: &R, value: R::Value) -> S;

    /// # Safety
    ///
    /// `value` is live and was made by `make` at `S`.
    unsafe fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&S) -> O) -> O;

    /// # Safety
    ///
    /// As `view`, and the caller holds `value` exclusively while `lent` runs.
    unsafe fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut S) -> O) -> O;
}

pub trait StringKind<R>
where
    R: Runtime,
{
    fn make(rt: &R, string: String) -> R::Value;

    /// # Safety
    ///
    /// `value` was made by `make`; the caller owns it and uses neither it nor
    /// a copy of it again.
    unsafe fn take(rt: &R, value: R::Value) -> String;

    /// # Safety
    ///
    /// `value` is live and was made by `make`.
    unsafe fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&String) -> O) -> O;

    /// # Safety
    ///
    /// As `view`, and the caller holds `value` exclusively while `lent` runs.
    unsafe fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut String) -> O) -> O;
}

/// A tuple of `N` parts, each a value the tuple owns.
pub trait TupleKind<R>
where
    R: Runtime,
{
    fn make<const N: usize>(rt: &R, parts: [R::Value; N]) -> R::Value;

    /// # Safety
    ///
    /// `value` was made by `make` with `N` parts; the caller owns it and uses
    /// neither it nor a copy of it again.
    unsafe fn take<const N: usize>(rt: &R, value: R::Value) -> [R::Value; N];

    /// # Safety
    ///
    /// `value` is live and was made by `make` with `N` parts.
    unsafe fn view<const N: usize, O>(rt: &R, value: &R::Value, lent: impl FnOnce([&R::Value; N]) -> O) -> O;

    /// # Safety
    ///
    /// As `view`, and the caller holds `value` exclusively while `lent` runs.
    unsafe fn view_mut<const N: usize, O>(rt: &R, value: &mut R::Value, lent: impl FnOnce([&mut R::Value; N]) -> O) -> O;
}

/// An `Option`, held in Rust's own shape with a payload it owns. A view lends
/// the whole of it, so an exclusive one writes the variant and the payload
/// together, as a store through an exclusive reference to a sum does
/// (rfc_compiler ownership §5).
pub trait OptionKind<R>
where
    R: Runtime,
{
    fn make(rt: &R, option: Option<R::Value>) -> R::Value;

    /// # Safety
    ///
    /// `value` was made by `make`; the caller owns it and uses neither it nor
    /// a copy of it again.
    unsafe fn take(rt: &R, value: R::Value) -> Option<R::Value>;

    /// # Safety
    ///
    /// `value` is live and was made by `make`.
    unsafe fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&Option<R::Value>) -> O) -> O;

    /// # Safety
    ///
    /// As `view`, and the caller holds `value` exclusively while `lent` runs.
    unsafe fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut Option<R::Value>) -> O) -> O;
}

/// A `Result`, as `OptionKind` is an `Option`.
pub trait ResultKind<R>
where
    R: Runtime,
{
    fn make(rt: &R, result: Result<R::Value, R::Value>) -> R::Value;

    /// # Safety
    ///
    /// `value` was made by `make`; the caller owns it and uses neither it nor
    /// a copy of it again.
    unsafe fn take(rt: &R, value: R::Value) -> Result<R::Value, R::Value>;

    /// # Safety
    ///
    /// `value` is live and was made by `make`.
    unsafe fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&Result<R::Value, R::Value>) -> O) -> O;

    /// # Safety
    ///
    /// As `view`, and the caller holds `value` exclusively while `lent` runs.
    unsafe fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut Result<R::Value, R::Value>) -> O) -> O;
}

/// An acvus object, its fields in their names' text order (rfc_compiler
/// types §2). It crosses by value alone.
pub trait ObjectKind<R>
where
    R: Runtime,
{
    fn make<const N: usize>(rt: &R, fields: [R::Value; N]) -> R::Value;

    /// # Safety
    ///
    /// `value` was made by `make` with `N` fields; the caller owns it and
    /// uses neither it nor a copy of it again.
    unsafe fn take<const N: usize>(rt: &R, value: R::Value) -> [R::Value; N];
}

/// An acvus enum: a tag, its variant's place in its names' text order, and
/// one payload, `()` for a variant that carries none. It crosses by value
/// alone. `take` picks the constructor for the tag from `variants`, so the
/// runtime is where a tag outside them is caught.
pub trait EnumKind<R>
where
    R: Runtime,
{
    fn make(rt: &R, tag: usize, payload: R::Value) -> R::Value;

    /// Takes the value back by calling `variants[tag]` with its payload.
    ///
    /// # Safety
    ///
    /// `value` was made by `make`; the caller owns it and uses neither it nor
    /// a copy of it again. Each `variants[t]` builds the variant of tag `t`
    /// from a payload made for that variant.
    unsafe fn take<E>(rt: &R, value: R::Value, variants: &[unsafe fn(&R, R::Value) -> E]) -> E;
}

/// The kind of an extern-defined type: a value of any Rust type `X`, held
/// with its type erased. `X` needs no `'static`, so a value holding a loan is
/// held as it is; it is `Send + Sync`, so a spawn may carry it.
///
/// The type a value is read at is the caller's premise: the type it was made
/// at, or one of that type's layout as the extern note's Type variables admit,
/// with lifetimes the value's loans live through.
pub trait OpaqueKind<R>
where
    R: Runtime,
{
    fn make<X>(rt: &R, value: X) -> R::Value
    where
        X: Send + Sync;

    /// # Safety
    ///
    /// `value` was made by `make` at `X`, as this trait states; the caller
    /// owns it and uses neither it nor a copy of it again.
    unsafe fn take<X>(rt: &R, value: R::Value) -> X;

    /// # Safety
    ///
    /// `value` is live and was made by `make` at `X`, as this trait states.
    unsafe fn view<X, O>(rt: &R, value: &R::Value, lent: impl FnOnce(&X) -> O) -> O;

    /// # Safety
    ///
    /// As `view`, and the caller holds `value` exclusively while `lent` runs.
    unsafe fn view_mut<X, O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut X) -> O) -> O;
}

/// The kind of `Ref`: a reference value, made from a Rust reference into
/// Rust memory and read back as one only within a view.
pub trait RefKind<R>
where
    R: Runtime,
{
    fn make<X>(rt: &R, reference: &X) -> R::Value
    where
        X: ?Sized;

    fn make_mut<X>(rt: &R, reference: &mut X) -> R::Value
    where
        X: ?Sized;

    /// # Safety
    ///
    /// `value` is live and was made by `make` or `make_mut` from a reference
    /// to `X`, or to a type of `X`'s layout, whose referent is live and is
    /// not written while `lent` runs (rfc_compiler ownership §5).
    unsafe fn view<X, O>(rt: &R, value: &R::Value, lent: impl FnOnce(&X) -> O) -> O
    where
        X: ?Sized;

    /// # Safety
    ///
    /// `value` is live and was made by `make_mut` from a reference to `X`, or
    /// to a type of `X`'s layout, whose referent is live and is reached
    /// through nothing else while `lent` runs (rfc_compiler ownership §5).
    unsafe fn view_mut<X, O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut X) -> O) -> O
    where
        X: ?Sized;
}

/// The kind of `Fn` and `Closures`: a function value, whether it runs on an
/// acvus frame or is a Rust function an extern gave.
pub trait FnKind<R>
where
    R: Runtime,
{
    fn make<const N: usize>(rt: &R, body: impl Fn(&R, [R::Value; N]) -> R::Value + 'static) -> R::Value;

    /// Calls `function` with `args`, which the call takes; the result is the
    /// caller's.
    ///
    /// # Safety
    ///
    /// `function` is live and was made by `make` with `N` arguments, and each
    /// argument is a live value of the type the function takes at its
    /// position, owned by the call.
    unsafe fn call<const N: usize>(rt: &R, function: &R::Value, args: [R::Value; N]) -> R::Value;
}
