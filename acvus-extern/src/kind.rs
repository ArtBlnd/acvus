//! The safe kinds. Each is a runtime's associated type, implementable without
//! `unsafe`; a misuse a safe runtime detects inside one panics.
//!
//! A value is given to the runtime by `make` and taken back whole by `take`,
//! which ends the runtime's hold of it. A view is lent in continuation-passing
//! style: the runtime calls `lent` with the view, so the view lives no longer
//! than that call and the runtime keeps whatever guard it needs around it.

use crate::runtime::Runtime;

/// The kind of `Int`, `Float`, `Char`, `Bool` and `Unit`, one implementation
/// per scalar type `S`.
pub trait ScalarKind<R, S>
where
    R: Runtime,
{
    fn make(rt: &R, scalar: S) -> R::Value;
    fn take(rt: &R, value: R::Value) -> S;
    fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&S) -> O) -> O;
    fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut S) -> O) -> O;
}

pub trait StringKind<R>
where
    R: Runtime,
{
    fn make(rt: &R, string: String) -> R::Value;
    fn take(rt: &R, value: R::Value) -> String;
    fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&String) -> O) -> O;
    fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut String) -> O) -> O;
}

/// A tuple of `N` parts, each a value the tuple owns.
pub trait TupleKind<R>
where
    R: Runtime,
{
    fn make<const N: usize>(rt: &R, parts: [R::Value; N]) -> R::Value;
    fn take<const N: usize>(rt: &R, value: R::Value) -> [R::Value; N];
    fn view<const N: usize, O>(rt: &R, value: &R::Value, lent: impl FnOnce([&R::Value; N]) -> O) -> O;
    fn view_mut<const N: usize, O>(rt: &R, value: &mut R::Value, lent: impl FnOnce([&mut R::Value; N]) -> O) -> O;
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
    fn take(rt: &R, value: R::Value) -> Option<R::Value>;
    fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&Option<R::Value>) -> O) -> O;
    fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut Option<R::Value>) -> O) -> O;
}

/// A `Result`, as `OptionKind` is an `Option`.
pub trait ResultKind<R>
where
    R: Runtime,
{
    fn make(rt: &R, result: Result<R::Value, R::Value>) -> R::Value;
    fn take(rt: &R, value: R::Value) -> Result<R::Value, R::Value>;
    fn view<O>(rt: &R, value: &R::Value, lent: impl FnOnce(&Result<R::Value, R::Value>) -> O) -> O;
    fn view_mut<O>(rt: &R, value: &mut R::Value, lent: impl FnOnce(&mut Result<R::Value, R::Value>) -> O) -> O;
}

/// An acvus object, its fields in their names' text order (rfc_compiler
/// types §2). It crosses by value alone.
pub trait ObjectKind<R>
where
    R: Runtime,
{
    fn make<const N: usize>(rt: &R, fields: [R::Value; N]) -> R::Value;
    fn take<const N: usize>(rt: &R, value: R::Value) -> [R::Value; N];
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
    fn take<E>(rt: &R, value: R::Value, variants: &[fn(&R, R::Value) -> E]) -> E;
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
    fn call<const N: usize>(rt: &R, function: &R::Value, args: [R::Value; N]) -> R::Value;
}
