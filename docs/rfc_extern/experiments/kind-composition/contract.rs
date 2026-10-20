//! The runtime contract as kinds (rfc_extern runtime §2–§4): one associated
//! type per kind, each with operations of its own. A kind that holds other
//! values deals in their `Value`s alone, so a nested type is its layers'
//! kinds composed. A projection goes through a reference value, which the
//! runtime makes from a place (references §1), and is lent to a closure. Making a
//! value is safe; an operation that consumes a value, or reads one as a type,
//! is an `unsafe fn` stating its premises (safety §3).

use std::future::Future;
use std::pin::Pin;

use crate::boundary::{Cross, TypeVar};

pub trait Runtime: Sized + 'static {
    type Value: Copy;

    type Scalars: Scalars<Self>;
    type Strings: Strings<Self>;
    type Tuples: Tuples<Self>;
    type Options: Options<Self>;
    type Results: Results<Self>;
    type Opaques: Opaques<Self>;
    type Objects: Objects<Self>;
    type Enums: Enums<Self>;
    type Functions: Functions<Self>;
    type Views: Views<Self>;

    /// Ends the owner of `value`, and with it every value it owns. A
    /// reference value owns nothing, and its release ends nothing.
    ///
    /// # Safety
    /// `value` is live and its holder owns it; it is released once.
    unsafe fn release(value: Self::Value);

    /// A shared reference value naming the value in `place`. A reference to
    /// a reference is that reference (references §1).
    ///
    /// # Safety
    /// `place` holds a live value or reference value, and is not changed
    /// while the reference made lives.
    unsafe fn value_as_ref(place: &Self::Value) -> Self::Value;

    /// An exclusive reference value naming the value in `place`. A reference
    /// to a reference is that reference (references §1): an exclusive one
    /// reborrowed, and a shared one, a `Ref` or a shared view, as itself,
    /// shared still.
    ///
    /// # Safety
    /// `place` holds a live value or reference value, and is reached through
    /// nothing else while the reference made lives.
    unsafe fn value_as_mut(place: &mut Self::Value) -> Self::Value;
}

/// The scalars the model has, `i64`, `bool` and `()`.
pub trait Scalar: Copy + 'static + seal::Scalar {}

pub(crate) mod seal {
    /// A scalar's bits, which a runtime that holds a scalar in its `Value`
    /// keeps there.
    pub trait Scalar {
        fn to_bits(self) -> u64;
        fn from_bits(bits: u64) -> Self;
    }
}

impl seal::Scalar for i64 {
    fn to_bits(self) -> u64 {
        u64::from_ne_bytes(self.to_ne_bytes())
    }

    fn from_bits(bits: u64) -> Self {
        i64::from_ne_bytes(bits.to_ne_bytes())
    }
}

impl Scalar for i64 {}

impl seal::Scalar for bool {
    fn to_bits(self) -> u64 {
        u64::from(self)
    }

    fn from_bits(bits: u64) -> Self {
        bits != 0
    }
}

impl Scalar for bool {}

impl seal::Scalar for () {
    fn to_bits(self) -> u64 {
        0
    }

    fn from_bits(_: u64) -> Self {}
}

impl Scalar for () {}

pub trait Scalars<R>
where
    R: Runtime,
{
    fn give<S>(scalar: S) -> R::Value
    where
        S: Scalar;

    /// # Safety
    /// `value` was made as `S`, is live, and is owned by the caller.
    unsafe fn take<S>(value: R::Value) -> S
    where
        S: Scalar;

    /// # Safety
    /// `reference` is a live shared reference value to an `S`.
    unsafe fn project<S, O>(reference: R::Value, lent: impl FnOnce(&S) -> O) -> O
    where
        S: Scalar;

    /// # Safety
    /// `reference` is a live exclusive reference value to an `S`, used
    /// nowhere else while `lent` runs.
    unsafe fn project_mut<S, O>(reference: R::Value, lent: impl FnOnce(&mut S) -> O) -> O
    where
        S: Scalar;
}

pub trait Strings<R>
where
    R: Runtime,
{
    fn give(string: String) -> R::Value;

    /// # Safety
    /// As `Scalars::take`, for a `String`.
    unsafe fn take(value: R::Value) -> String;

    /// # Safety
    /// As `Scalars::project`, for a `String`.
    unsafe fn project<O>(reference: R::Value, lent: impl FnOnce(&String) -> O) -> O;

    /// # Safety
    /// As `Scalars::project_mut`, for a `String`.
    unsafe fn project_mut<O>(reference: R::Value, lent: impl FnOnce(&mut String) -> O) -> O;
}

/// A view of a text or of a run of values in memory the view does not own
/// (references §2): given as a Rust `&str`, `&[T]` or `&mut [T]`, and taken
/// back as one at the lifetime the taker names. A run's elements are
/// `Value`s, so `T` is a variable or a `Uniform`. A view is a reference
/// value: it owns nothing, and its release ends nothing.
pub trait Views<R>
where
    R: Runtime,
{
    fn give_text(text: &str) -> R::Value;

    fn give_run<T>(run: &[T]) -> R::Value
    where
        T: TypeVar<R>;

    fn give_run_mut<T>(run: &mut [T]) -> R::Value
    where
        T: TypeVar<R>;

    /// # Safety
    /// `view` was made by `give_text`, and the text it names is live and
    /// unchanged for `'a`.
    unsafe fn take_text<'a>(view: R::Value) -> &'a str;

    /// # Safety
    /// `view` was made by `give_run` or `give_run_mut` from a run of `T`, or
    /// of an instantiation that differs from `T` only in a zero-sized marker
    /// (variables §3), live and unchanged for `'a`.
    unsafe fn take_run<'a, T>(view: R::Value) -> &'a [T]
    where
        T: TypeVar<R>;

    /// # Safety
    /// As `take_run`, the view made by `give_run_mut`, and the run reached
    /// through nothing else for `'a`.
    unsafe fn take_run_mut<'a, T>(view: R::Value) -> &'a mut [T]
    where
        T: TypeVar<R>;
}

/// A tuple: its parts' values, each owned by the tuple. A reference to one
/// is projected to its parts' reference values (runtime §2).
pub trait Tuples<R>
where
    R: Runtime,
{
    /// # Safety
    /// Each part is live and owned by the caller, which gives it up to the
    /// tuple.
    unsafe fn give<const N: usize>(parts: [R::Value; N]) -> R::Value;

    /// # Safety
    /// `value` was made as a tuple of `N` parts, is live, and is owned by the
    /// caller.
    unsafe fn take<const N: usize>(value: R::Value) -> [R::Value; N];

    /// A shared reference value to each part.
    ///
    /// # Safety
    /// `reference` is a live shared reference value to a tuple of `N` parts.
    unsafe fn parts<const N: usize>(reference: R::Value) -> [R::Value; N];

    /// An exclusive reference value to each part.
    ///
    /// # Safety
    /// `reference` is a live exclusive reference value to a tuple of `N`
    /// parts, used nowhere else while the parts' references live.
    unsafe fn parts_mut<const N: usize>(reference: R::Value) -> [R::Value; N];
}

/// An option, projected to a Rust reference to its shape, its payload uniform
/// (runtime §2); how the runtime holds it is its own (values §1).
/// Telling the variants apart is this kind's alone: the payload crosses by
/// its own kind inside these operations.
pub trait Options<R>
where
    R: Runtime,
{
    fn give<P>(option: Option<P>) -> R::Value
    where
        P: Cross<R>;

    /// # Safety
    /// `value` was made as `Option<P>`, is live, and is owned by the caller.
    unsafe fn take<P>(value: R::Value) -> Option<P>
    where
        P: Cross<R>;

    /// # Safety
    /// `reference` is a live shared reference value to an option whose
    /// payload's type is `P`'s at some instantiation (variables §3).
    unsafe fn project<P, O>(reference: R::Value, lent: impl FnOnce(&Option<P>) -> O) -> O
    where
        P: TypeVar<R>;

    /// # Safety
    /// As `project`, the reference exclusive and used nowhere else while `lent`
    /// runs.
    unsafe fn project_mut<P, O>(reference: R::Value, lent: impl FnOnce(&mut Option<P>) -> O) -> O
    where
        P: TypeVar<R>;
}

/// A result, as `Options` an option.
pub trait Results<R>
where
    R: Runtime,
{
    fn give<T, E>(result: Result<T, E>) -> R::Value
    where
        T: Cross<R>,
        E: Cross<R>;

    /// # Safety
    /// `value` was made as `Result<T, E>`, is live, and is owned by the
    /// caller.
    unsafe fn take<T, E>(value: R::Value) -> Result<T, E>
    where
        T: Cross<R>,
        E: Cross<R>;

    /// # Safety
    /// As `Options::project`, for either payload.
    unsafe fn project<T, E, O>(reference: R::Value, lent: impl FnOnce(&Result<T, E>) -> O) -> O
    where
        T: TypeVar<R>,
        E: TypeVar<R>;

    /// # Safety
    /// As `Options::project_mut`, for either payload.
    unsafe fn project_mut<T, E, O>(reference: R::Value, lent: impl FnOnce(&mut Result<T, E>) -> O) -> O
    where
        T: TypeVar<R>,
        E: TypeVar<R>;
}

/// Where a declared object's fields lie: the layout of its Rust type, and
/// each field's name with its offset, every field a `Value`. A field is
/// found by its name, so no order of the fields is assumed.
pub struct ObjectTable {
    pub size: usize,
    pub align: usize,
    pub fields: &'static [(&'static str, usize)],
}

/// One variant of a declared enum: its name, the tag its Rust type stores
/// at the start, and where its fields lie.
pub struct VariantTable {
    pub name: &'static str,
    pub tag: u32,
    pub fields: &'static [(&'static str, usize)],
}

/// Where a declared enum's variants and fields lie: its Rust type is
/// `repr(u32)`, so it stores a `u32` tag at its start and each variant's
/// fields where a `repr(C)` struct of the tag and those fields has them.
pub struct EnumTable {
    pub size: usize,
    pub align: usize,
    pub variants: &'static [VariantTable],
}

/// A declared object's layout, stated by its derive.
///
/// # Safety
/// `TABLE` states `Self`'s size, alignment and every field, each field a
/// `Value` at the offset stated.
pub unsafe trait ObjectLayout {
    const TABLE: &'static ObjectTable;
}

/// A declared enum's layout, stated by its derive.
///
/// # Safety
/// `Self` is `repr(u32)`, and `TABLE` states its size, alignment, and each
/// variant's tag and fields, each field a `Value` at the offset stated.
pub unsafe trait EnumLayout {
    const TABLE: &'static EnumTable;
}

/// A declared object: put in, taken out, and projected to `&` or `&mut`, as
/// an opaque value is, held as its Rust type lays it out, so that the
/// runtime reads its fields by its table.
pub trait Objects<R>
where
    R: Runtime,
{
    fn give<X>(value: X) -> R::Value
    where
        X: ObjectLayout;

    /// # Safety
    /// As `Opaques::take`.
    unsafe fn take<X>(value: R::Value) -> X
    where
        X: ObjectLayout;

    /// # Safety
    /// As `Opaques::project`.
    unsafe fn project<X, O>(reference: R::Value, lent: impl FnOnce(&X) -> O) -> O
    where
        X: ObjectLayout;

    /// # Safety
    /// As `Opaques::project_mut`.
    unsafe fn project_mut<X, O>(reference: R::Value, lent: impl FnOnce(&mut X) -> O) -> O
    where
        X: ObjectLayout;
}

/// A declared enum, as `Objects` holds a declared object.
pub trait Enums<R>
where
    R: Runtime,
{
    fn give<X>(value: X) -> R::Value
    where
        X: EnumLayout;

    /// # Safety
    /// As `Opaques::take`.
    unsafe fn take<X>(value: R::Value) -> X
    where
        X: EnumLayout;

    /// # Safety
    /// As `Opaques::project`.
    unsafe fn project<X, O>(reference: R::Value, lent: impl FnOnce(&X) -> O) -> O
    where
        X: EnumLayout;

    /// # Safety
    /// As `Opaques::project_mut`.
    unsafe fn project_mut<X, O>(reference: R::Value, lent: impl FnOnce(&mut X) -> O) -> O
    where
        X: EnumLayout;
}

/// A function value, which only acvus makes (functions): called through a
/// shared reference, an exclusive one, or by value, which consumes it
/// (functions §2). Its arguments are given as values by the side that holds
/// them as their types (safety §4), and its result is the caller's, taken
/// back by its own kind.
pub trait Functions<R>
where
    R: Runtime,
{
    /// # Safety
    /// `function` is a live shared reference value to a function value whose
    /// parameters are the types `arguments` were made as, each argument live
    /// and given up to the call; the caller takes the result back as the
    /// function's result type.
    unsafe fn call_ref<const N: usize>(function: R::Value, arguments: [R::Value; N]) -> R::Value;

    /// # Safety
    /// As `call_ref`, the reference exclusive and used nowhere else while the
    /// call runs.
    unsafe fn call_mut<const N: usize>(function: R::Value, arguments: [R::Value; N]) -> R::Value;

    /// # Safety
    /// As `call_ref`, `function` the function value itself, owned by the
    /// caller and consumed by the call.
    unsafe fn call_once<const N: usize>(function: R::Value, arguments: [R::Value; N]) -> R::Value;

    /// Calls the function asynchronously in place (functions §3), through a
    /// shared reference: the call runs as the future is polled, and dropping
    /// the future between its polls abandons the call where it suspended.
    /// The future names no lifetime; what keeps the function and the
    /// arguments is the scope that lent them.
    ///
    /// # Safety
    /// As `call_ref`, each argument a reference value a scope lent; the
    /// function and every argument stay live, each unchanged or reached
    /// through nothing else as it was lent, until the future has ended or
    /// is dropped.
    unsafe fn call_async_ref<const N: usize>(function: R::Value, arguments: [R::Value; N]) -> impl Future<Output = R::Value>;

    /// As `call_async_ref`, through an exclusive reference.
    ///
    /// # Safety
    /// As `call_async_ref`, the function's reference exclusive and used
    /// nowhere else until the future has ended or is dropped.
    unsafe fn call_async_mut<const N: usize>(function: R::Value, arguments: [R::Value; N]) -> impl Future<Output = R::Value>;

    /// As `call_async_ref`, by value: the call consumes the function, and the
    /// future owns it until it has ended or is dropped, which releases it.
    ///
    /// # Safety
    /// As `call_async_ref`, `function` the function value itself, owned by
    /// the caller and given up to the call.
    unsafe fn call_async_once<const N: usize>(function: R::Value, arguments: [R::Value; N]) -> impl Future<Output = R::Value>;
}

/// An acvus function value as the runtime holds it: what the compiler would
/// have compiled, here a Rust closure over `Value`s that the runtime alone
/// calls, taking over its arguments and giving its result. It is the
/// runtime's own, as compiled code is, and no position of the boundary.
pub trait Programs: Runtime {
    fn function(program: impl FnMut(&[Self::Value]) -> Self::Value + 'static) -> Self::Value;

    /// A function value called asynchronously: each call gives a future
    /// that runs the body as it is polled.
    fn async_function(program: impl FnMut(&[Self::Value]) -> Pin<Box<dyn Future<Output = Self::Value>>> + 'static) -> Self::Value;
}

/// What the program's own code does with a declared structure inside the
/// runtime: it reads a field by name through the structure's table, and
/// makes a value of the declaration's type where its caller coerces one into
/// it (compiler types §1, §3). These are the runtime's, not the boundary's.
pub trait Structures: Runtime {
    /// A reference value to the named field of the structure `reference`
    /// names, of its current variant for an enum; shared or exclusive as
    /// `reference` is.
    ///
    /// # Safety
    /// `reference` is a live reference value to a declared object or enum
    /// holding a field of that name.
    unsafe fn field(reference: Self::Value, name: &str) -> Self::Value;

    /// The name of the variant the enum `reference` names holds.
    ///
    /// # Safety
    /// `reference` is a live reference value to a declared enum.
    unsafe fn variant(reference: Self::Value) -> &'static str;

    /// An object of the declared type `table` lays out, from its fields'
    /// values, each named once; it takes them over.
    ///
    /// # Safety
    /// Each value is live, owned by the caller, and of its field's type.
    unsafe fn make_object(table: &'static ObjectTable, fields: &[(&str, Self::Value)]) -> Self::Value;

    /// An enum of the declared type `table` lays out, holding the named
    /// variant made from its fields' values; it takes them over.
    ///
    /// # Safety
    /// As `make_object`, for the variant's fields.
    unsafe fn make_enum(table: &'static EnumTable, variant: &str, fields: &[(&str, Self::Value)]) -> Self::Value;
}

/// An opaque value: put in, taken out, and projected to `&` or `&mut`, as a
/// `Box<dyn Any>` is, but held untyped and read by layout, since one value is
/// read under several Rust types, its generic and its concrete instantiations
/// (runtime §2, variables §3).
pub trait Opaques<R>
where
    R: Runtime,
{
    fn give<X>(value: X) -> R::Value;

    /// # Safety
    /// `value` was made as `X` or as an instantiation that differs from it only in a
    /// zero-sized marker, is live, and is owned by the caller.
    unsafe fn take<X>(value: R::Value) -> X;

    /// # Safety
    /// `reference` is a live shared reference value to a value made as `X`
    /// or as such an instantiation.
    unsafe fn project<X, O>(reference: R::Value, lent: impl FnOnce(&X) -> O) -> O;

    /// # Safety
    /// As `project`, the reference exclusive and used nowhere else while `lent`
    /// runs.
    unsafe fn project_mut<X, O>(reference: R::Value, lent: impl FnOnce(&mut X) -> O) -> O;
}
