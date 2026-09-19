//! The boundary's one crossing (RFC-0039): every type an ExternFn takes or
//! returns implements `Cross`, and the glue calls it and nothing else. A
//! scalar and an extension type are stored as themselves; a derived struct
//! or enum is rebuilt field by field (RFC-0032, RFC-0036); a container
//! crosses each element; a carrier (`Ref`, `Fn1`) is the runtime value it
//! holds. `Obj<V>` and `Variant<V>` are the runtime's own object and
//! variant shapes.

use std::any::{Any, TypeId};
use std::mem::ManuallyDrop;

use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::len::{Arr, LenVar};
use crate::owned::Owned;
use crate::runtime::Runtime;
use crate::ty_arg::{Never, TyVar};

/// An object as the runtime holds it: field name to value.
pub struct Obj<V>(pub FxHashMap<Astr, V>);

/// A variant as the runtime holds it: the tag and its payload, if any.
pub struct Variant<V> {
    pub tag: Astr,
    pub payload: Option<Box<V>>,
}

/// The run of the runtime's values a crossing occupies, as a type, so that a
/// bound can name it and the width is read off the type rather than repeated
/// as a number.
pub trait Form {
    const WIDTH: usize;
}

/// One of the runtime's values.
pub struct One;

/// The two registers the machine keeps a slice in (RFC-0047 amended).
pub struct Pair;

impl Form for One {
    const WIDTH: usize = 1;
}

impl Form for Pair {
    const WIDTH: usize = 2;
}

/// How a type crosses the boundary (RFC-0039): as the run of the runtime's
/// values it occupies. Every type an ExternFn takes or returns implements
/// this. The crossing that is one value is `OneValue`, which every one of
/// them but a slice also implements.
pub trait Cross<Rt>: Sized + Send + Sync + 'static
where
    Rt: Runtime,
{
    type Form: Form;

    /// How many of the runtime's values one `Self` occupies. A declaration's
    /// slot count is the sum of its parameters' widths, and the library adds
    /// them from these constants (RFC-0050 rule 6).
    const WIDTH: usize = <Self::Form as Form>::WIDTH;

    /// `Self` read out of the run of `WIDTH` values it was written into.
    ///
    /// # Safety
    /// `run` is `WIDTH` long and holds what `into_run` wrote.
    unsafe fn from_run(rt: &Rt, run: &[Rt::Value]) -> Self;

    /// `Self` written into a run of `WIDTH` values.
    fn into_run(self, rt: &Rt, out: &mut [Rt::Value]);
}

/// The crossing of a type that is one of the runtime's values: `erase` hands
/// the runtime that value, `materialize` takes it back, and `deref` reads a
/// `Self` through a reference the runtime holds, which only a type stored as
/// itself can do. Every bound that needs a value — a parameter, an object's
/// field, a container's element — says this and not `Cross`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not cross the boundary as one of the runtime's values",
    note = "a slice crosses as the two registers it occupies and is no value of the language: it is a result, never a parameter, a field, or a container's element (RFC-0047 amended)."
)]
pub trait OneValue<Rt>: Cross<Rt>
where
    Rt: Runtime,
{
    /// `Self` is the runtime's value under another name, with its layout:
    /// a container of `Self` is a container of values in place.
    const STORED_AS_VALUE: bool = false;

    fn erase(self, rt: &Rt) -> Rt::Value;

    /// # Safety
    /// `value` was erased from `Self` (by the runtime's `erase::<Self>` or
    /// `Self::erase`).
    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self;

    /// # Safety
    /// `reference` names a live storage of `Self`, exclusively for the
    /// duration when `deref_mut`.
    unsafe fn deref<'a>(_rt: &Rt, _reference: &'a Rt::Value) -> &'a Self {
        panic!("{}", NO_STORAGE)
    }

    /// # Safety
    /// As `deref`.
    #[allow(clippy::mut_from_ref)]
    unsafe fn deref_mut<'a>(_rt: &Rt, _reference: &'a Rt::Value) -> &'a mut Self {
        panic!("{}", NO_STORAGE)
    }
}

/// The `Cross` of a one-value crossing, whose run is that one value: written
/// here so that every such impl is these two calls and no third form of the
/// same sentence.
///
/// # Safety
/// As `Cross::from_run`.
pub unsafe fn one_from_run<T, Rt>(rt: &Rt, run: &[Rt::Value]) -> T
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    // SAFETY: the caller's contract, at one value.
    unsafe { T::materialize(rt, run[0]) }
}

/// The other half of `one_from_run`.
pub fn one_into_run<T, Rt>(value: T, rt: &Rt, out: &mut [Rt::Value])
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    out[0] = value.erase(rt);
}

/// The `Cross` impl of a type whose crossing is one value: its run is that
/// value, so both directions are `OneValue`'s.
#[macro_export]
macro_rules! cross_one_value {
    ($t:ty, at $rt:ty) => {
        impl $crate::Cross<$rt> for $t {
            type Form = $crate::One;

            unsafe fn from_run(rt: &$rt, run: &[<$rt as $crate::Runtime>::Value]) -> Self {
                // SAFETY: the caller's contract, at one value.
                unsafe { $crate::one_from_run(rt, run) }
            }

            fn into_run(self, rt: &$rt, out: &mut [<$rt as $crate::Runtime>::Value]) {
                $crate::one_into_run(self, rt, out)
            }
        }
    };
    ($t:ty $(, $($g:tt)*)?) => {
        impl<$($($g)*,)? __Rt> $crate::Cross<__Rt> for $t
        where
            __Rt: $crate::Runtime,
        {
            type Form = $crate::One;

            unsafe fn from_run(
                rt: &__Rt,
                run: &[<__Rt as $crate::Runtime>::Value],
            ) -> Self {
                // SAFETY: the caller's contract, at one value.
                unsafe { $crate::one_from_run(rt, run) }
            }

            fn into_run(self, rt: &__Rt, out: &mut [<__Rt as $crate::Runtime>::Value]) {
                $crate::one_into_run(self, rt, out)
            }
        }
    };
}

/// The message a converted type gives when read through a reference: it
/// has no storage of its own type (RFC-0032).
const NO_STORAGE: &str =
    "a value converted at the boundary has no storage of its own type to read through";

/// How the type of a `#` slot crosses (both.md, Revision): a family
/// (`Vec<T>`, `Deque<T>`) is one box holding the whole Rust value, a leaf
/// is stored as itself, and `Option`/`Result`/`Arr` forward to their
/// payloads. The glue of a `Monomorphize` member instance calls this for
/// every parameter and return whose type names the member.
///
/// A member of a `Monomorphize` family is one of the runtime's values, so
/// this trait needs no split into a run and a value the way `Cross` does:
/// the run is the value, here and at every impl.
pub trait CrossSpecialized<Rt>: Sized + Send + Sync + 'static
where
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value;

    /// # Safety
    /// `value` was erased from `Self` (by the runtime's `erase::<Self>` or
    /// `Self::erase`).
    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self;

    /// # Safety
    /// As `Cross::from_run`.
    unsafe fn from_run(rt: &Rt, run: &[Rt::Value]) -> Self {
        // SAFETY: the caller's contract, at one value.
        unsafe { Self::materialize(rt, run[0]) }
    }

    /// As `Cross::into_run`.
    fn into_run(self, rt: &Rt, out: &mut [Rt::Value]) {
        out[0] = self.erase(rt);
    }

    /// # Safety
    /// As `Cross::deref`.
    unsafe fn deref<'a>(_rt: &Rt, _reference: &'a Rt::Value) -> &'a Self {
        panic!("{}", NO_STORAGE)
    }

    /// # Safety
    /// As `Cross::deref_mut`.
    #[allow(clippy::mut_from_ref)]
    unsafe fn deref_mut<'a>(_rt: &Rt, _reference: &'a Rt::Value) -> &'a mut Self {
        panic!("{}", NO_STORAGE)
    }
}

/// A type the runtime stores as itself: its `erase` is `rt.erase::<Self>`,
/// so the runtime reads a value erased from it back as a `Self` in place.
pub trait Stored<Rt>: OneValue<Rt>
where
    Rt: Runtime,
{
}

/// A name for the runtime's value with its layout, so a `[Rt::Value]` in
/// storage is read in place as a `[Self]` (`Ref::as_slice`).
///
/// # Safety
/// `Self` is `#[repr(transparent)]` with `Rt::Value` as its one
/// non-zero-sized field.
pub unsafe trait TransparentOver<Rt>: OneValue<Rt>
where
    Rt: Runtime,
{
}

/// The `Value -> Self` step a body takes outside the glue: identity for the
/// runtime's own value, otherwise a downcast checked against
/// `Runtime::type_of`. There is no impl for a bare scalar: a scalar comes
/// out as `Erased<Rt, T>`, whose `from_value` checks the value's record of
/// its type like any other.
///
/// # Panics
/// When the value was not erased from `Self`.
pub trait FromValue<Rt>: Sized
where
    Rt: Runtime,
{
    fn from_value(rt: &Rt, value: Rt::Value) -> Self;
}

pub fn expect_type<T, Rt>(rt: &Rt, value: &Rt::Value)
where
    T: 'static,
    Rt: Runtime,
{
    let expected = TypeId::of::<T>();
    match rt.type_of(value) {
        Some(found) if found == expected => (),
        Some(found) => match rt.type_name_of(value) {
            Some(name) => panic!(
                "expected a value erased from `{}`, found one erased from `{name}`",
                std::any::type_name::<T>()
            ),
            None => panic!(
                "expected a value erased from `{}`, found a payload of {found:?}",
                std::any::type_name::<T>()
            ),
        },
        None => panic!(
            "expected a value erased from `{}`, found a value no Rust type was erased into",
            std::any::type_name::<T>()
        ),
    }
}

pub fn downcast<T, Rt>(rt: &Rt, value: Rt::Value) -> T
where
    T: Send + Sync + 'static,
    Rt: Runtime,
{
    expect_type::<T, Rt>(rt, &value);
    // SAFETY: `expect_type` just read `T`'s `TypeId` off this value, which
    // `Runtime::type_of` reports only for a value erased through
    // `erase::<T>`.
    unsafe { rt.materialize::<T>(value) }
}

/// A `Stored` type that lives in the runtime's value word itself, so
/// `Erased<R, T>` derefs to it with no runtime in hand.
pub trait Inline: Copy + Send + Sync + 'static {}

/// Calls `$m! { Name: type, ... }` with every `Inline` type and a name for
/// it. `Inline` is implemented from this list, and a runtime that tags its
/// value word per type builds the tag from the same list.
#[macro_export]
macro_rules! for_each_inline {
    ($m:ident) => {
        $m! {
            I8: i8, I16: i16, I32: i32, I64: i64,
            U8: u8, U16: u16, U32: u32, U64: u64,
            F64: f64, Char: char, Bool: bool, Unit: ()
        }
    };
}

macro_rules! inline {
    ($($name:ident: $t:ty),*) => { $(
        const _: () = assert!(
            std::mem::size_of::<$t>() <= 8
                && std::mem::align_of::<$t>() <= 8
                && !std::mem::needs_drop::<$t>(),
            "an Inline type fits the runtime's value word"
        );
        impl Inline for $t {}
    )* };
}
crate::for_each_inline!(inline);

/// A type stored as itself: the runtime keeps the Rust value and hands it
/// back untouched (RFC-0022).
#[macro_export]
macro_rules! cross_as_stored {
    ($t:ty $(, $($g:tt)*)?) => {
        impl<$($($g)*,)? __Rt> $crate::Stored<__Rt> for $t
        where
            __Rt: $crate::Runtime,
        {
        }

        impl<$($($g)*,)? __Rt> $crate::Borrowable<__Rt> for $t
        where
            __Rt: $crate::Runtime,
        {
        }

        $crate::cross_one_value!($t $(, $($g)*)?);
        $crate::cross_whole!(OneValue, $t $(, $($g)*)?);
        $crate::cross_whole!(CrossSpecialized, $t $(, $($g)*)?);
    };
}

/// One crossing trait implemented as the whole Rust value in one runtime
/// box.
#[doc(hidden)]
#[macro_export]
macro_rules! cross_whole {
    ($trait:ident, $t:ty $(, $($g:tt)*)?) => {
        impl<$($($g)*,)? __Rt> $crate::$trait<__Rt> for $t
        where
            __Rt: $crate::Runtime,
        {
            fn erase(self, rt: &__Rt) -> <__Rt as $crate::Runtime>::Value {
                // SAFETY: stored as itself (RFC-0022).
                unsafe { rt.erase::<$t>(self) }
            }

            unsafe fn materialize(rt: &__Rt, value: <__Rt as $crate::Runtime>::Value) -> Self {
                // SAFETY: the caller's contract, and `erase` is `rt.erase::<$t>`.
                unsafe { rt.materialize::<$t>(value) }
            }

            unsafe fn deref<'a>(rt: &__Rt, reference: &'a <__Rt as $crate::Runtime>::Value) -> &'a Self {
                // SAFETY: the caller's contract.
                unsafe { rt.deref::<$t>(reference) }
            }

            unsafe fn deref_mut<'a>(rt: &__Rt, reference: &'a <__Rt as $crate::Runtime>::Value) -> &'a mut Self {
                // SAFETY: the caller's contract.
                unsafe { rt.deref_mut::<$t>(reference) }
            }
        }
    };
}

cross_as_stored!(i8);
cross_as_stored!(i16);
cross_as_stored!(i32);
cross_as_stored!(i64);
cross_as_stored!(u8);
cross_as_stored!(u16);
cross_as_stored!(u32);
cross_as_stored!(u64);
cross_as_stored!(f64);
cross_as_stored!(char);
cross_as_stored!(bool);
cross_as_stored!(String);
cross_as_stored!(());

crate::cross_one_value!(Never);

impl<Rt> OneValue<Rt> for Never
where
    Rt: Runtime,
{
    fn erase(self, _: &Rt) -> Rt::Value {
        match self {}
    }

    unsafe fn materialize(_: &Rt, _: Rt::Value) -> Self {
        panic!("a value of type `!` was materialized")
    }
}

impl<Rt> CrossSpecialized<Rt> for Never
where
    Rt: Runtime,
{
    fn erase(self, _: &Rt) -> Rt::Value {
        match self {}
    }

    unsafe fn materialize(_: &Rt, _: Rt::Value) -> Self {
        panic!("a value of type `!` was materialized")
    }
}

/// Whether a container of `T` is stored as a container of `T`: when `T`
/// is the runtime's value or a `repr(transparent)` name for it; any other
/// element was converted on the way in, so the storage holds values, not
/// `T`s.
pub(crate) fn stored_as_container_of<T, Rt>() -> bool
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    T::STORED_AS_VALUE || TypeId::of::<T>() == TypeId::of::<Rt::Value>()
}

/// There is no cast arm beside this downcast: two instantiations of a
/// `repr(Rust)` type, such as `Option<Value>` and `Option<Erased<..>>`,
/// have no layout the language promises to be the same.
pub(crate) fn storage_as<S, T>(stored: &S) -> Option<&T>
where
    S: 'static,
    T: 'static,
{
    (stored as &dyn Any).downcast_ref::<T>()
}

pub(crate) fn storage_as_mut<S, T>(stored: &mut S) -> Option<&mut T>
where
    S: 'static,
    T: 'static,
{
    (stored as &mut dyn Any).downcast_mut::<T>()
}

crate::cross_one_value!(Option<T>, T: OneValue<__Rt>);

impl<T, Rt> OneValue<Rt> for Option<T>
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        match self {
            Some(v) => rt.some(v.erase(rt)),
            None => rt.none(),
        }
    }

    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        if rt.is_none(&value) {
            return None;
        }
        // SAFETY: the caller's contract, forwarded: `erase` put a `T`'s
        // value under the `some`.
        Some(unsafe { T::materialize(rt, rt.unwrap_some(value)) })
    }
}

crate::cross_one_value!(Result<T, E>, T: OneValue<__Rt>, E: OneValue<__Rt>);

impl<T, E, Rt> OneValue<Rt> for Result<T, E>
where
    T: OneValue<Rt>,
    E: OneValue<Rt>,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        let inner: Result<Owned<Rt>, Owned<Rt>> = self
            .map(|v| Owned::from_value(v.erase(rt)))
            .map_err(|e| Owned::from_value(e.erase(rt)));
        // SAFETY: the language's Result is the runtime's
        // `Result<Owned<Rt>, Owned<Rt>>` (RFC-0038, RFC-0048 §7).
        unsafe { rt.erase::<Result<Owned<Rt>, Owned<Rt>>>(inner) }
    }

    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        // SAFETY: the caller's contract, and `erase` boxes a
        // `Result<Owned<Rt>, Owned<Rt>>`.
        let inner = unsafe { rt.materialize::<Result<Owned<Rt>, Owned<Rt>>>(value) };
        // SAFETY: the caller's contract, forwarded: `erase` erased the payload
        // from a `T` or an `E`.
        unsafe {
            inner
                .map(|v| T::materialize(rt, v.into_value()))
                .map_err(|e| E::materialize(rt, e.into_value()))
        }
    }
}

impl<T, Rt> CrossSpecialized<Rt> for Option<T>
where
    T: CrossSpecialized<Rt>,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        match self {
            Some(v) => rt.some(v.erase(rt)),
            None => rt.none(),
        }
    }

    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        if rt.is_none(&value) {
            return None;
        }
        // SAFETY: the caller's contract, forwarded: `erase` put a `T`'s
        // value under the `some`.
        Some(unsafe { T::materialize(rt, rt.unwrap_some(value)) })
    }
}

impl<T, E, Rt> CrossSpecialized<Rt> for Result<T, E>
where
    T: CrossSpecialized<Rt>,
    E: CrossSpecialized<Rt>,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        let inner: Result<Owned<Rt>, Owned<Rt>> = self
            .map(|v| Owned::from_value(v.erase(rt)))
            .map_err(|e| Owned::from_value(e.erase(rt)));
        // SAFETY: the language's Result is the runtime's
        // `Result<Owned<Rt>, Owned<Rt>>` (RFC-0038, RFC-0048 §7).
        unsafe { rt.erase::<Result<Owned<Rt>, Owned<Rt>>>(inner) }
    }

    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        // SAFETY: the caller's contract, and `erase` boxes a
        // `Result<Owned<Rt>, Owned<Rt>>`.
        let inner = unsafe { rt.materialize::<Result<Owned<Rt>, Owned<Rt>>>(value) };
        // SAFETY: the caller's contract, forwarded: `erase` erased the payload
        // from a `T` or an `E`.
        unsafe {
            inner
                .map(|v| T::materialize(rt, v.into_value()))
                .map_err(|e| E::materialize(rt, e.into_value()))
        }
    }
}

impl<T, N, Rt> CrossSpecialized<Rt> for Arr<T, N>
where
    T: CrossSpecialized<Rt>,
    N: LenVar,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        let items: Vec<Owned<Rt>> = self
            .0
            .into_iter()
            .map(|v| Owned::from_value(v.erase(rt)))
            .collect();
        // SAFETY: the language's array is `Arr<Owned<Rt>, ()>` (RFC-0022,
        // RFC-0048 §7).
        unsafe { rt.erase::<Arr<Owned<Rt>, ()>>(Arr::new(items)) }
    }

    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        // SAFETY: the caller's contract, and `erase` boxes an
        // `Arr<Owned<Rt>, ()>`.
        let items = unsafe { rt.materialize::<Arr<Owned<Rt>, ()>>(value) };
        // SAFETY: the caller's contract, forwarded: `erase` erased every
        // element from a `T`.
        Arr::new(
            items
                .0
                .into_iter()
                .map(|v| unsafe { T::materialize(rt, v.into_value()) })
                .collect(),
        )
    }
}

crate::cross_one_value!(Arr<T, N>, T: OneValue<__Rt>, N: LenVar);

impl<T, N, Rt> OneValue<Rt> for Arr<T, N>
where
    T: OneValue<Rt>,
    N: LenVar,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        let items: Vec<Owned<Rt>> = self
            .0
            .into_iter()
            .map(|v| Owned::from_value(v.erase(rt)))
            .collect();
        // SAFETY: the language's array is `Arr<Owned<Rt>, ()>` (RFC-0022,
        // RFC-0048 §7).
        unsafe { rt.erase::<Arr<Owned<Rt>, ()>>(Arr::new(items)) }
    }

    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        // SAFETY: the caller's contract, and `erase` boxes an
        // `Arr<Owned<Rt>, ()>`.
        let items = unsafe { rt.materialize::<Arr<Owned<Rt>, ()>>(value) };
        // SAFETY: the caller's contract, forwarded: `erase` erased every
        // element from a `T`.
        Arr::new(
            items
                .0
                .into_iter()
                .map(|v| unsafe { T::materialize(rt, v.into_value()) })
                .collect(),
        )
    }

    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a Self {
        // SAFETY: the caller's contract, and `erase` boxes an
        // `Arr<Owned<Rt>, ()>`.
        let stored = unsafe { rt.deref::<Arr<Owned<Rt>, ()>>(reference) };
        let Some(items) = storage_as::<_, Vec<T>>(&stored.0) else {
            panic!("{NO_STORAGE}")
        };
        // SAFETY: `Arr<T, N>` is `repr(transparent)` over `Vec<T>`.
        unsafe { &*(items as *const Vec<T> as *const Self) }
    }

    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut Self {
        // SAFETY: the caller's contract, exclusively, and `erase` boxes an
        // `Arr<Owned<Rt>, ()>`.
        let stored = unsafe { rt.deref_mut::<Arr<Owned<Rt>, ()>>(reference) };
        let Some(items) = storage_as_mut::<_, Vec<T>>(&mut stored.0) else {
            panic!("{NO_STORAGE}")
        };
        // SAFETY: `Arr<T, N>` is `repr(transparent)` over `Vec<T>`.
        unsafe { &mut *(items as *mut Vec<T> as *mut Self) }
    }
}

/// The language's array keeps a `Vec` of the runtime's values, which is the
/// storage an `Arr<T, N>` reference reads in place.
impl<T, N, Rt> crate::Borrowable<Rt> for Arr<T, N>
where
    T: OneValue<Rt>,
    N: LenVar,
    Rt: Runtime,
{
}

/// The elements of a checked container box, each taken by its own
/// `FromValue`; the buffer itself is reused when the element is the value.
fn elements_from_values<E, Rt>(rt: &Rt, items: Vec<Owned<Rt>>) -> Vec<E>
where
    E: FromValue<Rt> + 'static,
    Rt: Runtime,
{
    if TypeId::of::<E>() == TypeId::of::<Rt::Value>() {
        let mut items = ManuallyDrop::new(items);
        // SAFETY: `E` is `Rt::Value` and `Owned<Rt>` is `repr(transparent)`
        // over it: one element layout, one allocator.
        return unsafe {
            Vec::from_raw_parts(items.as_mut_ptr().cast(), items.len(), items.capacity())
        };
    }
    items
        .into_iter()
        .map(|item| E::from_value(rt, item.into_value()))
        .collect()
}

impl<E, Rt> FromValue<Rt> for Vec<E>
where
    E: FromValue<Rt> + Send + Sync + 'static,
    Rt: Runtime,
{
    fn from_value(rt: &Rt, value: Rt::Value) -> Self {
        let items = downcast::<Vec<Owned<Rt>>, Rt>(rt, value);
        elements_from_values(rt, items)
    }
}

impl<E, N, Rt> FromValue<Rt> for Arr<E, N>
where
    E: FromValue<Rt> + TyVar,
    N: LenVar,
    Rt: Runtime,
{
    fn from_value(rt: &Rt, value: Rt::Value) -> Self {
        let items = downcast::<Arr<Owned<Rt>, ()>, Rt>(rt, value);
        Arr::new(elements_from_values(rt, items.0))
    }
}

/// The field of a derived object, crossed by its own type; what the derive
/// calls for each field.
pub fn erase_field<T, Rt>(rt: &Rt, value: T) -> Owned<Rt>
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    Owned::from_value(value.erase(rt))
}

/// # Safety
/// The field `name` of the object was erased from a `T`.
///
/// # Panics
/// When the object lacks the field: the checker admits only objects of
/// the declared type.
pub unsafe fn materialize_field<T, Rt>(
    rt: &Rt,
    fields: &mut FxHashMap<Astr, Owned<Rt>>,
    name: &str,
) -> T
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    let value = fields.remove(&rt.symbol(name)).unwrap_or_else(|| {
        panic!(
            "object field `{name}` is missing: the checker admits only objects of the declared type"
        )
    });
    // SAFETY: the caller's contract.
    unsafe { T::materialize(rt, value.into_value()) }
}

/// # Panics
/// When the variant has no payload: the checker admits only variants of
/// the declared enum.
pub fn take_payload<V>(payload: Option<V>, tag: &str) -> V {
    let Some(payload) = payload else {
        panic!(
            "variant `{tag}` has no payload: the checker admits only variants of the declared enum"
        )
    };
    payload
}

/// The payload of a derived variant, crossed by its own type.
///
/// # Safety
/// The payload of variant `tag` was erased from a `T`.
pub unsafe fn materialize_payload<T, Rt>(rt: &Rt, payload: Option<Owned<Rt>>, tag: &str) -> T
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    // SAFETY: the caller's contract.
    unsafe { T::materialize(rt, take_payload(payload, tag).into_value()) }
}
