//! The boundary's one crossing (RFC-0039): every type an ExternFn takes or
//! returns implements `Cross`, and the glue calls it and nothing else. A
//! scalar and an extension type are stored as themselves; a derived struct
//! or enum is rebuilt field by field (RFC-0032, RFC-0036); a container
//! crosses each element; a carrier (`Ref`, `Fn1`) is the runtime value it
//! holds. `Obj<V>` and `Variant<V>` are the runtime's own object and
//! variant shapes.

use std::any::TypeId;

use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::len::{Arr, LenVar};
use crate::runtime::Runtime;
use crate::ty_arg::Never;

/// An object as the runtime holds it: field name to value.
pub struct Obj<V>(pub FxHashMap<Astr, V>);

/// A variant as the runtime holds it: the tag and its payload, if any.
pub struct Variant<V> {
    pub tag: Astr,
    pub payload: Option<Box<V>>,
}

/// How a type crosses the boundary (RFC-0039). `erase` hands the runtime
/// a value; `materialize` takes one back; `deref` reads a `Self` through a
/// reference the runtime holds, which only a type stored as itself can do.
pub trait Cross<Rt>: Sized + Send + Sync + 'static
where
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value;
    fn materialize(rt: &Rt, value: Rt::Value) -> Self;

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

/// The message a converted type gives when read through a reference: it
/// has no storage of its own type (RFC-0032).
const NO_STORAGE: &str =
    "a value converted at the boundary has no storage of its own type to read through";

/// A type stored as itself: the runtime keeps the Rust value and hands it
/// back untouched (RFC-0022).
#[macro_export]
macro_rules! cross_as_stored {
    ($t:ty $(, $($g:tt)*)?) => {
        impl<$($($g)*,)? __Rt> $crate::Cross<__Rt> for $t
        where
            __Rt: $crate::Runtime,
        {
            fn erase(self, rt: &__Rt) -> <__Rt as $crate::Runtime>::Value {
                // SAFETY: stored as itself (RFC-0022).
                unsafe { rt.erase::<$t>(self) }
            }

            fn materialize(rt: &__Rt, value: <__Rt as $crate::Runtime>::Value) -> Self {
                // SAFETY: as in `erase`.
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
cross_as_stored!(bool);
cross_as_stored!(String);
cross_as_stored!(());

impl<Rt> Cross<Rt> for Never
where
    Rt: Runtime,
{
    fn erase(self, _: &Rt) -> Rt::Value {
        match self {}
    }

    fn materialize(_: &Rt, _: Rt::Value) -> Self {
        panic!("a value of type `!` was materialized")
    }
}

/// Whether a container of `T` is stored as a container of `T`: only when
/// `T` is the runtime's value; any other element was converted on the way
/// in, so the storage holds values, not `T`s.
fn stored_as_container_of<T, Rt>() -> bool
where
    T: 'static,
    Rt: Runtime,
{
    TypeId::of::<T>() == TypeId::of::<Rt::Value>()
}

impl<T, Rt> Cross<Rt> for Option<T>
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        let inner: Option<Rt::Value> = self.map(|v| v.erase(rt));
        // SAFETY: the language's Option is the runtime's `Option<Value>`
        // (RFC-0022).
        unsafe { rt.erase::<Option<Rt::Value>>(inner) }
    }

    fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        // SAFETY: as in `erase`.
        let inner = unsafe { rt.materialize::<Option<Rt::Value>>(value) };
        inner.map(|v| T::materialize(rt, v))
    }

    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a Self {
        assert!(stored_as_container_of::<T, Rt>(), "{NO_STORAGE}");
        // SAFETY: the storage is `Option<Value>` and `T` is `Value`.
        unsafe { rt.deref::<Self>(reference) }
    }

    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut Self {
        assert!(stored_as_container_of::<T, Rt>(), "{NO_STORAGE}");
        // SAFETY: as in `deref`.
        unsafe { rt.deref_mut::<Self>(reference) }
    }
}

impl<T, E, Rt> Cross<Rt> for Result<T, E>
where
    T: Cross<Rt>,
    E: Cross<Rt>,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        let inner: Result<Rt::Value, Rt::Value> =
            self.map(|v| v.erase(rt)).map_err(|e| e.erase(rt));
        // SAFETY: the language's Result is the runtime's `Result<Value, Value>`
        // (RFC-0038).
        unsafe { rt.erase::<Result<Rt::Value, Rt::Value>>(inner) }
    }

    fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        // SAFETY: as in `erase`.
        let inner = unsafe { rt.materialize::<Result<Rt::Value, Rt::Value>>(value) };
        inner
            .map(|v| T::materialize(rt, v))
            .map_err(|e| E::materialize(rt, e))
    }
}

impl<T, N, Rt> Cross<Rt> for Arr<T, N>
where
    T: Cross<Rt>,
    N: LenVar,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        let items: Vec<Rt::Value> = self.0.into_iter().map(|v| v.erase(rt)).collect();
        // SAFETY: the language's array is `Arr<Value, ()>` (RFC-0022).
        unsafe { rt.erase::<Arr<Rt::Value, ()>>(Arr::new(items)) }
    }

    fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        // SAFETY: as in `erase`.
        let items = unsafe { rt.materialize::<Arr<Rt::Value, ()>>(value) };
        Arr::new(items.0.into_iter().map(|v| T::materialize(rt, v)).collect())
    }

    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a Self {
        assert!(stored_as_container_of::<T, Rt>(), "{NO_STORAGE}");
        // SAFETY: the storage is `Arr<Value, ()>`, `T` is `Value`, and `N`
        // is a phantom: the two `Arr`s have one layout.
        unsafe {
            &*(rt.deref::<Arr<Rt::Value, ()>>(reference) as *const Arr<Rt::Value, ()>
                as *const Self)
        }
    }

    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut Self {
        assert!(stored_as_container_of::<T, Rt>(), "{NO_STORAGE}");
        // SAFETY: as in `deref`.
        unsafe {
            &mut *(rt.deref_mut::<Arr<Rt::Value, ()>>(reference) as *mut Arr<Rt::Value, ()>
                as *mut Self)
        }
    }
}

/// The field of a derived object, crossed by its own type; what the derive
/// calls for each field.
pub fn erase_field<T, Rt>(rt: &Rt, value: T) -> Rt::Value
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    value.erase(rt)
}

/// # Panics
/// When the object lacks the field: the checker admits only objects of
/// the declared type.
pub fn materialize_field<T, Rt>(rt: &Rt, fields: &mut FxHashMap<Astr, Rt::Value>, name: &str) -> T
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    let value = fields.remove(&rt.symbol(name)).unwrap_or_else(|| {
        panic!(
            "object field `{name}` is missing: the checker admits only objects of the declared type"
        )
    });
    T::materialize(rt, value)
}

/// # Panics
/// When the variant has no payload: the checker admits only variants of
/// the declared enum.
pub fn take_payload<V>(payload: Option<Box<V>>, tag: &str) -> V {
    let Some(payload) = payload else {
        panic!(
            "variant `{tag}` has no payload: the checker admits only variants of the declared enum"
        )
    };
    *payload
}

/// The payload of a derived variant, crossed by its own type.
pub fn materialize_payload<T, Rt>(rt: &Rt, payload: Option<Box<Rt::Value>>, tag: &str) -> T
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    T::materialize(rt, take_payload(payload, tag))
}
