//! The language's object at the boundary (RFC-0032): `Obj<V>` is the
//! runtime's own object shape, and `Cross` converts a type into and out of
//! the runtime's value field by field.

use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::len::{Arr, LenVar};
use crate::repr::Crossing;
use crate::runtime::Runtime;
use crate::ty_arg::TyVar;

/// An object as the runtime holds it: field name to value.
pub struct Obj<V>(pub FxHashMap<Astr, V>);

/// A variant as the runtime holds it: the tag and its payload, if any.
pub struct Variant<V> {
    pub tag: Astr,
    pub payload: Option<Box<V>>,
}

/// A type that crosses the boundary by conversion: its runtime value has
/// another shape than the Rust value.
pub trait Cross<Rt>: Sized
where
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value;
    fn materialize(rt: &Rt, value: Rt::Value) -> Self;
}

/// A scalar is its own runtime value, so a container of scalars crosses
/// as the runtime's container of values.
macro_rules! scalar_crosses_as_itself {
    ($($t:ty),*) => {$(
        impl<Rt> Cross<Rt> for $t
        where
            Rt: Runtime,
        {
            fn erase(self, rt: &Rt) -> Rt::Value {
                // SAFETY: a scalar erased as itself (RFC-0022).
                unsafe { rt.erase::<$t>(self) }
            }

            fn materialize(rt: &Rt, value: Rt::Value) -> Self {
                // SAFETY: as in `erase`.
                unsafe { rt.materialize::<$t>(value) }
            }
        }
    )*};
}

scalar_crosses_as_itself!(i8, i16, i32, i64, u8, u16, u32, u64, f64, bool, String, ());

impl<T, Rt> Cross<Rt> for Option<T>
where
    T: Cross<Rt> + Send + Sync + 'static,
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
}

impl<T, E, Rt> Cross<Rt> for Result<T, E>
where
    T: Cross<Rt> + Send + Sync + 'static,
    E: Cross<Rt> + Send + Sync + 'static,
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
    T: Cross<Rt> + TyVar,
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
}

/// The field of a derived object, crossed by its own type; what the derive
/// calls for each field.
pub fn erase_field<T, Rt>(rt: &Rt, value: T) -> Rt::Value
where
    T: Send + Sync + 'static,
    Rt: Runtime,
{
    #[allow(unused_imports)]
    use crate::repr::{AsCross as _, AsIs as _};
    // SAFETY: the field's declared type is `T`'s (RFC-0023).
    unsafe { (&Crossing::<T, Rt>::new()).erase(rt, value) }
}

/// # Panics
/// When the object lacks the field: the checker admits only objects of
/// the declared type.
pub fn materialize_field<T, Rt>(rt: &Rt, fields: &mut FxHashMap<Astr, Rt::Value>, name: &str) -> T
where
    T: Send + Sync + 'static,
    Rt: Runtime,
{
    #[allow(unused_imports)]
    use crate::repr::{AsCross as _, AsIs as _};
    let value = fields.remove(&rt.symbol(name)).unwrap_or_else(|| {
        panic!(
            "object field `{name}` is missing: the checker admits only objects of the declared type"
        )
    });
    // SAFETY: as in `erase_field`.
    unsafe { (&Crossing::<T, Rt>::new()).materialize(rt, value) }
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
    T: Send + Sync + 'static,
    Rt: Runtime,
{
    #[allow(unused_imports)]
    use crate::repr::{AsCross as _, AsIs as _};
    // SAFETY: as in `erase_field`.
    unsafe { (&Crossing::<T, Rt>::new()).materialize(rt, take_payload(payload, tag)) }
}
