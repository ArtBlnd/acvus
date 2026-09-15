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

scalar_crosses_as_itself!(i64, f64, bool, u8, String, ());

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
