//! The object a derived struct crosses as (RFC-0032, RFC-0048 §7).
//!
//! The layout — today an `Obj<Owned<Rt>>` keyed by the runtime's symbols —
//! lives in this file alone, so a flat layout changes it and not the derive.

use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::obj::{Obj, OneValue, materialize_field};
use crate::owned::Owned;
use crate::runtime::Runtime;

/// An object under construction: the derive adds each field by its declared
/// name, and `erase` closes it into the value the runtime holds.
pub struct Building<Rt>
where
    Rt: Runtime,
{
    fields: FxHashMap<Astr, Owned<Rt>>,
}

impl<Rt> Building<Rt>
where
    Rt: Runtime,
{
    pub fn new() -> Self {
        Self {
            fields: FxHashMap::default(),
        }
    }

    pub fn field<T>(&mut self, rt: &Rt, name: &str, value: T)
    where
        T: OneValue<Rt>,
    {
        self.fields
            .insert(rt.symbol(name), Owned::from_value(value.erase(rt)));
    }

    pub fn erase(self, rt: &Rt) -> Rt::Value {
        // SAFETY: the language's object is `Obj<Owned<Rt>>`, and `Opened::of`
        // is the only reader.
        unsafe { rt.erase::<Obj<Owned<Rt>>>(Obj(self.fields)) }
    }
}

impl<Rt> Default for Building<Rt>
where
    Rt: Runtime,
{
    fn default() -> Self {
        Self::new()
    }
}

/// An object opened for reading: each field is taken by its declared name and
/// removed as it is taken.
pub struct Opened<Rt>
where
    Rt: Runtime,
{
    fields: FxHashMap<Astr, Owned<Rt>>,
}

impl<Rt> Opened<Rt>
where
    Rt: Runtime,
{
    /// # Safety
    /// `value` is what `Building::erase` wrote.
    pub unsafe fn of(rt: &Rt, value: Rt::Value) -> Self {
        // SAFETY: the caller's contract.
        let Obj(fields) = unsafe { rt.materialize::<Obj<Owned<Rt>>>(value) };
        Self { fields }
    }

    /// # Safety
    /// The field `name` was erased from a `T`.
    ///
    /// # Panics
    /// The object lacks the field: the checker admits only objects of the
    /// declared type.
    pub unsafe fn field<T>(&mut self, rt: &Rt, name: &str) -> T
    where
        T: OneValue<Rt>,
    {
        // SAFETY: the caller's contract.
        unsafe { materialize_field::<T, Rt>(rt, &mut self.fields, name) }
    }
}
