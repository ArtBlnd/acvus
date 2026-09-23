//! The object a derived struct crosses as (RFC-0039 rule 4, RFC-0048 rule 7, RFC-0050
//! rules 3, 4 and 8).
//!
//! The layout — an `Obj<Owned<Rt>>`: the type's field names in rule 8's order,
//! shared, and one value per field in that order — lives in this file alone,
//! so a change to it changes the derive not at all beyond the order its field
//! table is written in.
//!
//! Both directions are one array of exactly `N` values, so the length of the
//! glue's field table and the length of the object's layout are equal by type
//! rather than by a check per field. `open_in_order` makes the one comparison
//! there is, once per crossing: the object's own width against `N`. It is
//! unreachable — a value crossing into a declared struct's parameter has
//! exactly that struct's fields, since `ObjectTy::meet` refuses an object that
//! lacks one (`Lacks`) or carries one the struct does not name (`Undeclared`),
//! RFC-0042 rule 1.

use acvus_utils::Astr;

use crate::obj::{Obj, ObjectShape};
use crate::owned::Owned;
use crate::runtime::Runtime;

/// The object a derived struct erases to: `names` and `values` in one order,
/// which is rule 8's — `acvus-extern-macro` sorts its field table by the field
/// names as string literals at expansion.
pub fn object_in_order<Rt, const N: usize>(
    rt: &Rt,
    names: [&str; N],
    values: [Owned<Rt>; N],
) -> Rt::Value
where
    Rt: Runtime,
{
    let names: Box<[Astr]> = names.iter().map(|name| rt.symbol(name)).collect();
    // SAFETY: the language's object is `Obj<Owned<Rt>>`, and `open_in_order` is
    // the only reader.
    unsafe {
        rt.erase::<Obj<Owned<Rt>>>(Obj::new(
            ObjectShape::in_order(names),
            Box::new(values) as Box<[Owned<Rt>]>,
        ))
    }
}

/// The fields of a derived struct's object, in the same order
/// `object_in_order` wrote them, for the derive to destructure.
///
/// # Safety
/// `value` is what `object_in_order` wrote.
///
/// # Panics
/// When the object's width is not `N`, which RFC-0042 rule 1 admits no value of.
pub unsafe fn open_in_order<Rt, const N: usize>(rt: &Rt, value: Rt::Value) -> [Owned<Rt>; N]
where
    Rt: Runtime,
{
    // SAFETY: the caller's contract.
    let Obj { values, .. } = unsafe { rt.materialize::<Obj<Owned<Rt>>>(value) };
    let width = values.len();
    let Ok(values) = <Box<[Owned<Rt>]> as TryInto<Box<[Owned<Rt>; N]>>>::try_into(values) else {
        panic!(
            "an object of {width} fields crossed into a struct of {N}: the checker admits only \
             objects of the declared type"
        )
    };
    *values
}

/// The components of a derived struct, written into the destination run the
/// caller lent (RFC-0050 rules 5, 6 and 8). The order is the one
/// `object_in_order` writes and `open_in_order` reads, which is rule 8's, so
/// the destination needs no names.
pub fn fields_into_run<Rt, const N: usize>(values: [Owned<Rt>; N], out: &mut [Rt::Value])
where
    Rt: Runtime,
{
    debug_assert_eq!(
        out.len(),
        N,
        "a struct of {N} fields was lent a destination run of {} registers",
        out.len()
    );
    for (slot, value) in out.iter_mut().zip(values) {
        *slot = value.into_value();
    }
}
