//! `Vec<T>`: the dynamic-length sequence is the language's `Vec<T>` and
//! Rust's `Vec<T>` alike. In a uniform slot it crosses the boundary as the
//! runtime's `Vec<Owned<Rt>>` (RFC-0022): as the whole buffer when the element
//! is a value in place, element by element when the element converts. In a
//! `#` slot it crosses as one box holding the Rust `Vec<T>` itself.

use std::mem::ManuallyDrop;

use acvus_mir::ty::{Ty, TypeArg};

use crate::obj::{OneValue, storage_as, storage_as_mut, stored_as_container_of};
use crate::owned::Owned;
use crate::registry::ExternTypeDecl;
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, SlotRepr, TyArg, TyVar};
use crate::{Interner, PolyTy, QualifiedRef, TyVarBound, UserDefinedDecl};

/// # Safety
/// `stored_as_container_of::<T, Rt>()`.
unsafe fn into_values<T, Rt>(items: Vec<T>) -> Vec<Owned<Rt>>
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    let mut items = ManuallyDrop::new(items);
    // SAFETY: `T` is `Value` or `repr(transparent)` over it, and `Owned<Rt>`
    // is `repr(transparent)` over `Value`, so all three element types share
    // size and alignment and the buffer came from one allocator.
    unsafe { Vec::from_raw_parts(items.as_mut_ptr().cast(), items.len(), items.capacity()) }
}

/// # Safety
/// As `into_values`.
unsafe fn from_values<T, Rt>(items: Vec<Owned<Rt>>) -> Vec<T>
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    let mut items = ManuallyDrop::new(items);
    // SAFETY: as in `into_values`.
    unsafe { Vec::from_raw_parts(items.as_mut_ptr().cast(), items.len(), items.capacity()) }
}

crate::cross_one_value!(Vec<T>, T: crate::OneValue<__Rt>);

impl<T, Rt> OneValue<Rt> for Vec<T>
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        let items: Vec<Owned<Rt>> = if stored_as_container_of::<T, Rt>() {
            // SAFETY: the branch condition is `into_values`'s contract.
            unsafe { into_values::<T, Rt>(self) }
        } else {
            self.into_iter()
                .map(|v| Owned::from_value(v.erase(rt)))
                .collect()
        };
        // SAFETY: `Vec<T>` is stored as `Vec<Owned<Rt>>` (RFC-0022,
        // RFC-0048 §7).
        unsafe { rt.erase::<Vec<Owned<Rt>>>(items) }
    }

    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        // SAFETY: the caller's contract, and `erase` boxes a `Vec<Owned<Rt>>`.
        let items = unsafe { rt.materialize::<Vec<Owned<Rt>>>(value) };
        if stored_as_container_of::<T, Rt>() {
            // SAFETY: the branch condition is `from_values`'s contract.
            unsafe { from_values::<T, Rt>(items) }
        } else {
            // SAFETY: the caller's contract, forwarded: `erase` erased every
            // element from a `T`.
            items
                .into_iter()
                .map(|v| unsafe { T::materialize(rt, v.into_value()) })
                .collect()
        }
    }

    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a Self {
        // SAFETY: the caller's contract, and `erase` boxes a `Vec<Owned<Rt>>`.
        let values = unsafe { rt.deref::<Vec<Owned<Rt>>>(reference) };
        let Some(same) = storage_as::<_, Self>(values) else {
            panic!("{NO_VEC_STORAGE}")
        };
        same
    }

    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut Self {
        // SAFETY: the caller's contract, exclusively, and `erase` boxes a
        // `Vec<Owned<Rt>>`.
        let values = unsafe { rt.deref_mut::<Vec<Owned<Rt>>>(reference) };
        let Some(same) = storage_as_mut::<_, Self>(values) else {
            panic!("{NO_VEC_STORAGE}")
        };
        same
    }
}

/// A container's storage is the `Vec<Owned<Rt>>` the runtime keeps, which a
/// `Vec<T>` reference reads in place when `T` is the runtime's own value.
impl<T, Rt> crate::Borrowable<Rt> for Vec<T>
where
    T: OneValue<Rt>,
    Rt: Runtime,
{
}

/// The message a `Vec<T>` gives when read through a reference and `T` is not
/// the runtime's value: the storage is a `Vec<Owned<Rt>>`, and only a slice
/// of a `repr(transparent)` element is a promised view of it.
const NO_VEC_STORAGE: &str = "a Vec whose element is not the runtime's value has no Vec of its own type to read through; a transparent element is read as a slice by `Ref::as_slice`";

crate::cross_whole!(crate::Specialized, Vec<T>, T: Send + Sync + 'static);

impl<T, Rt> crate::BorrowableSpecialized<Rt> for Vec<T>
where
    T: Send + Sync + 'static,
    Rt: Runtime,
{
}

impl<T> TyArg for Vec<T>
where
    T: TyArg + TyVar,
{
    const SLOT: SlotRepr = T::SLOT;

    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::UserDefined {
            id: QualifiedRef::root(i.intern("Vec")),
            type_args: vec![T::slot(i, vars)],
            effect_args: vec![],
            identity_args: vec![],
        }
    }
}

impl<T> ExternTypeDecl for Vec<T>
where
    T: TyVar,
{
    fn type_decl(i: &Interner) -> UserDefinedDecl {
        UserDefinedDecl {
            qref: QualifiedRef::root(i.intern("Vec")),
            type_params: vec![TyVarBound::Any],
            effect_params: 0,
            identity_params: 0,
            specializable: vec![true],
        }
    }
}

/// `Vec<elem>` as a concrete type, for contexts declared outside a script.
pub fn vec_ty(interner: &Interner, elem: Ty) -> Ty {
    Ty::UserDefined {
        id: QualifiedRef::root(interner.intern("Vec")),
        type_args: vec![TypeArg::uniform(elem)],
        effect_args: vec![],
        identity_args: vec![],
    }
}
