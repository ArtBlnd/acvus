//! `Vec<T>`: the dynamic-length sequence is the language's `Vec<T>` and
//! Rust's `Vec<T>` alike. In a uniform slot it crosses the boundary as the
//! runtime's `Vec<Value>` (RFC-0022): as the whole buffer when the element
//! is a value in place, element by element when the element converts. In a
//! `#` slot it crosses as one box holding the Rust `Vec<T>` itself.

use std::mem::ManuallyDrop;

use acvus_mir::ty::{Ty, TypeArg};

use crate::obj::{Cross, stored_as_container_of};
use crate::registry::ExternTypeDecl;
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, SlotRepr, TyArg, TyVar};
use crate::{Interner, PolyTy, QualifiedRef, TyVarBound, UserDefinedDecl};

/// # Safety
/// `stored_as_container_of::<T, Rt>()`.
unsafe fn into_values<T, Rt>(items: Vec<T>) -> Vec<Rt::Value>
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    let mut items = ManuallyDrop::new(items);
    // SAFETY: `T` is `Value` or `repr(transparent)` over it, so the two
    // element types share size and alignment and the buffer came from
    // `Vec<Value>`'s allocator.
    unsafe { Vec::from_raw_parts(items.as_mut_ptr().cast(), items.len(), items.capacity()) }
}

/// # Safety
/// As `into_values`.
unsafe fn from_values<T, Rt>(items: Vec<Rt::Value>) -> Vec<T>
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    let mut items = ManuallyDrop::new(items);
    // SAFETY: as in `into_values`.
    unsafe { Vec::from_raw_parts(items.as_mut_ptr().cast(), items.len(), items.capacity()) }
}

impl<T, Rt> Cross<Rt> for Vec<T>
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        let items: Vec<Rt::Value> = if stored_as_container_of::<T, Rt>() {
            // SAFETY: the branch condition is `into_values`'s contract.
            unsafe { into_values::<T, Rt>(self) }
        } else {
            self.into_iter().map(|v| v.erase(rt)).collect()
        };
        // SAFETY: `Vec<T>` is stored as `Vec<Value>` (RFC-0022).
        unsafe { rt.erase::<Vec<Rt::Value>>(items) }
    }

    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        // SAFETY: the caller's contract, and `erase` boxes a `Vec<Value>`.
        let items = unsafe { rt.materialize::<Vec<Rt::Value>>(value) };
        if stored_as_container_of::<T, Rt>() {
            // SAFETY: the branch condition is `from_values`'s contract.
            unsafe { from_values::<T, Rt>(items) }
        } else {
            // SAFETY: the caller's contract, forwarded: `erase` erased every
            // element from a `T`.
            items
                .into_iter()
                .map(|v| unsafe { T::materialize(rt, v) })
                .collect()
        }
    }

    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a Self {
        assert!(
            stored_as_container_of::<T, Rt>(),
            "a Vec converted at the boundary has no storage of its own type to read through"
        );
        // SAFETY: the storage is `Vec<Value>`, and `T` is `Value` or
        // `repr(transparent)` over it: the two `Vec`s have one layout.
        unsafe { &*(rt.deref::<Vec<Rt::Value>>(reference) as *const Vec<Rt::Value> as *const Self) }
    }

    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut Self {
        assert!(
            stored_as_container_of::<T, Rt>(),
            "a Vec converted at the boundary has no storage of its own type to read through"
        );
        // SAFETY: as in `deref`, exclusively.
        unsafe {
            &mut *(rt.deref_mut::<Vec<Rt::Value>>(reference) as *mut Vec<Rt::Value> as *mut Self)
        }
    }
}

crate::cross_whole!(CrossSpecialized, Vec<T>, T: Send + Sync + 'static);

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
