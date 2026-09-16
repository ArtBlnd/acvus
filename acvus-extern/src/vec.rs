//! `Vec<T>`: the dynamic-length sequence is the language's `Vec<T>` and
//! Rust's `Vec<T>` alike. It crosses the boundary as the runtime's
//! `Vec<Value>` (RFC-0022), element by element when the element converts.

use acvus_mir::ty::Ty;

use crate::obj::Cross;
use crate::registry::ExternTypeDecl;
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg, TyVar};
use crate::{Interner, PolyTy, QualifiedRef, TyVarBound, UserDefinedDecl};

impl<T, Rt> Cross<Rt> for Vec<T>
where
    T: Cross<Rt>,
    Rt: Runtime,
{
    fn erase(self, rt: &Rt) -> Rt::Value {
        let items: Vec<Rt::Value> = self.into_iter().map(|v| v.erase(rt)).collect();
        // SAFETY: `Vec<T>` is stored as `Vec<Value>` (RFC-0022).
        unsafe { rt.erase::<Vec<Rt::Value>>(items) }
    }

    fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        // SAFETY: as in `erase`.
        let items = unsafe { rt.materialize::<Vec<Rt::Value>>(value) };
        items.into_iter().map(|v| T::materialize(rt, v)).collect()
    }

    unsafe fn deref<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a Self {
        assert!(
            std::any::TypeId::of::<T>() == std::any::TypeId::of::<Rt::Value>(),
            "a Vec converted at the boundary has no storage of its own type to read through"
        );
        // SAFETY: the storage is `Vec<Value>` and `T` is `Value`.
        unsafe { rt.deref::<Self>(reference) }
    }

    unsafe fn deref_mut<'a>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut Self {
        assert!(
            std::any::TypeId::of::<T>() == std::any::TypeId::of::<Rt::Value>(),
            "a Vec converted at the boundary has no storage of its own type to read through"
        );
        // SAFETY: as in `deref`.
        unsafe { rt.deref_mut::<Self>(reference) }
    }
}

impl<T> TyArg for Vec<T>
where
    T: TyArg + TyVar,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::UserDefined {
            id: QualifiedRef::root(i.intern("Vec")),
            type_args: vec![T::poly_ty(i, vars)],
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
        }
    }
}

/// `Vec<elem>` as a concrete type, for contexts declared outside a script.
pub fn vec_ty(interner: &Interner, elem: Ty) -> Ty {
    Ty::UserDefined {
        id: QualifiedRef::root(interner.intern("Vec")),
        type_args: vec![elem],
        effect_args: vec![],
        identity_args: vec![],
    }
}
