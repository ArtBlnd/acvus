//! `List<T>`: the dynamic-length sequence, an extension type that crosses
//! the runtime boundary whole.

use acvus_extern::{
    Arr, ExternRegistry, ExternTypeDecl, Interner, LenVar, PolyTy, PolyVars, QualifiedRef, Runtime,
    TyArg, TyVar, TyVarBound, UserDefinedDecl, extern_fn, extern_registry,
};

#[derive(Debug, Clone, PartialEq)]
pub struct List<T>(pub Vec<T>)
where
    T: TyVar;

impl<T> IntoIterator for List<T>
where
    T: TyVar,
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T> TyArg for List<T>
where
    T: TyArg + TyVar,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::UserDefined {
            id: QualifiedRef::root(i.intern("List")),
            type_args: vec![T::poly_ty(i, vars)],
            effect_args: vec![],
            identity_args: vec![],
        }
    }
}

impl<T> ExternTypeDecl for List<T>
where
    T: TyVar,
{
    fn type_decl(i: &Interner) -> UserDefinedDecl {
        UserDefinedDecl {
            qref: QualifiedRef::root(i.intern("List")),
            type_params: vec![TyVarBound::Any],
            effect_params: 0,
            identity_params: 0,
        }
    }
}

/// `List<elem>` as a concrete type, for contexts declared outside a script.
pub fn list_ty(interner: &Interner, elem: acvus_mir::ty::Ty) -> acvus_mir::ty::Ty {
    acvus_mir::ty::Ty::UserDefined {
        id: QualifiedRef::root(interner.intern("List")),
        type_args: vec![elem],
        effect_args: vec![],
        identity_args: vec![],
    }
}

#[extern_fn(effect = pure)]
fn len<T, R>(_: &R, list: List<T>) -> i64
where
    T: TyVar,
    R: Runtime,
{
    list.0.len() as i64
}

#[extern_fn(effect = pure)]
fn reverse<T, R>(_: &R, list: List<T>) -> List<T>
where
    T: TyVar,
    R: Runtime,
{
    let mut items = list.0;
    items.reverse();
    List(items)
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn list<T, N, R>(_: &R, items: Arr<T, N>) -> List<T>
where
    T: TyVar,
    N: LenVar,
    R: Runtime,
{
    List(items.0)
}

pub fn list_registry<R: Runtime>() -> ExternRegistry<R> {
    extern_registry! {
        types: [List<_>],
        fns: [len, reverse, list],
    }
}
