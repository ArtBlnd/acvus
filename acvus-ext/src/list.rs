//! `List<T>`: the dynamic-length sequence, an extension type whose payload
//! is the runtime's own values.

use acvus_extern::{
    Arr, ExternError, ExternRegistry, ExternTypeDecl, ExternTypeName, ExternValue, FromValue,
    Interner, IntoValue, LenVar, PayloadMismatch, PolyTy, PolyVars, QualifiedRef, Runtime, TyArg,
    TyVar, UserDefinedDecl, extern_fn, extern_registry,
};

#[derive(Debug, Clone, PartialEq)]
pub struct List<T>(pub Vec<T>)
where
    T: TyVar;

impl<T> List<T>
where
    T: TyVar,
{
    pub const TYPE_NAME: ExternTypeName = ExternTypeName {
        ns: None,
        name: "List",
    };
}

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
            type_params: vec![None],
            effect_params: 0,
        }
    }
}

impl<R, T> FromValue<R> for List<T>
where
    R: Runtime,
    T: TyVar + FromValue<R>,
{
    fn from_value(value: R::Value, interner: &Interner) -> Result<Self, R::Error> {
        let o = R::into_extern(value)?;
        if o.type_name != Self::TYPE_NAME {
            return Err(ExternError::UnexpectedExtern {
                expected: Self::TYPE_NAME,
                got: o.type_name,
            }
            .into());
        }
        let items = o.into_cloned::<Vec<R::Value>>().map_err(|e| match e {
            PayloadMismatch::OtherType => {
                ExternError::internal("List payload is not the runtime's values")
            }
            PayloadMismatch::Shared(_) => ExternError::internal("into_cloned never reports Shared"),
        })?;
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            out.push(T::from_value(item, interner)?);
        }
        Ok(List(out))
    }
}

impl<R, T> IntoValue<R> for List<T>
where
    R: Runtime,
    T: TyVar + IntoValue<R>,
{
    fn into_value(self, interner: &Interner) -> R::Value {
        let items: Vec<R::Value> = self.0.into_iter().map(|v| v.into_value(interner)).collect();
        R::extern_value(ExternValue::new(Self::TYPE_NAME, items))
    }
}

/// `List<elem>` as a concrete type, for contexts declared outside a script.
pub fn list_ty(interner: &Interner, elem: acvus_mir::ty::Ty) -> acvus_mir::ty::Ty {
    acvus_mir::ty::Ty::UserDefined {
        id: QualifiedRef::root(interner.intern("List")),
        type_args: vec![elem],
        effect_args: vec![],
    }
}

#[extern_fn(effect = pure)]
fn len<T>(_: &Interner, list: List<T>) -> i64
where
    T: TyVar,
{
    list.0.len() as i64
}

#[extern_fn(effect = pure)]
fn reverse<T>(_: &Interner, list: List<T>) -> List<T>
where
    T: TyVar,
{
    let mut items = list.0;
    items.reverse();
    List(items)
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn list<T, N>(_: &Interner, items: Arr<T, N>) -> List<T>
where
    T: TyVar,
    N: LenVar,
{
    List(items.0)
}

pub fn list_registry<R: Runtime>() -> ExternRegistry<R> {
    extern_registry! {
        types: [List<_>],
        fns: [len, reverse, list],
    }
}
