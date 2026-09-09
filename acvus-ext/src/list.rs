//! `List<T>`: the dynamic-length sequence, an extension type over `Vec<Value>`.

use std::sync::Arc;

use acvus_extern::{
    Arr, ExternRegistry, ExternTypeDecl, ExternTypeName, ExternValue, FromValue, Interner,
    IntoValue, LenVar, PayloadMismatch, PolyTy, PolyVars, QualifiedRef, RuntimeError, TyArg, TyVar,
    UserDefinedDecl, Value, ValueKind, extern_fn, extern_registry,
};

#[derive(Debug, Clone)]
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

impl<T> FromValue for List<T>
where
    T: TyVar,
{
    fn from_value(value: Value, interner: &Interner) -> Result<Self, RuntimeError> {
        let items = list_items(value)?;
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            out.push(T::from_value(item, interner)?);
        }
        Ok(List(out))
    }
}

impl<T> IntoValue for List<T>
where
    T: TyVar,
{
    fn into_value(self, interner: &Interner) -> Value {
        let items: Vec<Value> = self.0.into_iter().map(|v| v.into_value(interner)).collect();
        list_value(items)
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

pub fn list_value(items: Vec<Value>) -> Value {
    Value::extern_value(ExternValue::new(List::<Value>::TYPE_NAME, items))
}

fn list_items(value: Value) -> Result<Vec<Value>, RuntimeError> {
    match value {
        Value::Extern(o) => {
            if o.type_name != List::<Value>::TYPE_NAME {
                return Err(RuntimeError::unexpected_extern(
                    List::<Value>::TYPE_NAME,
                    o.type_name,
                ));
            }
            o.into_cloned::<Vec<Value>>().map_err(|e| match e {
                PayloadMismatch::OtherType => {
                    RuntimeError::internal("List payload is not Vec<Value>")
                }
                PayloadMismatch::Shared(_) => {
                    RuntimeError::internal("into_cloned never reports Shared")
                }
            })
        }
        other => Err(RuntimeError::unexpected_type(
            "FromValue<List>",
            &[ValueKind::Extern],
            other.kind(),
        )),
    }
}

/// The elements of an `Array` or a `List`, as the flatten combinators see them.
pub fn sequence_items(value: Value) -> Result<Vec<Value>, RuntimeError> {
    match value {
        Value::Array(items) => {
            Ok(Arc::try_unwrap(items).unwrap_or_else(|arc| arc.as_ref().clone()))
        }
        other => list_items(other),
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

pub fn list_registry() -> ExternRegistry {
    extern_registry! {
        types: [List<_>],
        fns: [len, reverse, list],
    }
}
