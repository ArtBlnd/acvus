//! `List<T>`: the dynamic-length sequence, a UserDefined type over `Vec<Value>`.

use std::sync::Arc;

use acvus_interpreter::{
    ExternFnBuilder, ExternRegistry, ExternValue, FromValue, RuntimeError, Value,
    error::ValueKind,
};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::{
    CastRule, Effect, ParamTerm, Poly, PolyBuilder, PolyTy, Ty, TyTerm, TypeRegistry,
    UserDefinedDecl,
};
use acvus_utils::Interner;

pub fn list_qref(interner: &Interner) -> QualifiedRef {
    QualifiedRef::root(interner.intern("List"))
}

pub fn list_ty(interner: &Interner, elem: Ty) -> Ty {
    Ty::UserDefined {
        id: list_qref(interner),
        type_args: vec![elem],
        effect_args: vec![],
    }
}

pub fn list_poly_ty(interner: &Interner, elem: PolyTy) -> PolyTy {
    PolyTy::UserDefined {
        id: list_qref(interner),
        type_args: vec![elem],
        effect_args: vec![],
    }
}

pub fn list_value(interner: &Interner, items: Vec<Value>) -> Value {
    Value::extern_value(ExternValue::new(list_qref(interner), items))
}

pub struct List(pub Vec<Value>);

impl FromValue for List {
    fn from_value(value: Value) -> Result<Self, RuntimeError> {
        match value {
            Value::Extern(o) => match o.into_owned::<Vec<Value>>() {
                Ok(items) => Ok(List(items)),
                Err(o) => Err(RuntimeError::internal(format!(
                    "List was shared or is another extern type: {o:?}"
                ))),
            },
            other => Err(RuntimeError::unexpected_type(
                "FromValue<List>",
                &[ValueKind::Extern],
                other.kind(),
            )),
        }
    }
}

pub fn sequence_items(value: Value) -> Result<Vec<Value>, RuntimeError> {
    match value {
        Value::Array(items) => Ok(Arc::try_unwrap(items).unwrap_or_else(|arc| arc.as_ref().clone())),
        other => List::from_value(other).map(|l| l.0),
    }
}

fn h_len(_: &Interner, (list,): (List,)) -> Result<i64, RuntimeError> {
    Ok(list.0.len() as i64)
}

fn h_reverse(interner: &Interner, (list,): (List,)) -> Result<Value, RuntimeError> {
    let mut items = list.0;
    items.reverse();
    Ok(list_value(interner, items))
}

fn h_array_to_list(interner: &Interner, (items,): (Vec<Value>,)) -> Result<Value, RuntimeError> {
    Ok(list_value(interner, items))
}

fn sig(interner: &Interner, params: &[PolyTy], ret: PolyTy) -> PolyTy {
    TyTerm::Fn {
        params: params
            .iter()
            .enumerate()
            .map(|(i, ty)| ParamTerm::<Poly>::new(interner.intern(&format!("_{i}")), ty.clone()))
            .collect(),
        ret: Box::new(ret),
        captures: vec![],
        effect: Effect::Pure.into(),
    }
}

pub fn list_registry(interner: &Interner, type_registry: &mut TypeRegistry) -> ExternRegistry {
    type_registry.register(UserDefinedDecl {
        qref: list_qref(interner),
        type_params: vec![None],
        effect_params: 0,
    });
    {
        let mut b = PolyBuilder::new();
        let t = b.fresh_ty_var();
        let n = b.fresh_len_var();
        type_registry.register_cast(CastRule {
            from: TyTerm::Array(Box::new(t.clone()), n),
            to: list_poly_ty(interner, t),
            fn_ref: QualifiedRef::root(interner.intern("list")),
        });
    }

    ExternRegistry::new(move |interner| {
        let mut fns = Vec::new();
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            fns.push(
                ExternFnBuilder::new("len", sig(interner, &[list_poly_ty(interner, t)], TyTerm::Int))
                    .handler(h_len),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let list = list_poly_ty(interner, t);
            fns.push(
                ExternFnBuilder::new("reverse", sig(interner, &[list.clone()], list)).handler(h_reverse),
            );
        }
        {
            let mut b = PolyBuilder::new();
            let t = b.fresh_ty_var();
            let n = b.fresh_len_var();
            fns.push(
                ExternFnBuilder::new(
                    "list",
                    sig(
                        interner,
                        &[TyTerm::Array(Box::new(t.clone()), n)],
                        list_poly_ty(interner, t),
                    ),
                )
                .handler(h_array_to_list),
            );
        }
        fns
    })
}
