//! `Deque<T>`: a sequence whose only mutating operation is `append`. The
//! append-only restriction is deliberate — no insert, no pop — because it is
//! what later lets a context persist the value as a growing log instead of a
//! whole re-dump. The first concrete instance of the context model.

use acvus_extern::{
    ExternError, ExternRegistry, ExternTypeDecl, Interner, PolyTy, PolyVars, QualifiedRef, Runtime,
    TyArg, TyVar, TyVarBound, UserDefinedDecl, extern_fn, extern_registry,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Deque<T>(pub Vec<T>)
where
    T: TyVar;

impl<T> TyArg for Deque<T>
where
    T: TyArg + TyVar,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::UserDefined {
            id: QualifiedRef::root(i.intern("Deque")),
            type_args: vec![T::poly_ty(i, vars)],
            effect_args: vec![],
            identity_args: vec![],
        }
    }
}

impl<T> ExternTypeDecl for Deque<T>
where
    T: TyVar,
{
    fn type_decl(i: &Interner) -> UserDefinedDecl {
        UserDefinedDecl {
            qref: QualifiedRef::root(i.intern("Deque")),
            type_params: vec![TyVarBound::Any],
            effect_params: 0,
            identity_params: 0,
        }
    }
}

#[extern_fn(effect = pure)]
fn deque<T, R>(_: &R) -> Deque<T>
where
    T: TyVar,
    R: Runtime,
{
    Deque(Vec::new())
}

#[extern_fn(effect = pure)]
fn append<T, R>(_: &R, d: &mut Deque<T>, item: T)
where
    T: TyVar,
    R: Runtime,
{
    d.0.push(item);
}

#[extern_fn(effect = pure)]
fn deque_len<T, R>(_: &R, d: Deque<T>) -> i64
where
    T: TyVar,
    R: Runtime,
{
    d.0.len() as i64
}

#[extern_fn(effect = pure)]
fn deque_get<T, R>(_: &R, d: Deque<T>, index: i64) -> Result<T, ExternError>
where
    T: TyVar,
    R: Runtime,
{
    let len = d.0.len();
    let i = usize::try_from(index)
        .ok()
        .filter(|i| *i < len)
        .ok_or_else(|| ExternError::call("deque_get", format!("index {index} out of {len}")))?;
    Ok(d.0.into_iter().nth(i).expect("index checked against len"))
}

pub fn deque_registry<R: Runtime>() -> ExternRegistry<R> {
    extern_registry! {
        types: [Deque<_>],
        fns: [deque, append, deque_len, deque_get],
    }
}
