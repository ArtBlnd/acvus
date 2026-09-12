//! `Deque<T>`: a sequence whose only mutating operation is `append`. The
//! append-only restriction is deliberate — no insert, no pop — because it is
//! what later lets a context persist the value as a growing log instead of a
//! whole re-dump. The first concrete instance of the context model.

use acvus_extern::{
    ExternError, ExternRegistry, ExternTypeDecl, ExternTypeName, ExternValue, FromValue, Interner,
    IntoValue, PayloadMismatch, PolyTy, PolyVars, QualifiedRef, Runtime, TyArg, TyVar, TyVarBound,
    UserDefinedDecl, extern_fn, extern_registry,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Deque<T>(pub Vec<T>)
where
    T: TyVar;

impl<T> Deque<T>
where
    T: TyVar,
{
    pub const TYPE_NAME: ExternTypeName = ExternTypeName {
        ns: None,
        name: "Deque",
    };
}

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

impl<R, T> FromValue<R> for Deque<T>
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
                ExternError::internal("Deque payload is not the runtime's values")
            }
            PayloadMismatch::Shared(_) => ExternError::internal("into_cloned never reports Shared"),
        })?;
        Ok(Deque(T::from_value_seq(items, interner)?))
    }
}

impl<R, T> IntoValue<R> for Deque<T>
where
    R: Runtime,
    T: TyVar + IntoValue<R>,
{
    fn into_value(self, interner: &Interner) -> R::Value {
        R::extern_value(ExternValue::new(
            Self::TYPE_NAME,
            T::into_value_seq(self.0, interner),
        ))
    }
}

#[extern_fn(effect = pure)]
fn deque<T>(_: &Interner) -> Deque<T>
where
    T: TyVar,
{
    Deque(Vec::new())
}

#[extern_fn(effect = pure)]
fn append<T>(_: &Interner, d: &mut Deque<T>, item: T)
where
    T: TyVar,
{
    d.0.push(item);
}

#[extern_fn(effect = pure)]
fn deque_len<T>(_: &Interner, d: Deque<T>) -> i64
where
    T: TyVar,
{
    d.0.len() as i64
}

#[extern_fn(effect = pure)]
fn deque_get<T>(_: &Interner, d: Deque<T>, index: i64) -> Result<T, ExternError>
where
    T: TyVar,
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
