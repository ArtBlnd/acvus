//! Array-length positions in ExternFn signatures.
//!
//! `LenArg` names a length variable; `LenVar` is the bound of a generic
//! parameter that is one, and `()` fills it at runtime. `Arr<T, N>` is an
//! array whose length the script decides.

use std::marker::PhantomData;

use acvus_interpreter::{FromValue, IntoValue, RuntimeError, Value, ValueKind};
use acvus_mir::ty::{LenTerm, Poly, PolyTy};
use acvus_utils::Interner;
use std::sync::Arc;

use crate::ty_arg::{PolyVars, TyArg, TyVar};

pub trait LenArg: 'static {
    fn poly_len(vars: &PolyVars) -> LenTerm<Poly>;
}

pub trait LenVar: 'static {}

impl<N: LenArg> LenVar for N {}
impl LenVar for () {}

/// The K-th length variable of a declaration. Uninhabited.
pub enum Len<const K: usize> {}

impl<const K: usize> LenArg for Len<K> {
    fn poly_len(vars: &PolyVars) -> LenTerm<Poly> {
        vars.lens[K]
    }
}

/// `Array<T, N>` with N a length variable. Holds the elements at runtime.
pub struct Arr<T: TyVar, N: LenVar>(pub Vec<T>, PhantomData<N>);

impl<T: TyVar, N: LenVar> Arr<T, N> {
    pub fn new(items: Vec<T>) -> Self {
        Self(items, PhantomData)
    }
}

impl<T: TyArg + TyVar, N: LenArg> TyArg for Arr<T, N> {
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Array(Box::new(T::poly_ty(i, vars)), N::poly_len(vars))
    }
}

impl<T: TyVar, N: LenVar> FromValue for Arr<T, N> {
    fn from_value(value: Value, interner: &Interner) -> Result<Self, RuntimeError> {
        match value {
            Value::Array(items) => {
                let items = Arc::try_unwrap(items).unwrap_or_else(|arc| (*arc).clone());
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(T::from_value(item, interner)?);
                }
                Ok(Self::new(out))
            }
            other => Err(RuntimeError::unexpected_type(
                "FromValue<Arr>",
                &[ValueKind::Array],
                other.kind(),
            )),
        }
    }
}

impl<T: TyVar, N: LenVar> IntoValue for Arr<T, N> {
    fn into_value(self, interner: &Interner) -> Value {
        Value::array(self.0.into_iter().map(|v| v.into_value(interner)).collect())
    }
}
