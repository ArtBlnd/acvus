//! Function-typed parameters: `Fn0<R, E>`, `Fn1<A, R, E>`, `Fn2<A, B, R, E>`.
//! Each names `Fn(...) -> R with E` and holds the closure at runtime.

use std::marker::PhantomData;

use acvus_interpreter::{FnValue, FromValue, IntoValue, RuntimeError, Value, ValueKind};
use acvus_mir::ty::{ParamTerm, Poly, PolyTy};
use acvus_utils::Interner;

use crate::effect::{EffectArg, EffectVar};
use crate::ty_arg::{PolyVars, TyArg, TyVar};

macro_rules! define_fn_arg {
    ($name:ident; $($A:ident : $slot:literal),*) => {
        pub struct $name<$($A: TyVar,)* R: TyVar, E: EffectVar>(
            pub FnValue,
            PhantomData<($($A,)* R, E)>,
        );

        impl<$($A: TyArg + TyVar,)* R: TyArg + TyVar, E: EffectArg> TyArg for $name<$($A,)* R, E> {
            fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
                PolyTy::Fn {
                    params: vec![$(ParamTerm::<Poly>::new(i.intern($slot), $A::poly_ty(i, vars))),*],
                    ret: Box::new(R::poly_ty(i, vars)),
                    captures: vec![],
                    effect: E::poly_effect(vars),
                }
            }
        }

        impl<$($A: TyVar,)* R: TyVar, E: EffectVar> FromValue for $name<$($A,)* R, E> {
            fn from_value(value: Value, _: &Interner) -> Result<Self, RuntimeError> {
                match value {
                    Value::Fn(f) => Ok(Self(*f, PhantomData)),
                    other => Err(RuntimeError::unexpected_type(
                        concat!("FromValue<", stringify!($name), ">"),
                        &[ValueKind::Fn],
                        other.kind(),
                    )),
                }
            }
        }

        impl<$($A: TyVar,)* R: TyVar, E: EffectVar> IntoValue for $name<$($A,)* R, E> {
            fn into_value(self, _: &Interner) -> Value {
                Value::Fn(Box::new(self.0))
            }
        }
    };
}

define_fn_arg!(Fn0;);
define_fn_arg!(Fn1; A: "_0");
define_fn_arg!(Fn2; A: "_0", B: "_1");

impl<R: TyVar, E: EffectVar> Fn0<R, E> {
    pub async fn call(&self, interner: &Interner) -> Result<R, RuntimeError> {
        let out = acvus_interpreter::fn_value_call(&self.0, vec![]).await?;
        R::from_value(out, interner)
    }
}

impl<A: TyVar, R: TyVar, E: EffectVar> Fn1<A, R, E> {
    pub async fn call(&self, interner: &Interner, a: A) -> Result<R, RuntimeError> {
        let out = self.0.call(a.into_value(interner)).await?;
        R::from_value(out, interner)
    }
}

impl<A: TyVar, B: TyVar, R: TyVar, E: EffectVar> Fn2<A, B, R, E> {
    pub async fn call(&self, interner: &Interner, a: A, b: B) -> Result<R, RuntimeError> {
        let out = self.0.call2(a.into_value(interner), b.into_value(interner)).await?;
        R::from_value(out, interner)
    }
}
