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
        pub struct $name<$($A,)* R, E>(pub FnValue, PhantomData<($($A,)* R, E)>)
        where
            $($A: TyVar,)*
            R: TyVar,
            E: EffectVar;

        impl<$($A,)* R, E> TyArg for $name<$($A,)* R, E>
        where
            $($A: TyArg + TyVar,)*
            R: TyArg + TyVar,
            E: EffectArg,
        {
            fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
                PolyTy::Fn {
                    params: vec![$(ParamTerm::<Poly>::new(i.intern($slot), $A::poly_ty(i, vars))),*],
                    ret: Box::new(R::poly_ty(i, vars)),
                    captures: vec![],
                    effect: E::poly_effect(vars),
                }
            }
        }

        impl<$($A,)* R, E> FromValue for $name<$($A,)* R, E>
        where
            $($A: TyVar,)*
            R: TyVar,
            E: EffectVar,
        {
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

        impl<$($A,)* R, E> IntoValue for $name<$($A,)* R, E>
        where
            $($A: TyVar,)*
            R: TyVar,
            E: EffectVar,
        {
            fn into_value(self, _: &Interner) -> Value {
                Value::Fn(Box::new(self.0))
            }
        }
    };
}

define_fn_arg!(Fn0;);
define_fn_arg!(Fn1; A: "_0");
define_fn_arg!(Fn2; A: "_0", B: "_1");

impl<R, E> Fn0<R, E>
where
    R: TyVar,
    E: EffectVar,
{
    pub async fn call(&self, interner: &Interner) -> Result<R, RuntimeError> {
        let out = acvus_interpreter::fn_value_call(&self.0, vec![]).await?;
        R::from_value(out, interner)
    }
}

impl<A, R, E> Fn1<A, R, E>
where
    A: TyVar,
    R: TyVar,
    E: EffectVar,
{
    pub async fn call(&self, interner: &Interner, a: A) -> Result<R, RuntimeError> {
        let out = self.0.call(a.into_value(interner)).await?;
        R::from_value(out, interner)
    }
}

impl<A, B, R, E> Fn2<A, B, R, E>
where
    A: TyVar,
    B: TyVar,
    R: TyVar,
    E: EffectVar,
{
    pub async fn call(&self, interner: &Interner, a: A, b: B) -> Result<R, RuntimeError> {
        let out = self
            .0
            .call2(a.into_value(interner), b.into_value(interner))
            .await?;
        R::from_value(out, interner)
    }
}
