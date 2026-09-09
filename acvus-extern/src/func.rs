//! Function-typed parameters: `Fn0<R, E, Rt>`, `Fn1<A, R, E, Rt>`,
//! `Fn2<A, B, R, E, Rt>`. Each names `Fn(...) -> R with E` and holds the
//! runtime's closure. Calling one is the one place a generic body crosses
//! back into the runtime.

use std::marker::PhantomData;

use acvus_mir::ty::{ParamTerm, Poly, PolyTy};
use acvus_utils::Interner;

use crate::convert::{FromValue, IntoValue};
use crate::effect::{EffectArg, EffectVar};
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg, TyVar};

macro_rules! define_fn_arg {
    ($name:ident; $($A:ident : $slot:literal),*) => {
        pub struct $name<$($A,)* R, E, Rt>(pub Rt::Closure, PhantomData<($($A,)* R, E)>)
        where
            $($A: TyVar,)*
            R: TyVar,
            E: EffectVar,
            Rt: Runtime;

        impl<$($A,)* R, E, Rt> TyArg for $name<$($A,)* R, E, Rt>
        where
            $($A: TyArg + TyVar,)*
            R: TyArg + TyVar,
            E: EffectArg,
            Rt: Runtime,
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

        impl<$($A,)* R, E, Rt> FromValue<Rt> for $name<$($A,)* R, E, Rt>
        where
            $($A: TyVar,)*
            R: TyVar,
            E: EffectVar,
            Rt: Runtime,
        {
            fn from_value(value: Rt::Value, _: &Interner) -> Result<Self, Rt::Error> {
                Ok(Self(Rt::into_closure(value)?, PhantomData))
            }
        }

        impl<$($A,)* R, E, Rt> IntoValue<Rt> for $name<$($A,)* R, E, Rt>
        where
            $($A: TyVar,)*
            R: TyVar,
            E: EffectVar,
            Rt: Runtime,
        {
            fn into_value(self, _: &Interner) -> Rt::Value {
                Rt::closure(self.0)
            }
        }
    };
}

define_fn_arg!(Fn0;);
define_fn_arg!(Fn1; A: "_0");
define_fn_arg!(Fn2; A: "_0", B: "_1");

impl<R, E, Rt> Fn0<R, E, Rt>
where
    R: TyVar + FromValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    pub async fn call(&self, interner: &Interner) -> Result<R, Rt::Error> {
        let out = Rt::call(&self.0, vec![]).await?;
        R::from_value(out, interner)
    }
}

impl<A, R, E, Rt> Fn1<A, R, E, Rt>
where
    A: TyVar + IntoValue<Rt>,
    R: TyVar + FromValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    pub async fn call(&self, interner: &Interner, a: A) -> Result<R, Rt::Error> {
        let out = Rt::call(&self.0, vec![a.into_value(interner)]).await?;
        R::from_value(out, interner)
    }
}

impl<A, B, R, E, Rt> Fn2<A, B, R, E, Rt>
where
    A: TyVar + IntoValue<Rt>,
    B: TyVar + IntoValue<Rt>,
    R: TyVar + FromValue<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    pub async fn call(&self, interner: &Interner, a: A, b: B) -> Result<R, Rt::Error> {
        let out = Rt::call(
            &self.0,
            vec![a.into_value(interner), b.into_value(interner)],
        )
        .await?;
        R::from_value(out, interner)
    }
}
