//! Function-typed parameters: `Fn0<R, E, Rt>`, `Fn1<A, R, E, Rt>`,
//! `Fn2<A, B, R, E, Rt>`, `Fn3<A, B, C, R, E, Rt>`. Each names
//! `Fn(...) -> R with E` in an extern signature — the type solver reads the
//! closure's type from it — and holds the runtime's closure as a plain
//! value. Calling one is the one place a generic body crosses back into the
//! runtime: `f.call(rt, (a, b))` moves runtime values into the callee's
//! parameters and gets one back; a parameter declared `Ref<T>` is passed
//! `rt.reference(&a)`.

use std::marker::PhantomData;

use acvus_mir::ty::{ParamTerm, Poly, PolyTy};
use acvus_utils::Interner;

use crate::effect::{EffectArg, EffectVar};
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg, TyVar};

/// Proof that a call comes through `Fn0`/`Fn1`/…: only this module mints it,
/// so a handler cannot reach the runtime's `call_*` directly.
pub struct CallToken(());

impl CallToken {
    fn mint() -> Self {
        CallToken(())
    }
}

/// A closure value called with runtime values, arity fixed by the type.
pub trait ClosureFn<Rt: Runtime> {
    type Args;
    fn call<'a>(&'a self, rt: &'a Rt, args: Self::Args) -> Rt::CallFuture<'a>;
}

macro_rules! define_fn_arg {
    ($name:ident; $($A:ident : $slot:literal),*) => {
        pub struct $name<$($A,)* R, E, Rt>(Rt::Value, PhantomData<($($A,)* R, E)>)
        where
            $($A: TyVar,)*
            R: TyVar,
            E: EffectVar,
            Rt: Runtime;

        impl<$($A,)* R, E, Rt> $name<$($A,)* R, E, Rt>
        where
            $($A: TyVar,)*
            R: TyVar,
            E: EffectVar,
            Rt: Runtime,
        {
            pub fn new(value: Rt::Value) -> Self {
                Self(value, PhantomData)
            }

            pub fn into_value(self) -> Rt::Value {
                self.0
            }

            /// The same closure value under the erased types it has at run
            /// time, for code that keeps closures past their declaration.
            pub fn erased(self) -> $name<$(erased!($A),)* Rt::Value, (), Rt> {
                $name(self.0, PhantomData)
            }
        }

        impl<$($A,)* R, E, Rt> crate::Cross<Rt> for $name<$($A,)* R, E, Rt>
        where
            $($A: TyVar,)*
            R: TyVar,
            E: EffectVar,
            Rt: Runtime,
        {
            fn erase(self, _: &Rt) -> Rt::Value {
                self.0
            }

            fn materialize(_: &Rt, value: Rt::Value) -> Self {
                Self::new(value)
            }
        }

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
    };
}

macro_rules! erased {
    ($A:ident) => {
        Rt::Value
    };
}

define_fn_arg!(Fn0;);
define_fn_arg!(Fn1; A: "_0");
define_fn_arg!(Fn2; A: "_0", B: "_1");
define_fn_arg!(Fn3; A: "_0", B: "_1", C: "_2");

impl<R, E, Rt> ClosureFn<Rt> for Fn0<R, E, Rt>
where
    R: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    type Args = ();
    fn call<'a>(&'a self, rt: &'a Rt, _: ()) -> Rt::CallFuture<'a> {
        rt.call_0(&self.0, CallToken::mint())
    }
}

impl<A, R, E, Rt> ClosureFn<Rt> for Fn1<A, R, E, Rt>
where
    A: TyVar,
    R: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    type Args = (Rt::Value,);
    fn call<'a>(&'a self, rt: &'a Rt, (a,): Self::Args) -> Rt::CallFuture<'a> {
        rt.call_1(&self.0, a, CallToken::mint())
    }
}

impl<A, B, R, E, Rt> ClosureFn<Rt> for Fn2<A, B, R, E, Rt>
where
    A: TyVar,
    B: TyVar,
    R: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    type Args = (Rt::Value, Rt::Value);
    fn call<'a>(&'a self, rt: &'a Rt, (a, b): Self::Args) -> Rt::CallFuture<'a> {
        rt.call_n(&self.0, vec![a, b], CallToken::mint())
    }
}

impl<A, B, C, R, E, Rt> ClosureFn<Rt> for Fn3<A, B, C, R, E, Rt>
where
    A: TyVar,
    B: TyVar,
    C: TyVar,
    R: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    type Args = (Rt::Value, Rt::Value, Rt::Value);
    fn call<'a>(&'a self, rt: &'a Rt, (a, b, c): Self::Args) -> Rt::CallFuture<'a> {
        rt.call_n(&self.0, vec![a, b, c], CallToken::mint())
    }
}
