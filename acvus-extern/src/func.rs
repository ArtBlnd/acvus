//! Function-typed parameters: `Fn0<R, E, Rt>`, `Fn1<A, R, E, Rt>`,
//! `Fn2<A, B, R, E, Rt>`, `Fn3<A, B, C, R, E, Rt>`. Each names
//! `Fn(...) -> R with E` in an extern signature — the type solver reads the
//! closure's type from it — and holds the runtime's closure as a plain
//! value. Calling one is the one place a generic body crosses back into the
//! runtime: `f.call(rt, (a, b))` moves runtime values into the callee's
//! parameters and gets one back; a parameter declared `Ref<T>` is passed
//! `rt.reference(&a)`.

use std::future::Future;
use std::marker::PhantomData;

use acvus_mir::ty::{ParamTerm, Poly, PolyTy};
use acvus_utils::Interner;

use crate::effect::{EffectArg, EffectVar};
use crate::obj::Cross;
use crate::owned::Owned;
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

/// A closure value called at the types its declaration names: the
/// arguments cross in and the result crosses out here, once (RFC-0039).
pub trait ClosureFn<Rt: Runtime> {
    type Args: Send;
    type Ret;

    /// Reached where the closure's effect said `Task::Sync`; every
    /// implementation asserts `is_sync`, the run-time answer, against it.
    fn call_now(&self, rt: &Rt, args: Self::Args) -> Self::Ret;
    fn call<'a>(
        &'a self,
        rt: &'a Rt,
        args: Self::Args,
    ) -> impl Future<Output = Self::Ret> + Send + 'a;
}

macro_rules! define_fn_arg {
    ($name:ident; $($A:ident : $slot:literal),*) => {
        pub struct $name<$($A,)* R, E, Rt>(Owned<Rt>, bool, PhantomData<($($A,)* R, E)>)
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
            pub fn new(rt: &Rt, value: Rt::Value) -> Self {
                let sync = rt.call_is_sync(&value);
                Self(Owned::from_value(value), sync, PhantomData)
            }

            pub fn into_value(self) -> Rt::Value {
                self.0.into_value()
            }

            /// Whether a call reaches its result without a future,
            /// asked of the runtime once, here.
            pub fn is_sync(&self) -> bool {
                self.1
            }

            /// The same closure value under the erased types it has at run
            /// time, for code that keeps closures past their declaration.
            pub fn erased(self) -> $name<$(erased!($A),)* Rt::Value, (), Rt> {
                $name(self.0, self.1, PhantomData)
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
                self.0.into_value()
            }

            unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
                Self::new(rt, value)
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

/// The value a call produced, read at the closure's declared return type.
fn returned<R, Rt>(rt: &Rt, out: Rt::Value) -> R
where
    R: Cross<Rt>,
    Rt: Runtime,
{
    // SAFETY: the closure's declared return type is `R`.
    unsafe { R::materialize(rt, out) }
}

impl<R, E, Rt> ClosureFn<Rt> for Fn0<R, E, Rt>
where
    R: Cross<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    type Args = ();
    type Ret = R;

    fn call_now(&self, rt: &Rt, _: ()) -> R {
        debug_assert!(
            self.is_sync(),
            "a closure value whose effect's task is Sync suspends at run time (RFC-0046)"
        );
        returned(rt, rt.call_now(&self.0, &mut [], CallToken::mint()))
    }

    fn call<'a>(&'a self, rt: &'a Rt, _: ()) -> impl Future<Output = R> + Send + 'a {
        async move {
            let out = if self.1 {
                rt.call_now(&self.0, &mut [], CallToken::mint())
            } else {
                rt.call_0(&self.0, CallToken::mint()).await
            };
            returned(rt, out)
        }
    }
}

impl<A, R, E, Rt> Fn1<A, R, E, Rt>
where
    A: TyVar,
    R: TyVar,
    E: EffectVar,
    Rt: Runtime,
{
    /// The closure applied to a value the caller holds at `A`, the result
    /// left as the runtime holds it. As `ClosureFn::call_now`.
    pub fn call_value_now(&self, rt: &Rt, a: Rt::Value) -> Rt::Value {
        debug_assert!(
            self.is_sync(),
            "a closure value whose effect's task is Sync suspends at run time (RFC-0046)"
        );
        rt.call_now(&self.0, &mut [a], CallToken::mint())
    }

    pub fn call_value<'a>(
        &'a self,
        rt: &'a Rt,
        a: Rt::Value,
    ) -> impl Future<Output = Rt::Value> + Send + 'a {
        async move {
            if self.1 {
                return rt.call_now(&self.0, &mut [a], CallToken::mint());
            }
            rt.call_1(&self.0, a, CallToken::mint()).await
        }
    }
}

impl<A, R, E, Rt> ClosureFn<Rt> for Fn1<A, R, E, Rt>
where
    A: Cross<Rt>,
    R: Cross<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    type Args = (A,);
    type Ret = R;

    fn call_now(&self, rt: &Rt, (a,): Self::Args) -> R {
        debug_assert!(
            self.is_sync(),
            "a closure value whose effect's task is Sync suspends at run time (RFC-0046)"
        );
        let a = a.erase(rt);
        returned(rt, rt.call_now(&self.0, &mut [a], CallToken::mint()))
    }

    fn call<'a>(&'a self, rt: &'a Rt, (a,): Self::Args) -> impl Future<Output = R> + Send + 'a {
        async move {
            let a = a.erase(rt);
            let out = if self.1 {
                rt.call_now(&self.0, &mut [a], CallToken::mint())
            } else {
                rt.call_1(&self.0, a, CallToken::mint()).await
            };
            returned(rt, out)
        }
    }
}

impl<A, B, R, E, Rt> ClosureFn<Rt> for Fn2<A, B, R, E, Rt>
where
    A: Cross<Rt>,
    B: Cross<Rt>,
    R: Cross<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    type Args = (A, B);
    type Ret = R;

    fn call_now(&self, rt: &Rt, (a, b): Self::Args) -> R {
        debug_assert!(
            self.is_sync(),
            "a closure value whose effect's task is Sync suspends at run time (RFC-0046)"
        );
        let mut args = [a.erase(rt), b.erase(rt)];
        returned(rt, rt.call_now(&self.0, &mut args, CallToken::mint()))
    }

    fn call<'a>(&'a self, rt: &'a Rt, (a, b): Self::Args) -> impl Future<Output = R> + Send + 'a {
        async move {
            let mut args = [a.erase(rt), b.erase(rt)];
            let out = if self.1 {
                rt.call_now(&self.0, &mut args, CallToken::mint())
            } else {
                rt.call_n(&self.0, &mut args, CallToken::mint()).await
            };
            returned(rt, out)
        }
    }
}

impl<A, B, C, R, E, Rt> ClosureFn<Rt> for Fn3<A, B, C, R, E, Rt>
where
    A: Cross<Rt>,
    B: Cross<Rt>,
    C: Cross<Rt>,
    R: Cross<Rt>,
    E: EffectVar,
    Rt: Runtime,
{
    type Args = (A, B, C);
    type Ret = R;

    fn call_now(&self, rt: &Rt, (a, b, c): Self::Args) -> R {
        debug_assert!(
            self.is_sync(),
            "a closure value whose effect's task is Sync suspends at run time (RFC-0046)"
        );
        let mut args = [a.erase(rt), b.erase(rt), c.erase(rt)];
        returned(rt, rt.call_now(&self.0, &mut args, CallToken::mint()))
    }

    fn call<'a>(
        &'a self,
        rt: &'a Rt,
        (a, b, c): Self::Args,
    ) -> impl Future<Output = R> + Send + 'a {
        async move {
            let mut args = [a.erase(rt), b.erase(rt), c.erase(rt)];
            let out = if self.1 {
                rt.call_now(&self.0, &mut args, CallToken::mint())
            } else {
                rt.call_n(&self.0, &mut args, CallToken::mint()).await
            };
            returned(rt, out)
        }
    }
}
