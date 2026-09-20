//! Function-typed parameters: `Closure<A, R, E, Rt>`, where `A` is the tuple of the
//! closure's parameter types — `()`, `(T,)`, `(T, U)`, … It names
//! `Fn(...) -> R with E` in an extern signature — the type solver reads the
//! closure's type from it — and holds the runtime's closure as a plain value.
//! Calling one is the one place a generic body crosses back into the runtime:
//! `f.call(rt, (a, b))` moves runtime values into the callee's parameters and
//! gets one back; a parameter declared `Ref<T>` is passed `rt.reference(&a)`.

use std::future::Future;
use std::marker::PhantomData;

use acvus_mir::ty::{ParamTerm, Poly, PolyTy};
use acvus_utils::Interner;

use crate::obj::{Cross, OneValue};
use crate::owned::Owned;
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg};
use crate::ty_arg::{Term, Var, kind};

/// Proof that a call comes through `Closure`: only this module mints it, so a
/// handler cannot reach the runtime's `call_*` directly.
pub struct CallToken(());

impl CallToken {
    fn mint() -> Self {
        CallToken(())
    }
}

/// A closure's parameter tuple.
pub trait Args: Send + Sync + 'static {
    /// The same tuple with every member at the runtime's own owned value,
    /// which is what a closure kept past its declaration is called at.
    type Erased<Rt>: Args
    where
        Rt: Runtime;
}

/// The same tuple, as a call needs it.
pub trait CallArgs<Rt>: Args + crate::IntoRun<Rt>
where
    Rt: Runtime,
{
    /// The awaited call: the runtime entry that takes this many arguments.
    fn awaited<'a>(
        self,
        rt: &'a Rt,
        f: &'a Rt::Value,
        token: CallToken,
    ) -> impl Future<Output = Rt::Value> + Send + 'a;
}

/// A closure's parameter tuple, as the declared acvus type needs it.
pub trait ArgTypes: Send + Sync + 'static {
    fn params(i: &Interner, vars: &PolyVars) -> Vec<ParamTerm<Poly>>;
}

/// A closure value called at the types its declaration names: the arguments
/// cross in and the result crosses out here, once (RFC-0039).
pub trait ClosureFn<Rt: Runtime> {
    type Args: Send;
    type Ret;

    /// Reached where the closure's effect said `Task::Sync`; every
    /// implementation asserts `is_sync`, the run-time answer, against it.
    fn call_now(&self, rt: &Rt, frame: &mut Rt::Frame<'_>, args: Self::Args) -> Self::Ret;
    fn call<'a>(
        &'a self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
        args: Self::Args,
    ) -> impl Future<Output = Self::Ret> + Send + 'a;
}

pub struct Closure<A, R, E, Rt>(Owned<Rt>, bool, PhantomData<(A, R, E)>)
where
    A: Send + Sync + 'static,
    R: Send + Sync + 'static,
    E: Var<kind::Effect>,
    Rt: Runtime;

impl<A, R, E, Rt> Closure<A, R, E, Rt>
where
    A: Send + Sync + 'static,
    R: Send + Sync + 'static,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    pub fn new(rt: &Rt, value: Rt::Value) -> Self {
        let sync = rt.call_is_sync(&value);
        Self(Owned::from_value(value), sync, PhantomData)
    }

    pub fn into_value(self) -> Rt::Value {
        self.0.into_value()
    }

    /// Whether a call reaches its result without a future, asked of the
    /// runtime once, here.
    pub fn is_sync(&self) -> bool {
        self.1
    }
}

impl<A, R, E, Rt> Closure<A, R, E, Rt>
where
    A: Args,
    R: Send + Sync + 'static,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    /// The same closure value under the erased types it has at run time, for
    /// code that keeps closures past their declaration. The effect is kept:
    /// it is a fact of the declaration, not of the argument types.
    pub fn erased(self) -> Closure<A::Erased<Rt>, Owned<Rt>, E, Rt> {
        Closure(self.0, self.1, PhantomData)
    }
}

crate::cross_one_value!(
    Closure<A, R, E, __Rt>,
    A: Send + Sync + 'static, R: Send + Sync + 'static, E: Var<kind::Effect>
);

impl<A, R, E, Rt> crate::OneValue<Rt> for Closure<A, R, E, Rt>
where
    A: Send + Sync + 'static,
    R: Send + Sync + 'static,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    fn erase(self, _: &Rt) -> Rt::Value {
        self.0.into_value()
    }

    unsafe fn materialize(rt: &Rt, value: Rt::Value) -> Self {
        Self::new(rt, value)
    }
}

impl<A, R, E, Rt> Var<kind::Type> for Closure<A, R, E, Rt>
where
    A: ArgTypes,
    R: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
}

impl<A, R, E, Rt> TyArg for Closure<A, R, E, Rt>
where
    A: ArgTypes,
    R: TyArg + Send + Sync + 'static,
    E: Term<kind::Effect> + Var<kind::Effect>,
    Rt: Runtime,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Fn {
            params: A::params(i, vars),
            ret: Box::new(R::poly_ty(i, vars)),
            captures: vec![],
            effect: E::poly(vars),
        }
    }
}

impl<A, R, E, Rt> ClosureFn<Rt> for Closure<A, R, E, Rt>
where
    A: CallArgs<Rt>,
    R: OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    type Args = A;
    type Ret = R;

    fn call_now(&self, rt: &Rt, frame: &mut Rt::Frame<'_>, args: A) -> R {
        debug_assert!(
            self.is_sync(),
            "a closure value whose effect's task is Sync suspends at run time (RFC-0046)"
        );
        returned(rt, rt.call_now(&self.0, frame, args, CallToken::mint()))
    }

    fn call<'a>(
        &'a self,
        rt: &'a Rt,
        frame: &'a mut Rt::Frame<'_>,
        args: A,
    ) -> impl Future<Output = R> + Send + 'a {
        async move {
            if self.1 {
                return returned(rt, rt.call_now(&self.0, frame, args, CallToken::mint()));
            }
            returned(rt, args.awaited(rt, &self.0, CallToken::mint()).await)
        }
    }
}

/// The value a call produced, read at the closure's declared return type.
fn returned<R, Rt>(rt: &Rt, out: Rt::Value) -> R
where
    R: OneValue<Rt>,
    Rt: Runtime,
{
    // SAFETY: the closure's declared return type is `R`.
    unsafe { R::materialize(rt, out) }
}

impl Args for () {
    type Erased<Rt>
        = ()
    where
        Rt: Runtime;
}

impl<Rt> CallArgs<Rt> for ()
where
    Rt: Runtime,
{
    fn awaited<'a>(
        self,
        rt: &'a Rt,
        f: &'a Rt::Value,
        token: CallToken,
    ) -> impl Future<Output = Rt::Value> + Send + 'a {
        rt.call_0(f, token)
    }
}

impl ArgTypes for () {
    fn params(_: &Interner, _: &PolyVars) -> Vec<ParamTerm<Poly>> {
        Vec::new()
    }
}

impl<A0> Args for (A0,)
where
    A0: Send + Sync + 'static,
{
    type Erased<Rt>
        = (Owned<Rt>,)
    where
        Rt: Runtime;
}

/// The one argument the runtime has an entry of its own for.
impl<A0, Rt> CallArgs<Rt> for (A0,)
where
    A0: OneValue<Rt> + Cross<Rt>,
    Rt: Runtime,
{
    fn awaited<'a>(
        self,
        rt: &'a Rt,
        f: &'a Rt::Value,
        token: CallToken,
    ) -> impl Future<Output = Rt::Value> + Send + 'a {
        let a = self.0.erase(rt);
        rt.call_1(f, a, token)
    }
}

/// Every wider tuple: the arguments are erased into a run and handed to
/// `call_n`. One more line is one more arity.
macro_rules! args_of {
    ($($A:ident: $at:tt),+) => {
        impl<$($A,)+> Args for ($($A,)+)
        where
            $($A: Send + Sync + 'static,)+
        {
            type Erased<__Rt>
                = ($(erased!($A, __Rt),)+)
            where
                __Rt: Runtime;
        }

        impl<$($A,)+ Rt> CallArgs<Rt> for ($($A,)+)
        where
            $($A: OneValue<Rt> + Cross<Rt>,)+
            Rt: Runtime,
        {
            fn awaited<'a>(
                self,
                rt: &'a Rt,
                f: &'a Rt::Value,
                token: CallToken,
            ) -> impl Future<Output = Rt::Value> + Send + 'a {
                async move {
                    let mut run = [$(self.$at.erase(rt)),+];
                    rt.call_n(f, &mut run, token).await
                }
            }
        }
    };
}

macro_rules! erased {
    ($A:ident, $rt:ident) => {
        Owned<$rt>
    };
}

macro_rules! arg_types_of {
    ($($A:ident: $slot:literal),*) => {
        impl<$($A,)*> ArgTypes for ($($A,)*)
        where
            $($A: TyArg + Var<kind::Type> + Send + Sync + 'static,)*
        {
            fn params(_i: &Interner, _vars: &PolyVars) -> Vec<ParamTerm<Poly>> {
                vec![$(ParamTerm::<Poly>::new(_i.intern($slot), $A::poly_ty(_i, _vars))),*]
            }
        }
    };
}

args_of!(A0: 0, A1: 1);
args_of!(A0: 0, A1: 1, A2: 2);
args_of!(A0: 0, A1: 1, A2: 2, A3: 3);
args_of!(A0: 0, A1: 1, A2: 2, A3: 3, A4: 4);
args_of!(A0: 0, A1: 1, A2: 2, A3: 3, A4: 4, A5: 5);
args_of!(A0: 0, A1: 1, A2: 2, A3: 3, A4: 4, A5: 5, A6: 6);
args_of!(A0: 0, A1: 1, A2: 2, A3: 3, A4: 4, A5: 5, A6: 6, A7: 7);

arg_types_of!(A0: "_0");
arg_types_of!(A0: "_0", A1: "_1");
arg_types_of!(A0: "_0", A1: "_1", A2: "_2");
arg_types_of!(A0: "_0", A1: "_1", A2: "_2", A3: "_3");
arg_types_of!(A0: "_0", A1: "_1", A2: "_2", A3: "_3", A4: "_4");
arg_types_of!(A0: "_0", A1: "_1", A2: "_2", A3: "_3", A4: "_4", A5: "_5");
arg_types_of!(A0: "_0", A1: "_1", A2: "_2", A3: "_3", A4: "_4", A5: "_5", A6: "_6");
arg_types_of!(
    A0: "_0", A1: "_1", A2: "_2", A3: "_3", A4: "_4", A5: "_5", A6: "_6", A7: "_7"
);
