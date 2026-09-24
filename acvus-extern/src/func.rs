//! Function-typed parameters: `Closure<'a, A, R, E, Rt>`, where `A` is the tuple
//! of the closure's parameter types — `()`, `(T,)`, `(T, U)`, … It names
//! `Fn(...) -> R with E` in an extern signature — the type solver reads the
//! closure's type from it — and holds the runtime's closure as a plain value.
//! `'a` is the call it was handed to, and so are the carriers `A` and `R`
//! name, which the glue's `Within` puts at `'a` (RFC-0079 rule 6).
//!
//! A `Closure` is made in one place: the crossing, as `OneValue::materialize`
//! of a value the checker typed at exactly `A`, `R` and `E`. That is the
//! whole ground of those types, so the type has no constructor a handler can
//! call, no way out to the value, and no retyping: a `new` over a word put
//! `A`, `R`, `E` on it unchecked, and `erased` re-spelled a closure at the
//! runtime's own value in every position, which is the same word with the
//! checker's decision taken off it (RFC-0068 rule 1).
//!
//! Calling one is the one operation, and it is at the declared types only:
//! `f.call_now(ctx, (a, b))` takes each parameter as the handler passes it
//! (`Passed::As`) and returns `R`, and the crossing inside is the same one
//! an extern call makes. A parameter declared `Ref<T, M, Rt>` is passed as
//! Rust's `&T` / `&mut T`: Rust's lifetime says the borrow lives for the
//! call, and the crossing makes the reference word for that call and no
//! longer (RFC-0018, `Runtime::reference`). A body that must hold a closure
//! holds this type at the types it was declared with.

use std::future::Future;
use std::marker::PhantomData;

use acvus_mir::ty::{ParamTerm, Poly, PolyTy};
use acvus_utils::Interner;

use crate::crossing::Crossing;
use crate::ctx::Ctx;
use crate::obj::OneValue;
use crate::owned::Owned;
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg};
use crate::ty_arg::{Term, Var, kind};

/// A declared type at a handler's own boundary, for the call `'a` of the
/// storage it is passed out of: a value as itself, a declared `Ref<T, M, Rt>`
/// as the Rust borrow it stands for. What a handler passes to a closure at
/// that position, and what it receives back from a signature's instance
/// there (RFC-0068 rules 4 and 6).
///
/// `'a` is a parameter of the trait, not of an associated type, and `Held`
/// is `&'a Self`, so that `Self: 'a` holds wherever an impl is read at
/// `'a`: an associated type `As<'a>` would need `where Self: 'a`, which
/// `PassedByValue`'s `for<'a>` then asks at every `'a` and so of `'static`.
///
/// # Safety
/// The word `cross` hands back and the word `restore` reads are exactly the
/// value of `Self` at the type the checker settled for it, passed as `As`;
/// it crosses nothing else with the capability, a part only through that
/// part's own crossing; and it keeps no capability past the call.
pub unsafe trait Passed<'a, Rt, Held = &'a Self>: Send + Sync
where
    Rt: Runtime,
{
    type As: Send;

    /// The one value the parameter crosses as, for the call being made.
    fn cross(rt: Crossing<'_, Rt>, passed: Self::As) -> Rt::Value;

    /// The value back at this type, for `'a` of the storage it came out of.
    ///
    /// # Safety
    /// `word` was made by `cross` (or by the crossing) from a value of
    /// this type, and what it names is live and unmoved for `'a`.
    unsafe fn restore(rt: Crossing<'_, Rt>, word: Rt::Value) -> Self::As;
}

/// A closure parameter that is passed as the value itself, and a closure
/// result that is returned as itself: what a handler bounds a type variable
/// with where it calls a closure at that variable.
pub trait PassedByValue<Rt>: for<'a> Passed<'a, Rt, As = Self>
where
    Rt: Runtime,
{
}

impl<T, Rt> PassedByValue<Rt> for T
where
    T: for<'a> Passed<'a, Rt, As = T>,
    Rt: Runtime,
{
}

/// A closure's parameter tuple.
pub trait Args: Send + Sync {}

pub trait CanonicalArgs: Send + Sync {
    type Canon: CanonicalArgs<Canon = Self::Canon> + 'static;
}

/// The same tuple, as a call needs it.
///
/// # Safety
/// The run `cross_into` writes, and the run `awaited` calls with, are
/// exactly the tuple's members at the types the checker settled for them,
/// each crossed by its own `Passed`; it crosses nothing else with the
/// capability; and it keeps no capability past the call.
pub unsafe trait CallArgs<Rt>: Args
where
    Rt: Runtime,
{
    /// The tuple as the handler passes it.
    type Passed<'a>: Send
    where
        Self: 'a;

    const WIDTH: usize;

    /// The run the call is made with.
    fn cross_into(rt: Crossing<'_, Rt>, passed: Self::Passed<'_>, out: &mut [Rt::Value]);

    /// The awaited call: the runtime entry that takes this many arguments.
    ///
    /// # Safety
    /// As `Runtime::call_now`'s: `f` is one of the runtime's closures, and
    /// this tuple is the argument list its declaration names.
    unsafe fn awaited<'a>(
        passed: Self::Passed<'a>,
        rt: Crossing<'a, Rt>,
        f: &'a Rt::Value,
    ) -> impl Future<Output = Rt::Value> + Send + 'a;
}

/// A tuple of passed forms on its way into the runtime's run.
struct PassedRun<'p, A, Rt>(A::Passed<'p>)
where
    A: CallArgs<Rt> + 'p,
    Rt: Runtime;

// SAFETY: the run is `A::cross_into`'s, the tuple's own crossing; nothing else
// crosses, and the capability is not kept.
unsafe impl<'p, A, Rt> crate::IntoRun<Rt> for PassedRun<'p, A, Rt>
where
    A: CallArgs<Rt> + 'p,
    Rt: Runtime,
{
    const WIDTH: usize = A::WIDTH;

    fn into_run(self, rt: Crossing<'_, Rt>, out: &mut [Rt::Value]) {
        A::cross_into(rt, self.0, out)
    }
}

/// A closure's parameter tuple, as the declared acvus type needs it.
pub trait ArgTypes: CanonicalArgs + Send + Sync + 'static {
    fn params(i: &Interner, vars: &PolyVars) -> Vec<ParamTerm<Poly>>;
}

/// A closure value called at the types its declaration names: the arguments
/// cross in and the result crosses out here, once (RFC-0039).
pub trait ClosureFn<Rt: Runtime> {
    type Args: CallArgs<Rt>;
    type Ret;

    /// Reached where the closure's effect said `Task::Sync`; every
    /// implementation asserts `is_sync`, the run-time answer, against it.
    fn call_now(
        &self,
        ctx: &mut Ctx<'_, Rt>,
        args: <Self::Args as CallArgs<Rt>>::Passed<'_>,
    ) -> Self::Ret;
    fn call<'a>(
        &'a self,
        ctx: &'a mut Ctx<'_, Rt>,
        args: <Self::Args as CallArgs<Rt>>::Passed<'a>,
    ) -> impl Future<Output = Self::Ret> + Send + 'a;
}

pub struct Closure<'a, A, R, E, Rt>(Owned<Rt>, bool, PhantomData<(&'a (), A, R, E)>)
where
    A: Send + Sync,
    R: Send + Sync,
    E: Var<kind::Effect>,
    Rt: Runtime;

// SAFETY: a `Closure` is at its own `'s`, and what a call passes it and
// what a call returns are at `'s`.
unsafe impl<'s, A, R, E, Rt> crate::Within<'s> for Closure<'s, A, R, E, Rt>
where
    A: Send + Sync + crate::Within<'s>,
    R: Send + Sync + crate::Within<'s>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
}

impl<'a, A, R, E, Rt> Closure<'a, A, R, E, Rt>
where
    A: Send + Sync,
    R: Send + Sync,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    /// Whether a call reaches its result without a future, asked of the
    /// runtime once, at the crossing.
    pub fn is_sync(&self) -> bool {
        self.1
    }
}

impl<'a, A, R, E, Rt> Closure<'a, A, R, E, Rt>
where
    A: CallArgs<Rt>,
    R: OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    /// `ClosureFn::call` on cells of the call's own (`Runtime::rooted`),
    /// for a handler that runs calls side by side, each needing a frame.
    /// The rooted `Ctx` is made and used inside the future and never lent
    /// to the handler, so the handler holds no second `Ctx` to exchange
    /// its own with.
    pub fn call_rooted<'c>(
        &'c self,
        rt: &'c Rt,
        args: A::Passed<'c>,
    ) -> impl Future<Output = R> + Send + 'c {
        async move {
            // SAFETY: the arguments and the result cross at the types this
            // `Closure` was materialized at, which the checker settled.
            let crossing = unsafe { Crossing::new(rt) };
            if self.1 {
                let mut rooted = rt.rooted();
                // SAFETY: the `Ctx` is lent to `call_now` alone and never
                // leaves this block.
                let ctx = unsafe { Rt::ctx_of(&mut rooted) };
                // SAFETY: as `ClosureFn::call_now`'s.
                return returned::<R, Rt>(crossing, unsafe {
                    rt.call_now(&self.0, ctx, PassedRun::<A, Rt>(args))
                });
            }
            // SAFETY: as `ClosureFn::call_now`'s.
            returned::<R, Rt>(crossing, unsafe { A::awaited(args, crossing, &self.0) }.await)
        }
    }
}

crate::cross_one_value!(
    Closure<'c, A, R, E, __Rt>,
    'c, A: Send + Sync, R: Send + Sync, E: Var<kind::Effect>
);

// SAFETY: `erase` hands back the closure's own word, and `materialize` holds
// the word it is handed as the closure the checker settled at this type, asking
// the runtime only whether a call of it is sync; nothing else crosses, and the
// capability is not kept.
unsafe impl<'a, A, R, E, Rt> crate::OneValue<Rt> for Closure<'a, A, R, E, Rt>
where
    A: Send + Sync,
    R: Send + Sync,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    fn erase(self, rt: Crossing<'_, Rt>) -> Rt::Value {
        self.0.into_value(rt.holding())
    }

    unsafe fn materialize(rt: Crossing<'_, Rt>, value: Rt::Value) -> Self {
        let sync = rt.call_is_sync(&value);
        // SAFETY: `materialize`'s caller hands over the word it owned.
        Self(unsafe { Owned::from_value(rt.holding(), value) }, sync, PhantomData)
    }
}

impl<'a, A, R, E, Rt> Var<kind::Type> for Closure<'a, A, R, E, Rt>
where
    A: CanonicalArgs,
    R: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
}

// SAFETY: the parameters and the result are their own canonical forms', the
// effect is its own, and the lifetime is at `'static`.
unsafe impl<'a, A, R, E, Rt> crate::Canonical<kind::Type> for Closure<'a, A, R, E, Rt>
where
    A: CanonicalArgs,
    R: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    type Canon = Closure<'static, A::Canon, R::Canon, E::Canon, Rt>;
}

// SAFETY: a `Closure` is an `Owned<Rt>` and a flag at every `A`, `R` and
// `E`.
unsafe impl<'a, M, A, R, E, Rt> crate::UniformPayload<M> for Closure<'a, A, R, E, Rt>
where
    A: Send + Sync,
    R: Send + Sync,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
}

impl<A, R, E, Rt> TyArg for Closure<'static, A, R, E, Rt>
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
            flows: acvus_mir::ty::Flows::Every.into(),
        }
    }
}

impl<'a, A, R, E, Rt> ClosureFn<Rt> for Closure<'a, A, R, E, Rt>
where
    A: CallArgs<Rt>,
    R: OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    type Args = A;
    type Ret = R;

    fn call_now(&self, ctx: &mut Ctx<'_, Rt>, args: A::Passed<'_>) -> R {
        debug_assert!(
            self.is_sync(),
            "a closure value whose effect's task is Sync suspends at run time (RFC-0046)"
        );
        let rt = ctx.rt;
        // SAFETY: as `call_rooted`'s.
        let crossing = unsafe { Crossing::new(rt) };
        // SAFETY: `self.0` is the closure value this `Closure` was built
        // over, and `A` is the argument list its declaration names.
        returned::<R, Rt>(crossing, unsafe {
            rt.call_now(&self.0, ctx, PassedRun::<A, Rt>(args))
        })
    }

    fn call<'c>(
        &'c self,
        ctx: &'c mut Ctx<'_, Rt>,
        args: A::Passed<'c>,
    ) -> impl Future<Output = R> + Send + 'c {
        async move {
            let rt = ctx.rt;
            // SAFETY: as `call_rooted`'s.
            let crossing = unsafe { Crossing::new(rt) };
            if self.1 {
                // SAFETY: as `call_now`'s.
                return returned::<R, Rt>(crossing, unsafe {
                    rt.call_now(&self.0, ctx, PassedRun::<A, Rt>(args))
                });
            }
            // SAFETY: as `call_now`'s.
            returned::<R, Rt>(crossing, unsafe { A::awaited(args, crossing, &self.0) }.await)
        }
    }
}

/// The value a call produced, read at the closure's declared return type.
fn returned<R, Rt>(rt: Crossing<'_, Rt>, out: Rt::Value) -> R
where
    R: OneValue<Rt>,
    Rt: Runtime,
{
    // SAFETY: the closure's declared return type is `R`, and what a result
    // at that type names outlives the call the closure was handed to.
    unsafe { R::materialize(rt, out) }
}

impl Args for () {}

impl CanonicalArgs for () {
    type Canon = ();
}

// SAFETY: no member crosses, and the capability makes the call and ends with
// it.
unsafe impl<Rt> CallArgs<Rt> for ()
where
    Rt: Runtime,
{
    type Passed<'a>
        = ()
    where
        Self: 'a;

    const WIDTH: usize = 0;

    fn cross_into(_: Crossing<'_, Rt>, _: (), _: &mut [Rt::Value]) {}

    unsafe fn awaited<'a>(
        _: (),
        rt: Crossing<'a, Rt>,
        f: &'a Rt::Value,
    ) -> impl Future<Output = Rt::Value> + Send + 'a {
        // SAFETY: the caller's contract.
        unsafe { rt.rt().call_0(f) }
    }
}

impl ArgTypes for () {
    fn params(_: &Interner, _: &PolyVars) -> Vec<ParamTerm<Poly>> {
        Vec::new()
    }
}

impl<A0> Args for (A0,) where A0: Send + Sync {}

impl<A0> CanonicalArgs for (A0,)
where
    A0: crate::Canonical<kind::Type> + Send + Sync,
{
    type Canon = (A0::Canon,);
}

/// The one argument the runtime has an entry of its own for.
// SAFETY: the one member crosses by its own `Passed`, and the capability makes
// the call and ends with it.
unsafe impl<A0, Rt> CallArgs<Rt> for (A0,)
where
    A0: for<'p> Passed<'p, Rt>,
    Rt: Runtime,
{
    type Passed<'a>
        = (<A0 as Passed<'a, Rt>>::As,)
    where
        Self: 'a;

    const WIDTH: usize = 1;

    fn cross_into(rt: Crossing<'_, Rt>, passed: Self::Passed<'_>, out: &mut [Rt::Value]) {
        out[0] = <A0 as Passed<'_, Rt>>::cross(rt, passed.0);
    }

    unsafe fn awaited<'a>(
        passed: Self::Passed<'a>,
        rt: Crossing<'a, Rt>,
        f: &'a Rt::Value,
    ) -> impl Future<Output = Rt::Value> + Send + 'a {
        let a = <A0 as Passed<'a, Rt>>::cross(rt, passed.0);
        // SAFETY: the caller's contract.
        unsafe { rt.rt().call_1(f, a) }
    }
}

/// Every wider tuple: the arguments are erased into a run and handed to
/// `call_n`. One more line is one more arity.
macro_rules! args_of {
    ($($A:ident: $at:tt),+) => {
        impl<$($A,)+> Args for ($($A,)+) where $($A: Send + Sync,)+ {}

        impl<$($A,)+> CanonicalArgs for ($($A,)+)
        where
            $($A: crate::Canonical<kind::Type> + Send + Sync,)+
        {
            type Canon = ($($A::Canon,)+);
        }

        // SAFETY: each member crosses by its own `Passed`, and the capability
        // makes the call and ends with it.
        unsafe impl<$($A,)+ Rt> CallArgs<Rt> for ($($A,)+)
        where
            $($A: for<'p> Passed<'p, Rt>,)+
            Rt: Runtime,
        {
            type Passed<'a>
                = ($(<$A as Passed<'a, Rt>>::As,)+)
            where
                Self: 'a;

            const WIDTH: usize = 0 $(+ { let _ = $at; 1 })+;

            fn cross_into(rt: Crossing<'_, Rt>, passed: Self::Passed<'_>, out: &mut [Rt::Value]) {
                $(out[$at] = <$A as Passed<'_, Rt>>::cross(rt, passed.$at);)+
            }

            unsafe fn awaited<'a>(
                passed: Self::Passed<'a>,
                rt: Crossing<'a, Rt>,
                f: &'a Rt::Value,
            ) -> impl Future<Output = Rt::Value> + Send + 'a {
                let mut run = [$(<$A as Passed<'a, Rt>>::cross(rt, passed.$at)),+];
                async move {
                    // SAFETY: the caller's contract.
                    unsafe { rt.rt().call_n(f, &mut run) }.await
                }
            }
        }
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
