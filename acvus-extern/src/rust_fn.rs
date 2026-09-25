//! A Rust closure as a script function value (RFC-0097 rule 2).
//!
//! `P` admits no variable of the declaration, and that is a decision. `Args`
//! lends an argument only after checking the closure's parameter type
//! against the type the checker settled for it, and a call of a function
//! value hands its callee no settled type, so the view's types are fixed
//! where the value is made. A concrete member's type is its own; a
//! variable's is settled at the extern's call site as part of the site's
//! result type, which the prepared site does not carry until rule 3 adds it.
//! `R` needs no type: a result crosses by its own `erase`.

use std::marker::PhantomData;
use std::sync::Arc;

use acvus_mir::ty::{EffectTerm, Flows, ParamTerm, Poly, PolyTy, Ty, try_freeze_poly};
use acvus_utils::Interner;

use crate::args::{Args, ArgsSite, Members};
use crate::crossing::Crossing;
use crate::ctx::Ctx;
use crate::obj::OneValue;
use crate::owned::Owned;
use crate::runtime::Runtime;
use crate::ty_arg::{PolyVars, TyArg};

pub trait RustParams: Members + Send + Sync + 'static {
    fn declared_at_no_variable(interner: &Interner) -> Vec<PolyTy>;
}

macro_rules! rust_params {
    ($($member:ident),+) => {
        impl<$($member,)+> RustParams for ($($member,)+)
        where
            $($member: TyArg + Send + Sync + 'static,)+
        {
            fn declared_at_no_variable(interner: &Interner) -> Vec<PolyTy> {
                vec![$(<$member as TyArg>::poly_ty(interner, &PolyVars::empty())),+]
            }
        }
    };
}

rust_params!(A0);
rust_params!(A0, A1);
rust_params!(A0, A1, A2);
rust_params!(A0, A1, A2, A3);
rust_params!(A0, A1, A2, A3, A4);
rust_params!(A0, A1, A2, A3, A4, A5);
rust_params!(A0, A1, A2, A3, A4, A5, A6);
rust_params!(A0, A1, A2, A3, A4, A5, A6, A7);

const A_PARAMETER_NAMES_A_VARIABLE: &str = "a `RustFn` parameter is a concrete type: `Args` lends each \
     argument at the type its parameter names, and a call of a function value settles none \
     (RFC-0097 rule 2)";

/// The `expect` holds because `RustFn::declared_fn_ty` refuses the same
/// types where the extern's registry is built, before any value is made.
fn settled_tys<P>(interner: &Interner) -> Arc<[Ty]>
where
    P: RustParams,
{
    P::declared_at_no_variable(interner)
        .iter()
        .map(|ty| try_freeze_poly(ty).expect(A_PARAMETER_NAMES_A_VARIABLE))
        .collect()
}

pub struct RustFn<P, R, Rt>
where
    P: Members,
    Rt: Runtime,
{
    held: Held<Rt>,
    _types: PhantomData<fn() -> (P, R)>,
}

enum Held<Rt>
where
    Rt: Runtime,
{
    Unerased(RustBody<Rt>),
    Erased(Owned<Rt>),
}

impl<P, R, Rt> RustFn<P, R, Rt>
where
    P: RustParams,
    R: OneValue<Rt> + 'static,
    Rt: Runtime,
{
    pub fn new<F>(body: F) -> Self
    where
        F: Fn(&mut Ctx<'_, Rt>, Args<'_, P, Rt>) -> R + Send + Sync + 'static,
    {
        RustFn {
            held: Held::Unerased(RustBody {
                call: Box::new(Typed::<P, R, F>(body, PhantomData)),
                settled_tys: settled_tys::<P>,
            }),
            _types: PhantomData,
        }
    }
}

impl<P, R, Rt> RustFn<P, R, Rt>
where
    P: RustParams,
    R: TyArg,
    Rt: Runtime,
{
    /// `#[extern_fn]` writes this as the result type of an extern that
    /// returns a `RustFn`, with the effect the extern declares.
    #[doc(hidden)]
    pub fn declared_fn_ty(interner: &Interner, vars: &PolyVars, effect: EffectTerm<Poly>) -> PolyTy {
        let params = P::declared_at_no_variable(interner)
            .into_iter()
            .enumerate()
            .map(|(at, ty)| {
                assert!(try_freeze_poly(&ty).is_some(), "{A_PARAMETER_NAMES_A_VARIABLE}");
                ParamTerm::<Poly>::new(interner.intern(&format!("_{at}")), ty)
            })
            .collect();
        PolyTy::Fn {
            params,
            ret: Box::new(R::poly_ty(interner, vars)),
            captures: vec![],
            effect,
            flows: Flows::Every.into(),
        }
    }
}

struct Typed<P, R, F>(F, PhantomData<fn() -> (P, R)>);

trait ErasedCall<Rt>: Send + Sync
where
    Rt: Runtime,
{
    /// # Safety
    /// As `RustCallee::call`, and `site` holds `P`'s settled types.
    unsafe fn call(&self, ctx: &mut Ctx<'_, Rt>, run: &[Rt::Value], site: &ArgsSite) -> Rt::Value;
}

impl<P, R, F, Rt> ErasedCall<Rt> for Typed<P, R, F>
where
    P: RustParams,
    R: OneValue<Rt>,
    F: Fn(&mut Ctx<'_, Rt>, Args<'_, P, Rt>) -> R + Send + Sync,
    Rt: Runtime,
{
    unsafe fn call(&self, ctx: &mut Ctx<'_, Rt>, run: &[Rt::Value], site: &ArgsSite) -> Rt::Value {
        // SAFETY: the runtime makes this call only for a call of the
        // function value, whose arguments the checker typed at `P`'s types
        // and whose result at `R`; the closure is handed the view and `ctx`,
        // never the capability.
        let rt = unsafe { Crossing::new(ctx.rt) };
        // SAFETY: the caller's contract: `run` is `P::LEN` words the call
        // moved here, each crossed at the type `site` holds at its position,
        // owned by no other holder.
        let args = unsafe { Args::<P, Rt>::moved_in(rt, run, site) };
        (self.0)(ctx, args).erase(rt)
    }
}

pub struct RustBody<Rt>
where
    Rt: Runtime,
{
    call: Box<dyn ErasedCall<Rt>>,
    settled_tys: fn(&Interner) -> Arc<[Ty]>,
}

impl<Rt> RustBody<Rt>
where
    Rt: Runtime,
{
    /// `interner` is the one the runtime's checker settled its run's types
    /// under: `Args` compares types interned by both.
    pub fn settled_under(self, interner: &Interner) -> RustCallee<Rt> {
        let tys = (self.settled_tys)(interner);
        RustCallee {
            site: ArgsSite::new(interner.clone(), tys),
            call: self.call,
        }
    }
}

pub struct RustCallee<Rt>
where
    Rt: Runtime,
{
    site: ArgsSite,
    call: Box<dyn ErasedCall<Rt>>,
}

impl<Rt> RustCallee<Rt>
where
    Rt: Runtime,
{
    pub fn arity(&self) -> usize {
        self.site.len()
    }

    /// # Safety
    /// `run` is `arity()` words a call of this function value moved to it,
    /// each crossed at the type the checker settled for that parameter, and
    /// no other holder owns any of them. `ctx` is the call's, over cells
    /// that `run` does not name.
    pub unsafe fn call(&self, ctx: &mut Ctx<'_, Rt>, run: &[Rt::Value]) -> Rt::Value {
        // SAFETY: the caller's contract, and `site` is `P`'s settled types,
        // which a call of this value was checked at.
        unsafe { self.call.call(ctx, run, &self.site) }
    }
}

// SAFETY: `erase` hands the runtime the body to make its function value of,
// or the value a crossing handed over; `materialize` holds the word it is
// handed unopened. Nothing else crosses, and the capability is not kept.
unsafe impl<P, R, Rt> OneValue<Rt> for RustFn<P, R, Rt>
where
    P: Members,
    Rt: Runtime,
{
    fn erase(self, rt: Crossing<'_, Rt>) -> Rt::Value {
        match self.held {
            Held::Unerased(body) => rt.rt().rust_fn(body),
            Held::Erased(word) => word.into_value(rt.holding()),
        }
    }

    unsafe fn materialize(rt: Crossing<'_, Rt>, value: Rt::Value) -> Self {
        RustFn {
            // SAFETY: `materialize`'s caller hands over the word it owned.
            held: Held::Erased(unsafe { Owned::from_value(rt.holding(), value) }),
            _types: PhantomData,
        }
    }
}

crate::cross_as_one_value!(RustFn<P, R, Rt>, [P, R, Rt] at Rt where P: Members, Rt: Runtime);

// SAFETY: the type names no lifetime, and its state is `'static`, so it holds
// no carrier.
unsafe impl<'s, P, R, Rt> crate::Within<'s> for RustFn<P, R, Rt>
where
    P: Members,
    Rt: Runtime,
{
}
