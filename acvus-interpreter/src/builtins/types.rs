//! Phantom types and conversion traits for declarative builtin signatures.
//!
//! Builtin functions are written as plain Rust functions whose parameter and
//! return types encode the script-level type signature:
//!
//! ```ignore
//! fn trim(s: String) -> String { ... }
//! fn map(iter: Iter<T<0>, E<0>>, f: Fun1<T<0>, T<1>, E<0>>) -> Iter<T<1>, E<0>> { ... }
//! ```
//!
//! The `HasTy` trait extracts the `Ty` representation from each Rust type,
//! and `FromTyped`/`IntoTyped` handle runtime value conversion.

use std::marker::PhantomData;

use acvus_mir::ty::{Effect, FnKind, Origin, Ty, TySubst, TyVar};

use crate::error::RuntimeError;
use crate::iter::{IterHandle, SequenceChain};
use crate::value::{FnValue, LazyValue, PureValue, TypedValue, Value};
use acvus_utils::TrackedDeque;

// ── Signature context ──────────────────────────────────────────────

/// Context for building a type signature from phantom types.
///
/// Caches variable allocations so that `T<0>` always maps to the same
/// `TyVar` within a single signature, regardless of how many times it appears.
pub struct SigCtx<'a> {
    subst: &'a mut TySubst,
    ty_vars: [Option<Ty>; 8],
    effect_vars: [Option<Effect>; 4],
    origin_vars: [Option<Origin>; 4],
}

impl<'a> SigCtx<'a> {
    pub fn new(subst: &'a mut TySubst) -> Self {
        Self {
            subst,
            ty_vars: Default::default(),
            effect_vars: Default::default(),
            origin_vars: Default::default(),
        }
    }

    pub fn ty_var(&mut self, n: usize) -> Ty {
        if let Some(ref ty) = self.ty_vars[n] {
            return ty.clone();
        }
        let ty = self.subst.fresh_var();
        self.ty_vars[n] = Some(ty.clone());
        ty
    }

    pub fn effect_var(&mut self, n: usize) -> Effect {
        if let Some(e) = self.effect_vars[n] {
            return e;
        }
        let e = self.subst.fresh_effect_var();
        self.effect_vars[n] = Some(e);
        e
    }

    pub fn origin_var(&mut self, n: usize) -> Origin {
        if let Some(o) = self.origin_vars[n] {
            return o;
        }
        let o = self.subst.fresh_origin();
        self.origin_vars[n] = Some(o);
        o
    }
}

// ── Phantom types ──────────────────────────────────────────────────

/// Type variable placeholder. `T<0>` and `T<1>` are distinct type variables.
pub struct T<const N: usize>;

/// Effect variable placeholder. `E<0>` represents a polymorphic effect.
pub struct E<const N: usize>;

/// Fixed pure effect.
pub struct PureEffect;

/// Origin variable placeholder. `O<0>` represents a polymorphic origin.
pub struct O<const N: usize>;

// ── HasTy / HasEffect / HasOrigin ──────────────────────────────────

/// Extract a `Ty` from a Rust type, using `SigCtx` for variable allocation.
pub trait HasTy {
    fn ty(ctx: &mut SigCtx) -> Ty;
}

/// Extract an `Effect` from a Rust type.
pub trait HasEffect {
    fn effect(ctx: &mut SigCtx) -> Effect;
}

/// Extract an `Origin` from a Rust type.
pub trait HasOrigin {
    fn origin(ctx: &mut SigCtx) -> Origin;
}

// ── HasTy impls: scalars ───────────────────────────────────────────

impl HasTy for String {
    fn ty(_ctx: &mut SigCtx) -> Ty { Ty::String }
}

impl HasTy for i64 {
    fn ty(_ctx: &mut SigCtx) -> Ty { Ty::Int }
}

impl HasTy for f64 {
    fn ty(_ctx: &mut SigCtx) -> Ty { Ty::Float }
}

impl HasTy for bool {
    fn ty(_ctx: &mut SigCtx) -> Ty { Ty::Bool }
}

impl HasTy for u8 {
    fn ty(_ctx: &mut SigCtx) -> Ty { Ty::Byte }
}

impl HasTy for () {
    fn ty(_ctx: &mut SigCtx) -> Ty { Ty::Unit }
}

// ── HasTy impls: type variables ────────────────────────────────────

impl<const N: usize> HasTy for T<N> {
    fn ty(ctx: &mut SigCtx) -> Ty {
        ctx.ty_var(N)
    }
}

// ── HasEffect impls ────────────────────────────────────────────────

impl HasEffect for PureEffect {
    fn effect(_ctx: &mut SigCtx) -> Effect { Effect::Pure }
}

impl<const N: usize> HasEffect for E<N> {
    fn effect(ctx: &mut SigCtx) -> Effect {
        ctx.effect_var(N)
    }
}

// ── HasOrigin impls ────────────────────────────────────────────────

impl<const N: usize> HasOrigin for O<N> {
    fn origin(ctx: &mut SigCtx) -> Origin {
        ctx.origin_var(N)
    }
}

// ── Wrapper types (phantom + runtime value) ────────────────────────

/// Iterator wrapper: `Iter<T<0>, E<0>>` = `Iterator<T, E>` in script.
pub struct Iter<Elem, Eff = PureEffect>(pub IterHandle, pub PhantomData<(Elem, Eff)>);

/// Deque wrapper: `Deq<T<0>, O<0>>` = `Deque<T, O>` in script.
pub struct Deq<Elem, Orig>(pub TrackedDeque<Value>, pub PhantomData<(Elem, Orig)>);

/// Sequence wrapper: `Seq<T<0>, O<0>, E<0>>` = `Sequence<T, O, E>` in script.
pub struct Seq<Elem, Orig, Eff>(pub SequenceChain, pub PhantomData<(Elem, Orig, Eff)>);

/// 1-arg function wrapper: `Fun1<A, R, E>` = `Fn(A) -> R` with effect E.
pub struct Fun1<Arg, Ret, Eff = PureEffect>(pub FnValue, pub PhantomData<(Arg, Ret, Eff)>);

/// 2-arg function wrapper: `Fun2<A, B, R, E>` = `Fn(A, B) -> R` with effect E.
pub struct Fun2<A, B, Ret, Eff = PureEffect>(pub FnValue, pub PhantomData<(A, B, Ret, Eff)>);

/// Option wrapper: `Opt<T<0>>` = `Option<T>` in script.
pub struct Opt<Inner>(pub Option<Value>, pub PhantomData<Inner>);

/// Polymorphic value wrapper: `TVal<T<0>>` documents a type-variable position.
///
/// At runtime this is just a `Value`. The phantom parameter is purely for
/// self-describing function signatures and `HasTy` extraction.
pub struct TVal<P>(pub Value, pub PhantomData<P>);

/// Tuple-2 phantom marker: `Tup2<A, B>` = `(A, B)` in script.
///
/// Used inside other wrappers (e.g. `Opt<Tup2<TVal<T<0>>, Iter<T<0>, E<0>>>>`)
/// for type-level documentation. Not constructed at runtime — the actual value
/// is a `Value::Tuple` built directly.
pub struct Tup2<A, B>(pub PhantomData<(A, B)>);

// ── HasTy impls: containers ────────────────────────────────────────

impl<Elem: HasTy, Eff: HasEffect> HasTy for Iter<Elem, Eff> {
    fn ty(ctx: &mut SigCtx) -> Ty {
        Ty::Iterator(Box::new(Elem::ty(ctx)), Eff::effect(ctx))
    }
}

impl<Elem: HasTy, Orig: HasOrigin> HasTy for Deq<Elem, Orig> {
    fn ty(ctx: &mut SigCtx) -> Ty {
        Ty::Deque(Box::new(Elem::ty(ctx)), Orig::origin(ctx))
    }
}

impl<Elem: HasTy, Orig: HasOrigin, Eff: HasEffect> HasTy for Seq<Elem, Orig, Eff> {
    fn ty(ctx: &mut SigCtx) -> Ty {
        Ty::Sequence(Box::new(Elem::ty(ctx)), Orig::origin(ctx), Eff::effect(ctx))
    }
}

impl<Arg: HasTy, Ret: HasTy, Eff: HasEffect> HasTy for Fun1<Arg, Ret, Eff> {
    fn ty(ctx: &mut SigCtx) -> Ty {
        Ty::Fn {
            params: vec![Arg::ty(ctx)],
            ret: Box::new(Ret::ty(ctx)),
            kind: FnKind::Lambda,
            captures: vec![],
            effect: Eff::effect(ctx),
        }
    }
}

impl<A: HasTy, B: HasTy, Ret: HasTy, Eff: HasEffect> HasTy for Fun2<A, B, Ret, Eff> {
    fn ty(ctx: &mut SigCtx) -> Ty {
        Ty::Fn {
            params: vec![A::ty(ctx), B::ty(ctx)],
            ret: Box::new(Ret::ty(ctx)),
            kind: FnKind::Lambda,
            captures: vec![],
            effect: Eff::effect(ctx),
        }
    }
}

impl<Inner: HasTy> HasTy for Opt<Inner> {
    fn ty(ctx: &mut SigCtx) -> Ty {
        Ty::Option(Box::new(Inner::ty(ctx)))
    }
}

impl<Elem: HasTy> HasTy for Vec<Elem> {
    fn ty(ctx: &mut SigCtx) -> Ty {
        Ty::List(Box::new(Elem::ty(ctx)))
    }
}

impl<P: HasTy> HasTy for TVal<P> {
    fn ty(ctx: &mut SigCtx) -> Ty {
        P::ty(ctx)
    }
}

impl<A: HasTy, B: HasTy> HasTy for Tup2<A, B> {
    fn ty(ctx: &mut SigCtx) -> Ty {
        Ty::Tuple(vec![A::ty(ctx), B::ty(ctx)])
    }
}

// ── FromTyped / IntoTyped ──────────────────────────────────────────

/// Convert a `TypedValue` into a Rust type.
pub trait FromTyped: Sized {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError>;
}

/// Convert a Rust type into a `TypedValue`.
pub trait IntoTyped {
    fn into_typed(self) -> TypedValue;
}

// --- Scalars ---

impl FromTyped for String {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        match tv.into_inner() {
            Value::Pure(PureValue::String(s)) => Ok(s),
            other => Err(RuntimeError::unexpected_type("builtin", &[crate::error::ValueKind::String], other.kind())),
        }
    }
}

impl FromTyped for i64 {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        match tv.into_inner() {
            Value::Pure(PureValue::Int(n)) => Ok(n),
            other => Err(RuntimeError::unexpected_type("builtin", &[crate::error::ValueKind::Int], other.kind())),
        }
    }
}

impl FromTyped for f64 {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        match tv.into_inner() {
            Value::Pure(PureValue::Float(f)) => Ok(f),
            other => Err(RuntimeError::unexpected_type("builtin", &[crate::error::ValueKind::Float], other.kind())),
        }
    }
}

impl FromTyped for bool {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        match tv.into_inner() {
            Value::Pure(PureValue::Bool(b)) => Ok(b),
            other => Err(RuntimeError::unexpected_type("builtin", &[crate::error::ValueKind::Bool], other.kind())),
        }
    }
}

impl FromTyped for u8 {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        match tv.into_inner() {
            Value::Pure(PureValue::Byte(b)) => Ok(b),
            other => Err(RuntimeError::unexpected_type("builtin", &[crate::error::ValueKind::Byte], other.kind())),
        }
    }
}

impl FromTyped for Value {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        Ok(tv.into_inner())
    }
}

impl FromTyped for TypedValue {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        Ok(tv)
    }
}

// --- Containers ---

impl<Elem: FromTyped> FromTyped for Vec<Elem> {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        match tv.into_inner() {
            Value::Lazy(LazyValue::List(items)) => {
                items.into_iter().map(|v| {
                    // List elements don't carry individual types — use Ty::error as placeholder
                    Elem::from_typed(TypedValue::new(v, Ty::error()))
                }).collect()
            }
            other => Err(RuntimeError::unexpected_type("builtin", &[crate::error::ValueKind::List], other.kind())),
        }
    }
}

impl FromTyped for IterHandle {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        match tv.into_inner() {
            Value::Lazy(LazyValue::Iterator(ih)) => Ok(ih),
            other => Err(RuntimeError::unexpected_type("builtin", &[crate::error::ValueKind::Iterator], other.kind())),
        }
    }
}

impl FromTyped for FnValue {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        match tv.into_inner() {
            Value::Lazy(LazyValue::Fn(fv)) => Ok(fv),
            other => Err(RuntimeError::unexpected_type("builtin", &[crate::error::ValueKind::Fn], other.kind())),
        }
    }
}

impl FromTyped for SequenceChain {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        match tv.into_inner() {
            Value::Lazy(LazyValue::Sequence(sc)) => Ok(sc),
            other => Err(RuntimeError::unexpected_type("builtin", &[crate::error::ValueKind::Sequence], other.kind())),
        }
    }
}

impl FromTyped for TrackedDeque<Value> {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        match tv.into_inner() {
            Value::Lazy(LazyValue::Deque(d)) => Ok(d),
            other => Err(RuntimeError::unexpected_type("builtin", &[crate::error::ValueKind::Deque], other.kind())),
        }
    }
}

// --- Wrapper types ---

impl<Elem, Eff> FromTyped for Iter<Elem, Eff> {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        let ih = IterHandle::from_typed(tv)?;
        Ok(Iter(ih, PhantomData))
    }
}

impl<Elem, Orig> FromTyped for Deq<Elem, Orig> {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        let d = TrackedDeque::from_typed(tv)?;
        Ok(Deq(d, PhantomData))
    }
}

impl<Elem, Orig, Eff> FromTyped for Seq<Elem, Orig, Eff> {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        let sc = SequenceChain::from_typed(tv)?;
        Ok(Seq(sc, PhantomData))
    }
}

impl<Arg, Ret, Eff> FromTyped for Fun1<Arg, Ret, Eff> {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        let fv = FnValue::from_typed(tv)?;
        Ok(Fun1(fv, PhantomData))
    }
}

impl<A, B, Ret, Eff> FromTyped for Fun2<A, B, Ret, Eff> {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        let fv = FnValue::from_typed(tv)?;
        Ok(Fun2(fv, PhantomData))
    }
}

impl<Inner: FromTyped> FromTyped for Opt<Inner> {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        let interner = crate::interner_ctx::get_interner()
            .expect("FromTyped<Opt>: requires interner context");
        let some_tag = interner.intern("Some");
        let none_tag = interner.intern("None");
        match tv.into_inner() {
            Value::Lazy(LazyValue::Variant { tag, payload: Some(inner) }) if tag == some_tag => {
                Ok(Opt(Some(*inner), PhantomData))
            }
            Value::Lazy(LazyValue::Variant { tag, .. }) if tag == none_tag => {
                Ok(Opt(None, PhantomData))
            }
            other => Err(RuntimeError::unexpected_type("builtin", &[crate::error::ValueKind::Variant], other.kind())),
        }
    }
}

// --- IntoTyped: scalars ---

impl IntoTyped for String {
    fn into_typed(self) -> TypedValue { TypedValue::string(self) }
}

impl IntoTyped for i64 {
    fn into_typed(self) -> TypedValue { TypedValue::int(self) }
}

impl IntoTyped for f64 {
    fn into_typed(self) -> TypedValue { TypedValue::float(self) }
}

impl IntoTyped for bool {
    fn into_typed(self) -> TypedValue { TypedValue::bool_(self) }
}

impl IntoTyped for u8 {
    fn into_typed(self) -> TypedValue { TypedValue::byte(self) }
}

impl IntoTyped for () {
    fn into_typed(self) -> TypedValue { TypedValue::unit() }
}

impl IntoTyped for Value {
    fn into_typed(self) -> TypedValue {
        // Caller is responsible for correct Ty — this is a low-level escape hatch.
        TypedValue::new(self, Ty::error())
    }
}

impl IntoTyped for TypedValue {
    fn into_typed(self) -> TypedValue { self }
}

// --- IntoTyped: containers ---

impl IntoTyped for IterHandle {
    fn into_typed(self) -> TypedValue {
        let effect = self.effect();
        // Element type is erased at runtime — use Ty::error() as placeholder.
        // The type checker has already validated this.
        TypedValue::new(
            Value::iterator(self),
            Ty::Iterator(Box::new(Ty::error()), effect),
        )
    }
}

impl IntoTyped for SequenceChain {
    fn into_typed(self) -> TypedValue {
        TypedValue::new(
            Value::sequence(self),
            Ty::Sequence(Box::new(Ty::error()), Origin::Var(0), Effect::Pure),
        )
    }
}

impl IntoTyped for TrackedDeque<Value> {
    fn into_typed(self) -> TypedValue {
        TypedValue::new(
            Value::Lazy(LazyValue::Deque(self)),
            Ty::Deque(Box::new(Ty::error()), Origin::Var(0)),
        )
    }
}

// Wrapper IntoTyped — delegates to inner type

impl<Elem, Eff> IntoTyped for Iter<Elem, Eff> {
    fn into_typed(self) -> TypedValue { self.0.into_typed() }
}

impl<Elem, Orig> IntoTyped for Deq<Elem, Orig> {
    fn into_typed(self) -> TypedValue { self.0.into_typed() }
}

impl<Elem, Orig, Eff> IntoTyped for Seq<Elem, Orig, Eff> {
    fn into_typed(self) -> TypedValue { self.0.into_typed() }
}

impl<Inner> IntoTyped for Opt<Inner> {
    fn into_typed(self) -> TypedValue {
        let interner = crate::interner_ctx::get_interner()
            .expect("IntoTyped<Opt>: requires interner context");
        match self.0 {
            Some(v) => TypedValue::new(
                Value::variant(interner.intern("Some"), Some(Box::new(v))),
                Ty::Option(Box::new(Ty::error())),
            ),
            None => TypedValue::new(
                Value::variant(interner.intern("None"), None),
                Ty::Option(Box::new(Ty::error())),
            ),
        }
    }
}

impl<P> FromTyped for TVal<P> {
    fn from_typed(tv: TypedValue) -> Result<Self, RuntimeError> {
        Ok(TVal(tv.into_inner(), PhantomData))
    }
}

impl<P> IntoTyped for TVal<P> {
    fn into_typed(self) -> TypedValue {
        TypedValue::new(self.0, Ty::error())
    }
}

impl<Elem: IntoTyped> IntoTyped for Vec<Elem> {
    fn into_typed(self) -> TypedValue {
        let values: Vec<Value> = self.into_iter().map(|v| v.into_typed().into_inner()).collect();
        TypedValue::new(Value::list(values), Ty::List(Box::new(Ty::error())))
    }
}
