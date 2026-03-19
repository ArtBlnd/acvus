use crate::error::RuntimeError;
use crate::value::Value;

// ── IntoValue ──────────────────────────────────────────────────────
// Converts a Rust value into a `Value`. Used by external crates
// (pomollu-engine, acvus-ext) for ExternFn return values.

pub trait IntoValue {
    fn into_value(self) -> Value;
}

impl IntoValue for i64 {
    fn into_value(self) -> Value { Value::int(self) }
}

impl IntoValue for f64 {
    fn into_value(self) -> Value { Value::float(self) }
}

impl IntoValue for String {
    fn into_value(self) -> Value { Value::string(self) }
}

impl IntoValue for bool {
    fn into_value(self) -> Value { Value::bool_(self) }
}

impl IntoValue for u8 {
    fn into_value(self) -> Value { Value::byte(self) }
}

impl IntoValue for Value {
    fn into_value(self) -> Value { self }
}

impl<T> IntoValue for Option<T>
where
    T: IntoValue,
{
    fn into_value(self) -> Value {
        let interner = crate::interner_ctx::get_interner()
            .expect("IntoValue<Option>: requires interner context");
        match self {
            Some(v) => Value::variant(
                interner.intern("Some"),
                Some(Box::new(v.into_value())),
            ),
            None => Value::variant(
                interner.intern("None"),
                None,
            ),
        }
    }
}

// ── BuiltinFn trait (TypedValue-based, with auto signature) ────────

use super::types::{FromTyped, IntoTyped, HasTy, SigCtx};
use crate::value::TypedValue;
use acvus_mir::ty::Ty;

/// A builtin function that can automatically extract its type signature
/// from its Rust parameter types (via `HasTy`) and convert args at runtime
/// (via `FromTyped`/`IntoTyped`).
pub trait BuiltinFn<Args> {
    fn signature(ctx: &mut SigCtx) -> (Vec<Ty>, Ty);
    fn call(&self, args: Vec<TypedValue>) -> Result<TypedValue, RuntimeError>;
}

// 1-arg: Fn(A) -> R
impl<F, A, R> BuiltinFn<(A,)> for F
where
    F: Fn(A) -> R,
    A: FromTyped + HasTy,
    R: IntoTyped + HasTy,
{
    fn signature(ctx: &mut SigCtx) -> (Vec<Ty>, Ty) {
        (vec![A::ty(ctx)], R::ty(ctx))
    }
    fn call(&self, args: Vec<TypedValue>) -> Result<TypedValue, RuntimeError> {
        let mut it = args.into_iter();
        let a = A::from_typed(it.next().expect("missing arg 0"))?;
        Ok(self(a).into_typed())
    }
}

// 2-arg: Fn(A, B) -> R
impl<F, A, B, R> BuiltinFn<(A, B)> for F
where
    F: Fn(A, B) -> R,
    A: FromTyped + HasTy,
    B: FromTyped + HasTy,
    R: IntoTyped + HasTy,
{
    fn signature(ctx: &mut SigCtx) -> (Vec<Ty>, Ty) {
        (vec![A::ty(ctx), B::ty(ctx)], R::ty(ctx))
    }
    fn call(&self, args: Vec<TypedValue>) -> Result<TypedValue, RuntimeError> {
        let mut it = args.into_iter();
        let a = A::from_typed(it.next().expect("missing arg 0"))?;
        let b = B::from_typed(it.next().expect("missing arg 1"))?;
        Ok(self(a, b).into_typed())
    }
}

// 3-arg: Fn(A, B, C) -> R
impl<F, A, B, C, R> BuiltinFn<(A, B, C)> for F
where
    F: Fn(A, B, C) -> R,
    A: FromTyped + HasTy,
    B: FromTyped + HasTy,
    C: FromTyped + HasTy,
    R: IntoTyped + HasTy,
{
    fn signature(ctx: &mut SigCtx) -> (Vec<Ty>, Ty) {
        (vec![A::ty(ctx), B::ty(ctx), C::ty(ctx)], R::ty(ctx))
    }
    fn call(&self, args: Vec<TypedValue>) -> Result<TypedValue, RuntimeError> {
        let mut it = args.into_iter();
        let a = A::from_typed(it.next().expect("missing arg 0"))?;
        let b = B::from_typed(it.next().expect("missing arg 1"))?;
        let c = C::from_typed(it.next().expect("missing arg 2"))?;
        Ok(self(a, b, c).into_typed())
    }
}

// ── Result-returning variants ──────────────────────────────────────

/// Marker wrapper to distinguish `Fn(A) -> Result<R, RuntimeError>` from `Fn(A) -> R`.
pub struct Fallible<F>(pub F);

impl<F, A, R> BuiltinFn<(Fallible<A>,)> for Fallible<F>
where
    F: Fn(A) -> Result<R, RuntimeError>,
    A: FromTyped + HasTy,
    R: IntoTyped + HasTy,
{
    fn signature(ctx: &mut SigCtx) -> (Vec<Ty>, Ty) {
        (vec![A::ty(ctx)], R::ty(ctx))
    }
    fn call(&self, args: Vec<TypedValue>) -> Result<TypedValue, RuntimeError> {
        let mut it = args.into_iter();
        let a = A::from_typed(it.next().expect("missing arg 0"))?;
        Ok(self.0(a)?.into_typed())
    }
}

impl<F, A, B, R> BuiltinFn<(Fallible<A>, B)> for Fallible<F>
where
    F: Fn(A, B) -> Result<R, RuntimeError>,
    A: FromTyped + HasTy,
    B: FromTyped + HasTy,
    R: IntoTyped + HasTy,
{
    fn signature(ctx: &mut SigCtx) -> (Vec<Ty>, Ty) {
        (vec![A::ty(ctx), B::ty(ctx)], R::ty(ctx))
    }
    fn call(&self, args: Vec<TypedValue>) -> Result<TypedValue, RuntimeError> {
        let mut it = args.into_iter();
        let a = A::from_typed(it.next().expect("missing arg 0"))?;
        let b = B::from_typed(it.next().expect("missing arg 1"))?;
        Ok(self.0(a, b)?.into_typed())
    }
}

// ── sync() wrapper ─────────────────────────────────────────────────

/// Type-erased builtin implementation. Stored in the ImplRegistry.
pub type BuiltinExecute = Box<dyn Fn(Vec<TypedValue>) -> Result<TypedValue, RuntimeError> + Send + Sync>;

/// Wrap a synchronous `BuiltinFn` into a `BuiltinExecute`.
pub fn sync<F, Args>(f: F) -> BuiltinExecute
where
    F: BuiltinFn<Args> + Send + Sync + 'static,
{
    Box::new(move |args| f.call(args))
}

/// Signature extractor — calls `BuiltinFn::signature` for a given function type.
pub fn extract_signature<F, Args>(subst: &mut acvus_mir::ty::TySubst) -> (Vec<Ty>, Ty)
where
    F: BuiltinFn<Args>,
{
    let mut ctx = SigCtx::new(subst);
    F::signature(&mut ctx)
}
