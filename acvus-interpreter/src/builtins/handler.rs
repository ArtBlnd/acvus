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

// ── BuiltinFn trait (TypedValue-based) ─────────────────────────────

use super::types::{FromTyped, IntoTyped};
use crate::value::TypedValue;

/// A builtin function that converts args at runtime
/// (via `FromTyped`/`IntoTyped`).
pub trait BuiltinFn<Args> {
    fn call(&self, args: Vec<TypedValue>) -> Result<TypedValue, RuntimeError>;
}

// 1-arg: Fn(A) -> R
impl<F, A, R> BuiltinFn<(A,)> for F
where
    F: Fn(A) -> R,
    A: FromTyped,
    R: IntoTyped,
{
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
    A: FromTyped,
    B: FromTyped,
    R: IntoTyped,
{
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
    A: FromTyped,
    B: FromTyped,
    C: FromTyped,
    R: IntoTyped,
{
    fn call(&self, args: Vec<TypedValue>) -> Result<TypedValue, RuntimeError> {
        let mut it = args.into_iter();
        let a = A::from_typed(it.next().expect("missing arg 0"))?;
        let b = B::from_typed(it.next().expect("missing arg 1"))?;
        let c = C::from_typed(it.next().expect("missing arg 2"))?;
        Ok(self(a, b, c).into_typed())
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
