//! The contract a runtime signs to run every declared ExternFn.
//!
//! A runtime chooses its own value representation. Each method builds or
//! opens one shape of value; a shape that is not there is the runtime's
//! own error.

use acvus_utils::{Astr, Interner};
use futures::future::BoxFuture;
use rustc_hash::FxHashMap;

use crate::convert::{FromValue, IntoValue};
use crate::error::ExternError;
use crate::extern_value::ExternValue;

pub trait Runtime: Sized + Send + Sync + 'static {
    /// The erased representation. A generic ExternFn's type variable is
    /// this at runtime, so it converts to and from itself.
    type Value: Clone + Send + Sync + 'static + FromValue<Self> + IntoValue<Self>;
    type Closure: Send + Sync + 'static;
    type Error: From<ExternError> + Send + Sync + 'static;

    fn materialize<T>(value: Self::Value, interner: &Interner) -> Result<T, Self::Error>
    where
        T: FromValue<Self>,
    {
        T::from_value(value, interner)
    }

    /// The language's `==`.
    fn equals(a: &Self::Value, b: &Self::Value) -> bool;

    fn unit() -> Self::Value;
    fn into_unit(value: Self::Value) -> Result<(), Self::Error>;
    fn int(n: i64) -> Self::Value;
    fn float(f: f64) -> Self::Value;
    fn bool(b: bool) -> Self::Value;
    fn byte(b: u8) -> Self::Value;
    fn small_bits(value: Self::Value) -> u64;
    fn string(s: String) -> Self::Value;
    fn into_string(value: Self::Value) -> Result<String, Self::Error>;

    fn array(items: Vec<Self::Value>) -> Self::Value;
    fn into_array(value: Self::Value) -> Result<Vec<Self::Value>, Self::Error>;
    fn tuple(items: Vec<Self::Value>) -> Self::Value;
    fn into_tuple(value: Self::Value) -> Result<Vec<Self::Value>, Self::Error>;
    fn object(fields: FxHashMap<Astr, Self::Value>) -> Self::Value;
    fn into_object(value: Self::Value) -> Result<FxHashMap<Astr, Self::Value>, Self::Error>;
    fn some(interner: &Interner, value: Self::Value) -> Self::Value;
    fn none(interner: &Interner) -> Self::Value;
    fn into_option(
        interner: &Interner,
        value: Self::Value,
    ) -> Result<Option<Self::Value>, Self::Error>;

    fn extern_value(value: ExternValue) -> Self::Value;
    fn into_extern(value: Self::Value) -> Result<ExternValue, Self::Error>;

    fn closure(closure: Self::Closure) -> Self::Value;
    fn into_closure(value: Self::Value) -> Result<Self::Closure, Self::Error>;
    fn call(
        closure: &Self::Closure,
        args: Vec<Self::Value>,
    ) -> BoxFuture<'_, Result<Self::Value, Self::Error>>;
}

/// A runtime that holds no values: for registering declarations where
/// nothing will ever run.
pub struct TypesOnly;

fn no_values<T>() -> Result<T, ExternError> {
    Err(ExternError::internal("TypesOnly runtime holds no values"))
}

impl Runtime for TypesOnly {
    type Value = ();
    type Closure = ();
    type Error = ExternError;

    fn equals(_: &(), _: &()) -> bool {
        true
    }
    fn unit() {}
    fn into_unit(_: ()) -> Result<(), ExternError> {
        no_values()
    }
    fn int(_: i64) {}
    fn float(_: f64) {}
    fn bool(_: bool) {}
    fn byte(_: u8) {}
    fn small_bits(_: ()) -> u64 {
        panic!("TypesOnly runtime holds no values")
    }
    fn string(_: String) {}
    fn into_string(_: ()) -> Result<String, ExternError> {
        no_values()
    }
    fn array(_: Vec<()>) {}
    fn into_array(_: ()) -> Result<Vec<()>, ExternError> {
        no_values()
    }
    fn tuple(_: Vec<()>) {}
    fn into_tuple(_: ()) -> Result<Vec<()>, ExternError> {
        no_values()
    }
    fn object(_: FxHashMap<Astr, ()>) {}
    fn into_object(_: ()) -> Result<FxHashMap<Astr, ()>, ExternError> {
        no_values()
    }
    fn some(_: &Interner, _: ()) {}
    fn none(_: &Interner) {}
    fn into_option(_: &Interner, _: ()) -> Result<Option<()>, ExternError> {
        no_values()
    }
    fn extern_value(_: ExternValue) {}
    fn into_extern(_: ()) -> Result<ExternValue, ExternError> {
        no_values()
    }
    fn closure(_: ()) {}
    fn into_closure(_: ()) -> Result<(), ExternError> {
        no_values()
    }
    fn call(_: &(), _: Vec<()>) -> BoxFuture<'_, Result<(), ExternError>> {
        Box::pin(std::future::ready(no_values()))
    }
}
