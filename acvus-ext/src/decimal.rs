//! The `Decimal` extension type: `rust_decimal::Decimal`, exact where a
//! `Float` is not. A wire struct carries it as a field; on the wire it is
//! the number's text.

use acvus_extern::{ExternError, ExternType, Registry, Runtime, extern_fn, extern_registry};
use serde::{Deserialize, Serialize};

use crate::conversion::sig;

#[derive(ExternType, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Decimal(pub rust_decimal::Decimal);

#[extern_fn(effect = pure)]
fn decimal(text: String) -> Result<Decimal, ExternError> {
    text.parse()
        .map(Decimal)
        .map_err(|e| ExternError::call("decimal", format!("invalid decimal '{text}': {e}")))
}

#[extern_fn(instance_of = sig::to_string, effect = pure)]
fn to_string_decimal(a: &Decimal) -> String {
    a.0.to_string()
}

#[extern_fn(effect = pure)]
fn decimal_to_float(a: &Decimal) -> Result<f64, ExternError> {
    rust_decimal::prelude::ToPrimitive::to_f64(&a.0)
        .ok_or_else(|| ExternError::call("decimal_to_float", format!("{} has no f64", a.0)))
}

#[extern_fn(instance_of = acvus_extern::core::eq, effect = pure)]
fn eq_decimal(a: &Decimal, b: &Decimal) -> bool {
    a.0 == b.0
}

#[extern_fn(instance_of = acvus_extern::core::clone, effect = pure)]
fn clone_decimal(a: &Decimal) -> Decimal {
    a.clone()
}

pub fn decimal_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        types: [Decimal],
        fns: [decimal, to_string_decimal, decimal_to_float, eq_decimal, clone_decimal],
    }
}
