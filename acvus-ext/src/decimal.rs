//! The `Decimal` extension type: `rust_decimal::Decimal`, exact where a
//! `Float` is not. A wire struct carries it as a field; on the wire it is
//! the number's text.

use acvus_extern::{ExternType, Registry, Runtime, TyArg, extern_fn, extern_registry};
use serde::{Deserialize, Serialize};

use crate::word::verdict;

#[derive(ExternType, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct Decimal(pub rust_decimal::Decimal);

/// Why a text is not a decimal: the text itself.
#[derive(TyArg)]
pub enum DecimalError {
    Unparsable(String),
}

#[extern_fn(effect = pure)]
fn decimal(text: String) -> Result<Decimal, DecimalError> {
    text.parse()
        .map(Decimal)
        .map_err(|_| DecimalError::Unparsable(text))
}

#[extern_fn(instance_of = acvus_extern::core::to_string, effect = pure)]
fn to_string_decimal(a: &Decimal) -> String {
    a.0.to_string()
}

/// The `Option` that `ToPrimitive::to_f64` returns is the trait's shape,
/// not a failure of this type: rust_decimal 1.40.0 builds a `Some` on every
/// path of its `to_f64`, and the `to_i128` its integral branch maps over is
/// `Some` on every path as well. That reading is of `src/decimal.rs` in
/// rust_decimal 1.40.0; a change of that dependency re-opens it.
#[extern_fn(effect = pure)]
fn decimal_to_float(a: &Decimal) -> f64 {
    let Some(f) = rust_decimal::prelude::ToPrimitive::to_f64(&a.0) else {
        unreachable!("rust_decimal::Decimal::to_f64 is Some for every value")
    };
    f
}

#[extern_fn(instance_of = acvus_extern::core::eq, effect = pure)]
fn eq_decimal(a: &Decimal, b: &Decimal) -> bool {
    a.0 == b.0
}

#[extern_fn(instance_of = acvus_extern::core::clone, effect = pure)]
fn clone_decimal(a: &Decimal) -> Decimal {
    a.clone()
}

#[extern_fn(instance_of = acvus_extern::core::cmp, effect = pure)]
fn cmp_decimal(a: &Decimal, b: &Decimal) -> i64 {
    verdict(a.0.cmp(&b.0))
}

// The arithmetic instances are rust_decimal's own operators, and no checked
// form answering a value on overflow or a zero divisor is built: the panic
// is the program's failure, as an integer operator's is (RFC-0037 rule 2).
#[extern_fn(instance_of = acvus_extern::core::add, effect = pure)]
fn add_decimal(a: &Decimal, b: &Decimal) -> Decimal {
    Decimal(a.0 + b.0)
}

#[extern_fn(instance_of = acvus_extern::core::sub, effect = pure)]
fn sub_decimal(a: &Decimal, b: &Decimal) -> Decimal {
    Decimal(a.0 - b.0)
}

#[extern_fn(instance_of = acvus_extern::core::mul, effect = pure)]
fn mul_decimal(a: &Decimal, b: &Decimal) -> Decimal {
    Decimal(a.0 * b.0)
}

#[extern_fn(instance_of = acvus_extern::core::div, effect = pure)]
fn div_decimal(a: &Decimal, b: &Decimal) -> Decimal {
    Decimal(a.0 / b.0)
}

#[extern_fn(instance_of = acvus_extern::core::rem, effect = pure)]
fn rem_decimal(a: &Decimal, b: &Decimal) -> Decimal {
    Decimal(a.0 % b.0)
}

#[extern_fn(instance_of = acvus_extern::core::neg, effect = pure)]
fn neg_decimal(a: &Decimal) -> Decimal {
    Decimal(-a.0)
}

pub fn decimal_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        types: [Decimal],
        fns: [
            decimal, to_string_decimal, decimal_to_float,
            eq_decimal, clone_decimal, cmp_decimal,
            add_decimal, sub_decimal, mul_decimal, div_decimal, rem_decimal, neg_decimal,
        ],
    }
}
