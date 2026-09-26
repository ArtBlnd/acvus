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

/// `total` rests on rust_decimal 1.40.0: its `Display` writes into a 32-byte
/// buffer that panics when full, and its documented bound of scale 28 over a
/// 96-bit mantissa keeps the text within 31 bytes. A change of that
/// dependency re-opens this.
#[extern_fn(instance_of = acvus_extern::core::display, effect = pure, total)]
fn display_decimal(a: &Decimal, out: &mut String) {
    use std::fmt::Write;
    // `String`'s `write_str` returns `Ok` on every path, and
    // `rust_decimal::Decimal`'s `Display` returns what `pad_integral` into
    // that writer returns and no error of its own (rust_decimal 1.40.0,
    // `src/decimal.rs`; a change of that dependency re-opens this).
    write!(out, "{}", a.0).expect("a String's fmt::Write does not fail");
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

#[extern_fn(instance_of = acvus_extern::core::eq, effect = pure, law(equivalence))]
fn eq_decimal(a: &Decimal, b: &Decimal) -> bool {
    a.0 == b.0
}

#[extern_fn(instance_of = acvus_extern::core::clone, effect = pure, means(*a))]
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
            decimal, display_decimal, decimal_to_float,
            eq_decimal, clone_decimal, cmp_decimal,
            add_decimal, sub_decimal, mul_decimal, div_decimal, rem_decimal, neg_decimal,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC-0082 rule 9 sampled over the widest mantissas at every scale and
    /// a fixed-seed sample of parts.
    #[test]
    fn total_holds_over_display_decimal() {
        let mut state: u64 = 0x5eed_dec1_0c1a_7e00;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let mut values = Vec::new();
        for scale in 0..=rust_decimal::Decimal::MAX_SCALE {
            for negative in [false, true] {
                for (lo, mid, hi) in [(0, 0, 0), (1, 0, 0), (u32::MAX, u32::MAX, u32::MAX)] {
                    values.push(rust_decimal::Decimal::from_parts(lo, mid, hi, negative, scale));
                }
            }
        }
        for _ in 0..256 {
            let word = next();
            let scale = (word % u64::from(rust_decimal::Decimal::MAX_SCALE + 1)) as u32;
            let (lo, mid, hi) = (next() as u32, next() as u32, next() as u32);
            values.push(rust_decimal::Decimal::from_parts(lo, mid, hi, word & 1 == 1, scale));
        }
        for value in values {
            let mut out = String::new();
            display_decimal(&Decimal(value), &mut out);
            assert_eq!(out, value.to_string());
        }
    }

    /// RFC-0082 rule 11 sampled over values of one number at several scales:
    /// `eq` is reflexive, symmetric and transitive. No `core::hash` instance
    /// stands at `Decimal`, so no hash is held to agree with it.
    #[test]
    fn equivalence_holds_over_eq_decimal() {
        let values: Vec<Decimal> = ["0", "0.0", "-0", "1", "1.0", "1.00", "-1", "0.1", "0.10", "3.14159", "1000", "1000.0"]
            .into_iter()
            .map(|text| Decimal(text.parse().expect("a decimal literal")))
            .collect();
        for a in &values {
            assert!(eq_decimal(a, a), "reflexive at {}", a.0);
            for b in &values {
                assert_eq!(eq_decimal(a, b), eq_decimal(b, a), "symmetric at {}, {}", a.0, b.0);
                for c in &values {
                    if eq_decimal(a, b) && eq_decimal(b, c) {
                        assert!(eq_decimal(a, c), "transitive at {}, {}, {}", a.0, b.0, c.0);
                    }
                }
            }
        }
        assert!(eq_decimal(&values[3], &values[5]), "1 and 1.00 are one number");
    }
}
