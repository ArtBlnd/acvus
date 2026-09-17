//! `num::` and `core::hash` at the script contract: one name resolves to
//! the `i64` or the `f64` instance by its argument, a value the type admits
//! but the function cannot honor traps, and `hash` agrees with `==`.

use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::*;
use acvus_utils::Interner;

async fn run(source: &str) -> Value {
    let i = Interner::new();
    let registries = acvus_ext::std_registries::<AcvusRuntime>();
    run_script_mode_with_externs(&i, source, Context::default(), registries)
        .await
        .value
}

async fn int(source: &str) -> i64 {
    run(source).await.as_int()
}

async fn float(source: &str) -> f64 {
    run(source).await.as_float()
}

async fn boolean(source: &str) -> bool {
    run(source).await.as_bool()
}

// -- abs ----------------------------------------------------------------

#[tokio::test]
async fn abs_of_an_int() {
    assert_eq!(int("abs(-3)").await, 3);
}

#[tokio::test]
async fn abs_of_a_float() {
    assert_eq!(float("abs(-2.5)").await, 2.5);
}

#[tokio::test]
async fn abs_of_i64_min_is_i64_min() {
    assert_eq!(int("abs(-9223372036854775807 - 1)").await, i64::MIN);
}

// -- min / max ----------------------------------------------------------

#[tokio::test]
async fn min_of_ints_and_floats() {
    assert_eq!(int("min(3, 5)").await, 3);
    assert_eq!(float("min(2.5, 1.5)").await, 1.5);
}

#[tokio::test]
async fn max_of_ints_and_floats() {
    assert_eq!(int("max(3, 5)").await, 5);
    assert_eq!(float("max(2.5, 1.5)").await, 2.5);
}

// -- clamp --------------------------------------------------------------

#[tokio::test]
async fn clamp_of_an_int() {
    assert_eq!(int("clamp(7, 0, 5)").await, 5);
    assert_eq!(int("clamp(-2, 0, 5)").await, 0);
    assert_eq!(int("clamp(3, 0, 5)").await, 3);
}

#[tokio::test]
async fn clamp_of_a_float() {
    assert_eq!(float("clamp(7.5, 0.0, 5.0)").await, 5.0);
}

#[tokio::test]
#[should_panic(expected = "lower bound 5 is above upper bound 0")]
async fn clamp_of_an_int_with_reversed_bounds_traps() {
    run("clamp(1, 5, 0)").await;
}

#[tokio::test]
#[should_panic(expected = "not ordered")]
async fn clamp_of_a_float_with_reversed_bounds_traps() {
    run("clamp(1.0, 5.0, 0.0)").await;
}

#[tokio::test]
#[should_panic(expected = "not ordered")]
async fn clamp_of_a_float_with_a_nan_bound_traps() {
    run("clamp(1.0, 0.0, 0.0 / 0.0)").await;
}

// -- pow ----------------------------------------------------------------

#[tokio::test]
async fn pow_of_an_int() {
    assert_eq!(int("pow(2, 10)").await, 1024);
    assert_eq!(int("pow(-3, 3)").await, -27);
    assert_eq!(int("pow(7, 0)").await, 1);
}

#[tokio::test]
async fn pow_of_a_float() {
    let v = float("pow(2.0, 0.5)").await;
    assert!((v - 2f64.sqrt()).abs() < 1e-12, "pow(2.0, 0.5) = {v}");
}

#[tokio::test]
#[should_panic(expected = "negative exponent -1")]
async fn pow_of_an_int_with_a_negative_exponent_traps() {
    run("pow(2, -1)").await;
}

#[tokio::test]
async fn pow_of_an_int_past_the_width_wraps() {
    assert_eq!(int("pow(2, 63)").await, i64::MIN);
    assert_eq!(int("pow(2, 64)").await, 0);
    assert_eq!(int("pow(-2, 63)").await, i64::MIN);
}

#[tokio::test]
async fn pow_with_an_exponent_beyond_u32_is_finite_for_a_unit_or_zero_base() {
    assert_eq!(int("pow(1, 4294967296)").await, 1);
    assert_eq!(int("pow(0, 4294967296)").await, 0);
    assert_eq!(int("pow(-1, 4294967296)").await, 1);
    assert_eq!(int("pow(-1, 4294967297)").await, -1);
}

/// The odd-base values are `pow(3, 2**32, 2**64)` and
/// `pow(3, 2**32 + 1, 2**64)` read as an `i64`.
#[tokio::test]
async fn pow_with_an_exponent_beyond_u32_wraps() {
    assert_eq!(int("pow(2, 4294967296)").await, 0);
    assert_eq!(int("pow(-2, 4294967296)").await, 0);
    assert_eq!(int("pow(3, 4294967296)").await, 2491309678558969857);
    assert_eq!(int("pow(3, 4294967297)").await, 7473929035676909571);
}

// -- signum -------------------------------------------------------------

#[tokio::test]
async fn signum_of_ints_and_floats() {
    assert_eq!(int("signum(-7)").await, -1);
    assert_eq!(int("signum(0)").await, 0);
    assert_eq!(float("signum(-2.5)").await, -1.0);
    assert_eq!(float("signum(2.5)").await, 1.0);
}

// -- f64 only -----------------------------------------------------------

#[tokio::test]
async fn rounding_functions() {
    assert_eq!(float("floor(2.7)").await, 2.0);
    assert_eq!(float("ceil(2.1)").await, 3.0);
    assert_eq!(float("round(2.5)").await, 3.0);
    assert_eq!(float("trunc(-2.7)").await, -2.0);
}

#[tokio::test]
async fn roots_and_exponentials() {
    assert_eq!(float("sqrt(9.0)").await, 3.0);
    assert_eq!(float("cbrt(27.0)").await, 3.0);
    assert_eq!(float("exp(0.0)").await, 1.0);
    assert_eq!(float("ln(1.0)").await, 0.0);
    assert_eq!(float("log10(1000.0)").await, 3.0);
    assert_eq!(float("log2(8.0)").await, 3.0);
}

#[tokio::test]
async fn trigonometric_functions() {
    assert_eq!(float("sin(0.0)").await, 0.0);
    assert_eq!(float("cos(0.0)").await, 1.0);
    assert_eq!(float("tan(0.0)").await, 0.0);
}

#[tokio::test]
async fn nan_and_finiteness_predicates() {
    assert!(boolean("is_nan(0.0 / 0.0)").await);
    assert!(!boolean("is_nan(1.0)").await);
    assert!(boolean("is_finite(1.0)").await);
    assert!(!boolean("is_finite(1.0 / 0.0)").await);
}

// -- core::hash ---------------------------------------------------------

#[tokio::test]
#[should_panic(expected = "type mismatch")]
async fn hash_of_a_literal_is_a_type_error_because_a_literal_has_no_place_to_lend() {
    run("hash(1)").await;
}

#[tokio::test]
async fn hash_of_equal_ints_is_equal() {
    assert!(boolean("let a = 1; let b = 1; hash(&a) == hash(&b)").await);
}

#[tokio::test]
async fn hash_of_equal_strings_is_equal() {
    assert!(boolean(r#"let a = "a"; let b = "a"; hash(&a) == hash(&b)"#).await);
}

#[tokio::test]
async fn hash_of_different_strings_differs() {
    assert!(boolean(r#"let a = "a"; let b = "b"; hash(&a) != hash(&b)"#).await);
}

#[tokio::test]
async fn hash_of_bools() {
    assert!(boolean("let t = true; let u = true; hash(&t) == hash(&u)").await);
    assert!(boolean("let t = true; let f = false; hash(&t) != hash(&f)").await);
}

#[tokio::test]
async fn hash_of_floats_agrees_with_equality_on_signed_zero() {
    assert!(boolean("let a = 1.5; let b = 1.5; a.hash() == b.hash()").await);
    assert!(!boolean("let z = 0.0; let n = -0.0; z == n").await);
    assert!(boolean("let z = 0.0; let n = -0.0; hash(&z) != hash(&n)").await);
}
