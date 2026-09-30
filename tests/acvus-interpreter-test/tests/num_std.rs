//! Every `num::` name this crate offers, at the script contract and at
//! both optimization levels.
//!
//! Each case is one script run through `corpus::attempt` at `Opt::None`
//! and at `Opt::Full`; a name whose two levels disagree fails here before
//! its value is read. The values are Rust's: a case is written as the
//! `std` expression it mirrors, and a float whose decimal expansion is not
//! exact is compared inside the script against a tolerance rather than
//! through the printed form.

use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_mir::graph::optimize::Opt;

/// The value both levels produce, rendered as `acvus run` prints it.
fn value(source: &str) -> String {
    let full = corpus::attempt(source, Opt::Full, Stage::Run);
    let none = corpus::attempt(source, Opt::None, Stage::Run);
    assert_eq!(full, none, "the two levels disagree on `{source}`");
    match full {
        Outcome::Value(v) => v,
        other => panic!("`{source}` produced no value: {other:?}"),
    }
}

fn is_true(source: &str) {
    assert_eq!(value(source), "true", "`{source}` is not true");
}

/// The message the run trapped with, at both levels.
fn trap(source: &str) -> String {
    let full = corpus::attempt(source, Opt::Full, Stage::Run);
    let none = corpus::attempt(source, Opt::None, Stage::Run);
    assert_eq!(full, none, "the two levels disagree on `{source}`");
    match full {
        Outcome::RunPanicked(m) => m,
        other => panic!("`{source}` did not trap: {other:?}"),
    }
}

/// The diagnostic the checker refused `source` with, at both levels.
fn refusal(source: &str) -> String {
    let full = corpus::attempt(source, Opt::Full, Stage::Run);
    let none = corpus::attempt(source, Opt::None, Stage::Run);
    assert_eq!(full, none, "the two levels disagree on `{source}`");
    match full {
        Outcome::Refused(m) => m,
        other => panic!("`{source}` was admitted: {other:?}"),
    }
}

// -- checked_* ----------------------------------------------------------

#[test]
fn checked_add_is_some_inside_the_width_and_none_past_it() {
    assert_eq!(value("1i8.checked_add(2i8).unwrap() as i64"), "3");
    assert_eq!(value("127i8.checked_add(1i8).unwrap_or(-7i8) as i64"), "-7");
}

#[test]
fn checked_sub_is_none_below_zero_on_an_unsigned_width() {
    assert_eq!(value("5u8.checked_sub(2u8).unwrap() as i64"), "3");
    assert_eq!(value("1u8.checked_sub(2u8).unwrap_or(9u8) as i64"), "9");
}

#[test]
fn checked_mul_is_none_past_the_width() {
    assert_eq!(value("16i32.checked_mul(16i32).unwrap() as i64"), "256");
    assert_eq!(
        value("2147483647i32.checked_mul(2i32).unwrap_or(-1i32) as i64"),
        "-1"
    );
}

#[test]
fn checked_div_and_rem_are_none_on_a_zero_divisor() {
    assert_eq!(value("7.checked_div(2).unwrap()"), "3");
    assert_eq!(value("7.checked_div(0).unwrap_or(-1)"), "-1");
    assert_eq!(value("7.checked_rem(2).unwrap()"), "1");
    assert_eq!(value("7.checked_rem(0).unwrap_or(-1)"), "-1");
}

#[test]
fn checked_neg_is_none_where_the_width_has_no_negative() {
    assert_eq!(value("5.checked_neg().unwrap()"), "-5");
    assert_eq!(value("0u8.checked_neg().unwrap() as i64"), "0");
    assert_eq!(value("1u8.checked_neg().unwrap_or(9u8) as i64"), "9");
}

#[test]
fn checked_pow_takes_a_u32_exponent_like_rusts() {
    assert_eq!(value("2.checked_pow(10).unwrap()"), "1024");
    assert_eq!(value("2i8.checked_pow(10).unwrap_or(-1i8) as i64"), "-1");
}

// -- wrapping_* ---------------------------------------------------------

#[test]
fn wrapping_add_and_sub_wrap_at_the_width() {
    assert_eq!(value("127i8.wrapping_add(1i8) as i64"), "-128");
    assert_eq!(value("0u8.wrapping_sub(1u8) as i64"), "255");
}

#[test]
fn wrapping_mul_wraps_at_the_width() {
    assert_eq!(value("200u8.wrapping_mul(2u8) as i64"), "144");
}

#[test]
fn wrapping_div_and_rem_wrap_where_the_quotient_leaves_the_width() {
    assert_eq!(value("7.wrapping_div(2)"), "3");
    assert_eq!(value("7.wrapping_rem(2)"), "1");
    assert_eq!(
        value("let a = -128i8; let b = -1i8; a.wrapping_div(b) as i64"),
        "-128"
    );
}

#[test]
fn wrapping_neg_wraps_at_the_width() {
    assert_eq!(value("0u8.wrapping_neg() as i64"), "0");
    assert_eq!(value("1u8.wrapping_neg() as i64"), "255");
}

#[test]
fn wrapping_pow_wraps_at_the_width() {
    assert_eq!(value("2i8.wrapping_pow(10) as i64"), "0");
    assert_eq!(value("3u8.wrapping_pow(5) as i64"), "243");
}

// -- overflowing_* ------------------------------------------------------
// The result is the tuple Rust returns, rendered as a two-element array:
// the wrapped value at its width, then whether the exact one left it.

#[test]
fn overflowing_add_wraps_and_says_so() {
    assert_eq!(value("127i8.overflowing_add(1i8)"), "[-128,true]");
    assert_eq!(value("1i8.overflowing_add(2i8)"), "[3,false]");
    assert_eq!(value("255u8.overflowing_add(1u8)"), "[0,true]");
    assert_eq!(value("1u8.overflowing_add(2u8)"), "[3,false]");
}

#[test]
fn overflowing_sub_wraps_and_says_so() {
    assert_eq!(
        value("let a = -128i8; a.overflowing_sub(1i8)"),
        "[127,true]"
    );
    assert_eq!(value("5i8.overflowing_sub(2i8)"), "[3,false]");
    assert_eq!(value("0u8.overflowing_sub(1u8)"), "[255,true]");
    assert_eq!(value("5u8.overflowing_sub(2u8)"), "[3,false]");
}

#[test]
fn overflowing_mul_wraps_and_says_so() {
    assert_eq!(
        value("2147483647i32.overflowing_mul(2i32)"),
        "[-2,true]"
    );
    assert_eq!(value("16i32.overflowing_mul(16i32)"), "[256,false]");
    assert_eq!(value("200u8.overflowing_mul(2u8)"), "[144,true]");
    assert_eq!(value("16u8.overflowing_mul(15u8)"), "[240,false]");
}

/// An unsigned quotient never leaves its width, so the unsigned case is
/// the exact one alone.
#[test]
fn overflowing_div_wraps_at_min_over_minus_one() {
    assert_eq!(
        value("let a = -128i8; let b = -1i8; a.overflowing_div(b)"),
        "[-128,true]"
    );
    assert_eq!(
        value("let a = i64::MIN(); let b = -1; a.overflowing_div(b)"),
        "[-9223372036854775808,true]"
    );
    assert_eq!(value("7i8.overflowing_div(2i8)"), "[3,false]");
    assert_eq!(value("7u8.overflowing_div(2u8)"), "[3,false]");
}

/// As for the quotient, an unsigned remainder never overflows.
#[test]
fn overflowing_rem_is_zero_and_overflowed_at_min_over_minus_one() {
    assert_eq!(
        value("let a = -128i8; let b = -1i8; a.overflowing_rem(b)"),
        "[0,true]"
    );
    assert_eq!(
        value("let a = i64::MIN(); let b = -1; a.overflowing_rem(b)"),
        "[0,true]"
    );
    assert_eq!(value("7i8.overflowing_rem(2i8)"), "[1,false]");
    assert_eq!(value("7u8.overflowing_rem(2u8)"), "[1,false]");
}

#[test]
fn overflowing_div_and_rem_trap_on_a_zero_divisor() {
    assert!(trap("7.overflowing_div(0)").contains("overflowing_div: divisor is zero"));
    assert!(trap("7.overflowing_rem(0)").contains("overflowing_rem: divisor is zero"));
}

#[test]
fn overflowing_neg_wraps_and_says_so() {
    assert_eq!(value("let a = -128i8; a.overflowing_neg()"), "[-128,true]");
    assert_eq!(value("5i8.overflowing_neg()"), "[-5,false]");
    assert_eq!(value("1u8.overflowing_neg()"), "[255,true]");
    assert_eq!(value("0u8.overflowing_neg()"), "[0,false]");
}

#[test]
fn overflowing_pow_wraps_and_says_so() {
    assert_eq!(value("2i8.overflowing_pow(10)"), "[0,true]");
    assert_eq!(value("2i8.overflowing_pow(6)"), "[64,false]");
    assert_eq!(value("3u8.overflowing_pow(6)"), "[217,true]");
    assert_eq!(value("3u8.overflowing_pow(5)"), "[243,false]");
}

/// Both halves reach a pattern: the tuple is a value the script takes
/// apart, not only one it returns.
#[test]
fn overflowing_add_s_halves_bind_in_a_pattern() {
    assert_eq!(
        value("match 255u8.overflowing_add(3u8) { (v, true) => v as i64, (v, false) => -(v as i64), _ => 0 }"),
        "2"
    );
}

// -- saturating_* -------------------------------------------------------

#[test]
fn saturating_add_and_sub_stop_at_the_bound() {
    assert_eq!(value("127i8.saturating_add(10i8) as i64"), "127");
    assert_eq!(value("0u8.saturating_sub(5u8) as i64"), "0");
}

#[test]
fn saturating_mul_and_pow_stop_at_the_bound() {
    assert_eq!(value("200u8.saturating_mul(2u8) as i64"), "255");
    assert_eq!(value("2i8.saturating_pow(10) as i64"), "127");
}

#[test]
fn saturating_div_stops_at_the_bound_where_the_quotient_leaves_the_width() {
    assert_eq!(value("7.saturating_div(2)"), "3");
    assert_eq!(
        value("let a = -128i8; let b = -1i8; a.saturating_div(b) as i64"),
        "127"
    );
}

#[test]
fn saturating_neg_stops_at_the_bound() {
    assert_eq!(value("let a = -128i8; a.saturating_neg() as i64"), "127");
}

// -- euclidean division -------------------------------------------------

#[test]
fn div_euclid_and_rem_euclid_round_toward_negative_infinity() {
    assert_eq!(value("let a = -7; a.div_euclid(4)"), "-2");
    assert_eq!(value("let a = -7; a.rem_euclid(4)"), "1");
    assert_eq!(value("7.div_euclid(4)"), "1");
    assert_eq!(value("7.rem_euclid(4)"), "3");
}

// -- bits ---------------------------------------------------------------

#[test]
fn leading_and_trailing_zeros_and_count_ones_read_the_width() {
    assert_eq!(value("1i8.leading_zeros() as i64"), "7");
    assert_eq!(value("1u64.leading_zeros() as i64"), "63");
    assert_eq!(value("8u8.trailing_zeros() as i64"), "3");
    assert_eq!(value("255u8.count_ones() as i64"), "8");
}

#[test]
fn swap_bytes_reverses_the_bytes() {
    assert_eq!(value("1u16.swap_bytes() as i64"), "256");
    assert_eq!(value("258u16.swap_bytes() as i64"), "513");
}

/// `to_be` and `to_le` are the host's endianness, so what is pinned here is
/// the identity that holds on either: one of the two is the value itself
/// and the other is its byte reversal.
#[test]
fn to_be_and_to_le_differ_by_a_byte_swap() {
    is_true("let x = 258u16; x.to_be().swap_bytes() == x.to_le()");
    is_true("let x = 258u32; x.to_le().swap_bytes() == x.to_be()");
}

// -- isqrt, powers of two -----------------------------------------------

#[test]
fn isqrt_is_the_floor_of_the_square_root() {
    assert_eq!(value("17.isqrt()"), "4");
    assert_eq!(value("16.isqrt()"), "4");
    assert_eq!(value("255u8.isqrt() as i64"), "15");
}

#[test]
fn is_power_of_two_and_next_power_of_two_are_unsigned() {
    is_true("8u8.is_power_of_two()");
    is_true("!6u8.is_power_of_two()");
    assert_eq!(value("5u8.next_power_of_two() as i64"), "8");
    assert_eq!(value("8u8.next_power_of_two() as i64"), "8");
}

// -- abs_diff -----------------------------------------------------------

/// The value is Rust's; the width is always `u64`, because one type
/// variable cannot say "the unsigned counterpart of `T`".
#[test]
fn abs_diff_is_the_distance_as_a_u64() {
    assert_eq!(value("3.abs_diff(10) as i64"), "7");
    assert_eq!(value("10.abs_diff(3) as i64"), "7");
    assert_eq!(value("3u8.abs_diff(10u8) as i64"), "7");
    is_true(
        "let a = -9223372036854775807 - 1; a.abs_diff(9223372036854775807) == 18446744073709551615u64",
    );
}

// -- radix text ---------------------------------------------------------

#[test]
fn to_hex_binary_and_octal_are_the_format_specifiers() {
    assert_eq!(value("255.to_hex()"), "\"ff\"");
    assert_eq!(value("5u8.to_binary()"), "\"101\"");
    assert_eq!(value("8u8.to_octal()"), "\"10\"");
}

/// `{:x}` on a signed value formats its two's complement, as Rust does.
#[test]
fn to_hex_of_a_negative_is_its_twos_complement() {
    assert_eq!(value("let a = 0i8 - 1i8; a.to_hex()"), "\"ff\"");
}

#[test]
fn from_str_radix_parses_at_the_radix() {
    assert_eq!(
        value("match i64::from_str_radix(\"ff\", 16) { Ok(v) => v, Err(e) => -1 }"),
        "255"
    );
    assert_eq!(
        value("match i64::from_str_radix(\"zz\", 16) { Ok(v) => v, Err(e) => -1 }"),
        "-1"
    );
    assert_eq!(
        value("match u8::from_str_radix(\"1000\", 2) { Ok(v) => v as i64, Err(e) => -1 }"),
        "8"
    );
}

// -- the associated constants -------------------------------------------

#[test]
fn min_max_and_bits_are_zero_argument_functions() {
    assert_eq!(value("i8::MAX() as i64"), "127");
    assert_eq!(value("i8::MIN() as i64"), "-128");
    assert_eq!(value("i32::BITS() as i64"), "32");
    assert_eq!(value("u64::MIN() as i64"), "0");
    is_true("u64::MAX() == 18446744073709551615u64");
    is_true("i64::MAX() == 9223372036854775807");
}

#[test]
fn the_float_constants_are_zero_argument_functions() {
    is_true("f64::NAN().is_nan()");
    is_true("!f64::INFINITY().is_finite()");
    is_true("f64::NEG_INFINITY() < 0.0");
    is_true("f64::EPSILON() > 0.0");
    is_true("f64::MAX() > 0.0");
    is_true("f64::MIN() < 0.0");
}

// -- the shared signatures at the widths they gained --------------------

#[test]
fn abs_min_max_clamp_signum_and_pow_reach_every_width() {
    assert_eq!(value("let a = 0i32 - 5i32; a.abs() as i64"), "5");
    assert_eq!(value("let a = 0i16 - 5i16; a.signum() as i64"), "-1");
    assert_eq!(value("min(3u16, 5u16) as i64"), "3");
    assert_eq!(value("max(3u16, 5u16) as i64"), "5");
    assert_eq!(value("clamp(9u8, 0u8, 5u8) as i64"), "5");
    assert_eq!(value("2u8.pow(3u8) as i64"), "8");
    assert_eq!(value("2u64.pow(10u64) as i64"), "1024");
}

// -- f64 ----------------------------------------------------------------

#[test]
fn rounding_and_fraction() {
    assert_eq!(value("floor(2.7)"), "2.0");
    assert_eq!(value("ceil(2.1)"), "3.0");
    assert_eq!(value("round(2.5)"), "3.0");
    assert_eq!(value("trunc(-2.7)"), "-2.0");
    assert_eq!(value("fract(2.5)"), "0.5");
}

#[test]
fn roots_exponentials_and_logarithms() {
    assert_eq!(value("sqrt(9.0)"), "3.0");
    assert_eq!(value("cbrt(27.0)"), "3.0");
    assert_eq!(value("exp(0.0)"), "1.0");
    assert_eq!(value("exp2(3.0)"), "8.0");
    assert_eq!(value("ln(1.0)"), "0.0");
    assert_eq!(value("log10(1000.0)"), "3.0");
    assert_eq!(value("log2(8.0)"), "3.0");
    is_true("(log(8.0, 2.0) - 3.0).abs() < 0.000000000001");
}

#[test]
fn the_trigonometric_functions_and_their_inverses() {
    assert_eq!(value("sin(0.0)"), "0.0");
    assert_eq!(value("cos(0.0)"), "1.0");
    assert_eq!(value("tan(0.0)"), "0.0");
    assert_eq!(value("asin(0.0)"), "0.0");
    assert_eq!(value("acos(1.0)"), "0.0");
    assert_eq!(value("atan(0.0)"), "0.0");
    assert_eq!(value("atan2(0.0, 1.0)"), "0.0");
    assert_eq!(value("sinh(0.0)"), "0.0");
    assert_eq!(value("cosh(0.0)"), "1.0");
    assert_eq!(value("tanh(0.0)"), "0.0");
}

#[test]
fn angles_reciprocals_and_the_pythagorean_length() {
    is_true("(to_degrees(3.141592653589793) - 180.0).abs() < 0.000000000001");
    is_true("(to_radians(180.0) - 3.141592653589793).abs() < 0.000000000001");
    assert_eq!(value("recip(4.0)"), "0.25");
    assert_eq!(value("hypot(3.0, 4.0)"), "5.0");
}

#[test]
fn powers_fused_multiply_add_and_the_sign_copy() {
    assert_eq!(value("powf(2.0, 10.0)"), "1024.0");
    assert_eq!(value("powi(2.0, 10)"), "1024.0");
    assert_eq!(value("mul_add(2.0, 3.0, 4.0)"), "10.0");
    assert_eq!(value("copysign(3.0, -1.0)"), "-3.0");
}

#[test]
fn the_float_predicates() {
    is_true("is_nan(0.0 / 0.0)");
    is_true("is_finite(1.0)");
    is_true("is_infinite(1.0 / 0.0)");
    is_true("is_normal(1.0)");
    is_true("!is_normal(0.0)");
    is_true("is_sign_negative(-0.0)");
    is_true("is_sign_positive(0.0)");
}

/// `total_cmp` is Rust's total order, so it separates the zeros and reads
/// a `NaN`'s sign bit, where `==` and `<` cannot. `0.0 / 0.0` is the
/// negative `NaN` on x86-64, which is why it sorts below `1.0` and
/// `f64::NAN`, the positive one, sorts above it.
#[test]
fn total_cmp_orders_every_f64() {
    assert_eq!(value("total_cmp(1.0, 2.0)"), "-1");
    assert_eq!(value("total_cmp(2.0, 2.0)"), "0");
    assert_eq!(value("total_cmp(2.0, 1.0)"), "1");
    assert_eq!(value("total_cmp(-0.0, 0.0)"), "-1");
    assert_eq!(value("total_cmp(f64::NAN(), 1.0)"), "1");
    assert_eq!(value("total_cmp(0.0 / 0.0, 1.0)"), "-1");
}

#[test]
fn to_bits_and_from_bits_are_inverse() {
    is_true("to_bits(1.0) == 4607182418800017408u64");
    assert_eq!(value("f64::from_bits(4607182418800017408u64)"), "1.0");
    is_true("f64::from_bits(to_bits(-2.5)) == -2.5");
}

// -- the traps ----------------------------------------------------------

#[test]
fn a_zero_divisor_traps_where_the_result_is_not_an_option() {
    assert!(trap("7.wrapping_div(0)").contains("wrapping_div: divisor is zero"));
    assert!(trap("7.wrapping_rem(0)").contains("wrapping_rem: divisor is zero"));
    assert!(trap("7.saturating_div(0)").contains("saturating_div: divisor is zero"));
    assert!(trap("7.div_euclid(0)").contains("div_euclid: divisor is zero"));
    assert!(trap("7.rem_euclid(0)").contains("rem_euclid: divisor is zero"));
}

#[test]
fn isqrt_of_a_negative_traps() {
    assert!(trap("let a = -1; a.isqrt()").contains("isqrt: negative argument -1"));
}

/// Rust's `next_power_of_two` panics in a debug build and wraps to `0` in a
/// release one. A script's behaviour may not depend on how the host was
/// compiled, so this traps at both.
#[test]
fn next_power_of_two_past_the_width_traps() {
    assert!(
        trap("200u8.next_power_of_two()")
            .contains("next_power_of_two: 200 has no next power of two")
    );
}

#[test]
fn a_radix_outside_rusts_range_traps() {
    assert!(
        trap("match i64::from_str_radix(\"1\", 1) { Ok(v) => v, Err(e) => -1 }")
            .contains("from_str_radix: radix 1 is outside 2..=36")
    );
}

// -- the refusals -------------------------------------------------------

#[test]
fn a_float_only_name_refuses_an_integer() {
    assert!(!refusal("is_nan(1)").is_empty());
    assert!(!refusal("sqrt(9)").is_empty());
}

#[test]
fn an_integer_only_name_refuses_a_float() {
    assert!(!refusal("1.5.count_ones()").is_empty());
    assert!(!refusal("1.5.checked_add(2.5)").is_empty());
}

/// The receiver is written with its width here. An *unsuffixed* literal
/// under a signature whose instances exclude `i64` — `8.is_power_of_two()`
/// — is admitted by `check` and reaches a poison instruction at run time
/// instead of a diagnostic. That is a hole in the solve, not in this
/// module, and it is not pinned here: a test asserting the poison would
/// make the defect the contract.
#[test]
fn a_signed_width_has_no_is_power_of_two_and_an_unsigned_one_has_no_abs() {
    assert!(!refusal("8i32.is_power_of_two()").is_empty());
    assert!(!refusal("8i32.next_power_of_two()").is_empty());
    assert!(!refusal("8u8.abs()").is_empty());
    assert!(!refusal("8u8.signum()").is_empty());
    assert!(!refusal("8u8.saturating_neg()").is_empty());
}

#[test]
fn a_name_over_numbers_refuses_a_string() {
    assert!(!refusal("\"x\".to_string().checked_add(1)").is_empty());
}

#[test]
fn from_str_radix_refuses_a_non_text_first_argument() {
    assert!(!refusal("match i64::from_str_radix(1, 16) { Ok(v) => v, Err(e) => -1 }").is_empty());
}
