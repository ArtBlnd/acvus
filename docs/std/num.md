# `num`: numbers under Rust's names

Every name in this module is the `std` method of the same name, with the
same result. A script writes `x.checked_add(y)`, `x.abs()`,
`sqrt(x)`, `i64::MAX()` — no `_int`/`_float` suffix, no hand-rolled
`if a < b { a } else { b }`.

A method Rust writes over many integer widths is a shared signature
(RFC-0019) with an instance per width, so one name serves all eight. A
method Rust puts only on `f64` is a plain function. Below, `int` is the
eight widths `i8 i16 i32 i64 u8 u16 u32 u64`, `signed` is `i8 i16 i32
i64`, and `unsigned` is `u8 u16 u32 u64`.

`from_str_radix` is in the table because it is a number's `std` method,
but it is declared beside `from_str` under each integer namespace, where
the parse error type lives.

## The integer rule

The language's own arithmetic is Rust's in a release build, at the width
(RFC-0037, RFC-0058): `+`, `-`, `*` and negation wrap, a shift takes its
amount modulo the width, and `/` and `%` trap on a zero divisor and at
`MIN / -1` with Rust's own texts. `127i8 + 1i8` is `-128`.

Where Rust's own choice is a debug-build panic, the function here follows
the language's rule instead: `abs`, `pow`, `div_euclid` and `rem_euclid`
wrap. The `checked_*`, `wrapping_*` and `saturating_*` families are how a
script asks for the other answers explicitly. `next_power_of_two` is the
one exception and traps at every build profile, because Rust's release
answer there is `0` — a substituted failure, not an arithmetic one.

## What `std` has and this module does not

- **`overflowing_*`.** Its result is `(T, bool)` and a Rust tuple does not
  cross the extern boundary, so it is not declared. A script that wants
  both halves calls `checked_*` and `wrapping_*`.
- **`abs_diff`'s result type.** Rust returns the unsigned counterpart of
  the argument — `i64::abs_diff` gives a `u64`. One type variable cannot
  say "the unsigned counterpart of `T`", so the result is `u64` at every
  width. The value is Rust's; only the width is widened.
- **Associated constants.** `MIN`, `MAX`, `BITS`, `EPSILON`, `INFINITY`,
  `NEG_INFINITY` and `NAN` are values in Rust and the language has no
  associated constant, so each is a zero-argument function under its
  type's namespace: `i64::MAX()`, `f64::NAN()`.

## The surface

A trap's *text* is this module's own and names the function that raised it
(`clamp: lower bound 5 is above upper bound 0`), where Rust's is generic.
That holds for every trap below and is not repeated per row; the
difference column carries behaviour only.

| language name | signature | Rust `std` twin | semantic difference |
| --- | --- | --- | --- |
| `abs` | `abs(a: T) -> T`, T ∈ signed, f64 | `T::abs` | `MIN.abs()` is `MIN`; Rust's `i8::abs` panics in a debug build |
| `signum` | `signum(a: T) -> T`, T ∈ signed, f64 | `T::signum` | none |
| `min` | `min(a: T, b: T) -> T`, T ∈ int, f64 | `Ord::min`, `f64::min` | none |
| `max` | `max(a: T, b: T) -> T`, T ∈ int, f64 | `Ord::max`, `f64::max` | none |
| `clamp` | `clamp(x: T, lo: T, hi: T) -> T`, T ∈ int, f64 | `Ord::clamp`, `f64::clamp` | none |
| `pow` | `pow(base: T, exp: T) -> T`, T ∈ int, f64 | `T::pow`, `f64::powf` | the exponent is `T`, not `u32`: a negative one traps, one past `u32::MAX` runs the same square-and-multiply, and overflow wraps where Rust's `pow` panics in a debug build |
| `checked_add` | `checked_add(a: T, b: T) -> Option<T>`, T ∈ int | `T::checked_add` | none |
| `checked_sub` | `checked_sub(a: T, b: T) -> Option<T>`, T ∈ int | `T::checked_sub` | none |
| `checked_mul` | `checked_mul(a: T, b: T) -> Option<T>`, T ∈ int | `T::checked_mul` | none |
| `checked_div` | `checked_div(a: T, b: T) -> Option<T>`, T ∈ int | `T::checked_div` | none |
| `checked_rem` | `checked_rem(a: T, b: T) -> Option<T>`, T ∈ int | `T::checked_rem` | none |
| `checked_neg` | `checked_neg(a: T) -> Option<T>`, T ∈ int | `T::checked_neg` | none |
| `checked_pow` | `checked_pow(base: T, exp: u32) -> Option<T>`, T ∈ int | `T::checked_pow` | none |
| `wrapping_add` | `wrapping_add(a: T, b: T) -> T`, T ∈ int | `T::wrapping_add` | none |
| `wrapping_sub` | `wrapping_sub(a: T, b: T) -> T`, T ∈ int | `T::wrapping_sub` | none |
| `wrapping_mul` | `wrapping_mul(a: T, b: T) -> T`, T ∈ int | `T::wrapping_mul` | none |
| `wrapping_div` | `wrapping_div(a: T, b: T) -> T`, T ∈ int | `T::wrapping_div` | none |
| `wrapping_rem` | `wrapping_rem(a: T, b: T) -> T`, T ∈ int | `T::wrapping_rem` | none |
| `wrapping_neg` | `wrapping_neg(a: T) -> T`, T ∈ int | `T::wrapping_neg` | none |
| `wrapping_pow` | `wrapping_pow(base: T, exp: u32) -> T`, T ∈ int | `T::wrapping_pow` | none |
| `saturating_add` | `saturating_add(a: T, b: T) -> T`, T ∈ int | `T::saturating_add` | none |
| `saturating_sub` | `saturating_sub(a: T, b: T) -> T`, T ∈ int | `T::saturating_sub` | none |
| `saturating_mul` | `saturating_mul(a: T, b: T) -> T`, T ∈ int | `T::saturating_mul` | none |
| `saturating_div` | `saturating_div(a: T, b: T) -> T`, T ∈ int | `T::saturating_div` | none |
| `saturating_neg` | `saturating_neg(a: T) -> T`, T ∈ signed | `T::saturating_neg` | none |
| `saturating_pow` | `saturating_pow(base: T, exp: u32) -> T`, T ∈ int | `T::saturating_pow` | none |
| `div_euclid` | `div_euclid(a: T, b: T) -> T`, T ∈ int | `T::div_euclid` | `MIN.div_euclid(-1)` wraps, where Rust panics |
| `rem_euclid` | `rem_euclid(a: T, b: T) -> T`, T ∈ int | `T::rem_euclid` | `MIN.rem_euclid(-1)` wraps, where Rust panics |
| `leading_zeros` | `leading_zeros(a: T) -> u32`, T ∈ int | `T::leading_zeros` | none |
| `trailing_zeros` | `trailing_zeros(a: T) -> u32`, T ∈ int | `T::trailing_zeros` | none |
| `count_ones` | `count_ones(a: T) -> u32`, T ∈ int | `T::count_ones` | none |
| `swap_bytes` | `swap_bytes(a: T) -> T`, T ∈ int | `T::swap_bytes` | none |
| `to_be` | `to_be(a: T) -> T`, T ∈ int | `T::to_be` | none |
| `to_le` | `to_le(a: T) -> T`, T ∈ int | `T::to_le` | none |
| `isqrt` | `isqrt(a: T) -> T`, T ∈ int | `T::isqrt` | none |
| `is_power_of_two` | `is_power_of_two(a: T) -> bool`, T ∈ unsigned | `T::is_power_of_two` | none |
| `next_power_of_two` | `next_power_of_two(a: T) -> T`, T ∈ unsigned | `T::next_power_of_two` | overflow traps at every build profile; Rust panics only in a debug build and wraps to `0` in a release one |
| `abs_diff` | `abs_diff(a: T, b: T) -> u64`, T ∈ int | `T::abs_diff` | the result is `u64` at every width, not the unsigned counterpart of `T`; the value is Rust's |
| `to_hex` | `to_hex(a: T) -> String`, T ∈ int | `format!("{:x}", a)` | none |
| `to_binary` | `to_binary(a: T) -> String`, T ∈ int | `format!("{:b}", a)` | none |
| `to_octal` | `to_octal(a: T) -> String`, T ∈ int | `format!("{:o}", a)` | none |
| `floor` | `floor(a: f64) -> f64` | `f64::floor` | none |
| `ceil` | `ceil(a: f64) -> f64` | `f64::ceil` | none |
| `round` | `round(a: f64) -> f64` | `f64::round` | none |
| `trunc` | `trunc(a: f64) -> f64` | `f64::trunc` | none |
| `fract` | `fract(a: f64) -> f64` | `f64::fract` | none |
| `sqrt` | `sqrt(a: f64) -> f64` | `f64::sqrt` | none |
| `cbrt` | `cbrt(a: f64) -> f64` | `f64::cbrt` | none |
| `exp` | `exp(a: f64) -> f64` | `f64::exp` | none |
| `exp2` | `exp2(a: f64) -> f64` | `f64::exp2` | none |
| `ln` | `ln(a: f64) -> f64` | `f64::ln` | none |
| `log10` | `log10(a: f64) -> f64` | `f64::log10` | none |
| `log2` | `log2(a: f64) -> f64` | `f64::log2` | none |
| `sin` | `sin(a: f64) -> f64` | `f64::sin` | none |
| `cos` | `cos(a: f64) -> f64` | `f64::cos` | none |
| `tan` | `tan(a: f64) -> f64` | `f64::tan` | none |
| `asin` | `asin(a: f64) -> f64` | `f64::asin` | none |
| `acos` | `acos(a: f64) -> f64` | `f64::acos` | none |
| `atan` | `atan(a: f64) -> f64` | `f64::atan` | none |
| `sinh` | `sinh(a: f64) -> f64` | `f64::sinh` | none |
| `cosh` | `cosh(a: f64) -> f64` | `f64::cosh` | none |
| `tanh` | `tanh(a: f64) -> f64` | `f64::tanh` | none |
| `to_degrees` | `to_degrees(a: f64) -> f64` | `f64::to_degrees` | none |
| `to_radians` | `to_radians(a: f64) -> f64` | `f64::to_radians` | none |
| `recip` | `recip(a: f64) -> f64` | `f64::recip` | none |
| `powf` | `powf(a: f64, n: f64) -> f64` | `f64::powf` | none |
| `log` | `log(a: f64, base: f64) -> f64` | `f64::log` | none |
| `atan2` | `atan2(a: f64, b: f64) -> f64` | `f64::atan2` | none |
| `hypot` | `hypot(a: f64, b: f64) -> f64` | `f64::hypot` | none |
| `copysign` | `copysign(a: f64, sign: f64) -> f64` | `f64::copysign` | none |
| `powi` | `powi(a: f64, n: i32) -> f64` | `f64::powi` | none |
| `mul_add` | `mul_add(a: f64, b: f64, c: f64) -> f64` | `f64::mul_add` | none |
| `is_nan` | `is_nan(a: f64) -> bool` | `f64::is_nan` | none |
| `is_finite` | `is_finite(a: f64) -> bool` | `f64::is_finite` | none |
| `is_infinite` | `is_infinite(a: f64) -> bool` | `f64::is_infinite` | none |
| `is_normal` | `is_normal(a: f64) -> bool` | `f64::is_normal` | none |
| `is_sign_negative` | `is_sign_negative(a: f64) -> bool` | `f64::is_sign_negative` | none |
| `is_sign_positive` | `is_sign_positive(a: f64) -> bool` | `f64::is_sign_positive` | none |
| `total_cmp` | `total_cmp(a: f64, b: f64) -> i64` | `f64::total_cmp` | the `Ordering` arrives as `-1`, `0`, `1`; the language has no `Ordering` |
| `to_bits` | `to_bits(a: f64) -> u64` | `f64::to_bits` | none |
| `<int>::MIN` | `<int>::MIN() -> T`, one under each of the eight integer namespaces | `T::MIN` | an associated constant in Rust; the language has none, so it is a zero-argument function |
| `<int>::MAX` | `<int>::MAX() -> T`, one under each of the eight integer namespaces | `T::MAX` | an associated constant in Rust; the language has none, so it is a zero-argument function |
| `<int>::BITS` | `<int>::BITS() -> u32`, one under each of the eight integer namespaces | `T::BITS` | an associated constant in Rust; the language has none, so it is a zero-argument function |
| `f64::MIN` | `f64::MIN() -> f64` | `f64::MIN` | an associated constant in Rust; the language has none, so it is a zero-argument function |
| `f64::MAX` | `f64::MAX() -> f64` | `f64::MAX` | an associated constant in Rust; the language has none, so it is a zero-argument function |
| `f64::EPSILON` | `f64::EPSILON() -> f64` | `f64::EPSILON` | an associated constant in Rust; the language has none, so it is a zero-argument function |
| `f64::INFINITY` | `f64::INFINITY() -> f64` | `f64::INFINITY` | an associated constant in Rust; the language has none, so it is a zero-argument function |
| `f64::NEG_INFINITY` | `f64::NEG_INFINITY() -> f64` | `f64::NEG_INFINITY` | an associated constant in Rust; the language has none, so it is a zero-argument function |
| `f64::NAN` | `f64::NAN() -> f64` | `f64::NAN` | an associated constant in Rust; the language has none, so it is a zero-argument function |
| `f64::from_bits` | `f64::from_bits(bits: u64) -> f64` | `f64::from_bits` | none |
| `<int>::from_str_radix` | `<int>::from_str_radix(text: &str, radix: u32) -> Result<T, ParseIntError>`, one under each of the eight integer namespaces | `T::from_str_radix` | a radix outside `2..=36` traps, as Rust's own assertion does |
