//! Numbers under Rust `std`'s names and contracts.
//!
//! Every name here is the `std` method of the same name, with the same
//! result. A method Rust writes over many integer widths is a shared
//! signature (RFC-0019) with an instance per width, so `x.checked_add(y)`
//! is one name whatever `x` is; a method Rust puts only on `f64` is a
//! plain function. `MIN`, `MAX` and `BITS` are associated constants in
//! Rust and the language has no such thing, so each is a zero-argument
//! function under its type's namespace: `i64::MAX()`, `f64::NAN()`.
//!
//! Three `std` shapes do not reach here. `overflowing_*` returns
//! `(T, bool)` and a Rust tuple does not cross the boundary, so it is not
//! declared. `abs_diff` returns the unsigned counterpart of its argument,
//! which one type variable cannot say, so its result is `u64` at every
//! width — the same value Rust computes, widened. `next_power_of_two`
//! panics on overflow here at every build profile, where Rust panics only
//! in debug.

use std::cmp::Ordering;
use std::fmt;

use acvus_extern::{Registry, Runtime, extern_fn, extern_registry};

pub mod sig {
    use acvus_extern::extern_signature;

    macro_rules! same_type_sigs {
        ($($name:ident($($arg:ident),+)),* $(,)?) => {$(
            extern_signature! {
                ns: "num",
                fn $name<T>($($arg: T),+) -> T
                where
                    T: Var<kind::Type>;
            }
        )*};
    }

    same_type_sigs! {
        abs(a),
        signum(a),
        min(a, b),
        max(a, b),
        clamp(x, lo, hi),
        pow(base, exp),
        wrapping_add(a, b),
        wrapping_sub(a, b),
        wrapping_mul(a, b),
        wrapping_div(a, b),
        wrapping_rem(a, b),
        wrapping_neg(a),
        saturating_add(a, b),
        saturating_sub(a, b),
        saturating_mul(a, b),
        saturating_div(a, b),
        saturating_neg(a),
        div_euclid(a, b),
        rem_euclid(a, b),
        swap_bytes(a),
        to_be(a),
        to_le(a),
        isqrt(a),
        next_power_of_two(a),
    }

    macro_rules! checked_sigs {
        ($($name:ident($($arg:ident),+)),* $(,)?) => {$(
            extern_signature! {
                ns: "num",
                fn $name<T>($($arg: T),+) -> Option<T>
                where
                    T: Var<kind::Type>;
            }
        )*};
    }

    checked_sigs! {
        checked_add(a, b),
        checked_sub(a, b),
        checked_mul(a, b),
        checked_div(a, b),
        checked_rem(a, b),
        checked_neg(a),
    }

    macro_rules! bit_count_sigs {
        ($($name:ident),* $(,)?) => {$(
            extern_signature! {
                ns: "num",
                fn $name<T>(a: T) -> u32
                where
                    T: Var<kind::Type>;
            }
        )*};
    }

    bit_count_sigs! {
        leading_zeros,
        trailing_zeros,
        count_ones,
    }

    macro_rules! radix_text_sigs {
        ($($name:ident),* $(,)?) => {$(
            extern_signature! {
                ns: "num",
                fn $name<T>(a: T) -> String
                where
                    T: Var<kind::Type>;
            }
        )*};
    }

    radix_text_sigs! {
        to_hex,
        to_binary,
        to_octal,
    }

    extern_signature! {
        ns: "num",
        fn checked_pow<T>(base: T, exp: u32) -> Option<T>
        where
            T: Var<kind::Type>;
    }

    extern_signature! {
        ns: "num",
        fn wrapping_pow<T>(base: T, exp: u32) -> T
        where
            T: Var<kind::Type>;
    }

    extern_signature! {
        ns: "num",
        fn saturating_pow<T>(base: T, exp: u32) -> T
        where
            T: Var<kind::Type>;
    }

    extern_signature! {
        ns: "num",
        fn abs_diff<T>(a: T, b: T) -> u64
        where
            T: Var<kind::Type>;
    }

    extern_signature! {
        ns: "num",
        fn is_power_of_two<T>(a: T) -> bool
        where
            T: Var<kind::Type>;
    }
}

/// `wrapping_pow` beyond the `u32` exponent Rust's own signature admits:
/// the same square-and-multiply, over the base's own width as the
/// exponent. An even base reaches `0` on its own here, since the product's
/// factor of two outruns the width long before the exponent is spent.
macro_rules! wrapping_pow_wide {
    ($t:ty, $base:expr, $exp:expr) => {{
        let mut acc: $t = 1;
        let mut square: $t = $base;
        let mut rest: $t = $exp;
        while rest > 0 {
            if rest & 1 == 1 {
                acc = acc.wrapping_mul(square);
            }
            square = square.wrapping_mul(square);
            rest >>= 1;
        }
        acc
    }};
}

/// `format!` builds its `Arguments` on the stack and hands their address to
/// the formatting machinery. An `Op::run` that inlines a handler doing that
/// holds a stack address across a callee, and LLVM's sibling-call rule then
/// refuses the tail call `acvus-interpreter-test/benches/asm_probe.rs`
/// asserts every operation ends with. Keeping the format in a function of
/// its own leaves the address in that function's frame; the `inline(never)`
/// is that contract's half here, and the bench is the other.
#[inline(never)]
fn hex_text(a: impl fmt::LowerHex) -> String {
    format!("{a:x}")
}

#[inline(never)]
fn binary_text(a: impl fmt::Binary) -> String {
    format!("{a:b}")
}

#[inline(never)]
fn octal_text(a: impl fmt::Octal) -> String {
    format!("{a:o}")
}

/// The operations every integer width carries, under `num::`'s shared
/// signatures. Each width gets its own Rust module, so a handler is named
/// exactly like the `std` method it calls.
macro_rules! int_common {
    ($t:ident) => {
        use acvus_extern::extern_fn;

        // -- checked ----------------------------------------------------

        #[extern_fn(instance_of = crate::num::sig::checked_add, effect = pure)]
        pub fn checked_add(a: $t, b: $t) -> Option<$t> {
            a.checked_add(b)
        }

        #[extern_fn(instance_of = crate::num::sig::checked_sub, effect = pure)]
        pub fn checked_sub(a: $t, b: $t) -> Option<$t> {
            a.checked_sub(b)
        }

        #[extern_fn(instance_of = crate::num::sig::checked_mul, effect = pure)]
        pub fn checked_mul(a: $t, b: $t) -> Option<$t> {
            a.checked_mul(b)
        }

        #[extern_fn(instance_of = crate::num::sig::checked_div, effect = pure)]
        pub fn checked_div(a: $t, b: $t) -> Option<$t> {
            a.checked_div(b)
        }

        #[extern_fn(instance_of = crate::num::sig::checked_rem, effect = pure)]
        pub fn checked_rem(a: $t, b: $t) -> Option<$t> {
            a.checked_rem(b)
        }

        #[extern_fn(instance_of = crate::num::sig::checked_neg, effect = pure)]
        pub fn checked_neg(a: $t) -> Option<$t> {
            a.checked_neg()
        }

        #[extern_fn(instance_of = crate::num::sig::checked_pow, effect = pure)]
        pub fn checked_pow(base: $t, exp: u32) -> Option<$t> {
            base.checked_pow(exp)
        }

        // -- wrapping ---------------------------------------------------

        #[extern_fn(
            instance_of = crate::num::sig::wrapping_add,
            effect = pure,
            law(associative, commutative, identity = 0)
        )]
        pub fn wrapping_add(a: $t, b: $t) -> $t {
            a.wrapping_add(b)
        }

        #[extern_fn(instance_of = crate::num::sig::wrapping_sub, effect = pure)]
        pub fn wrapping_sub(a: $t, b: $t) -> $t {
            a.wrapping_sub(b)
        }

        #[extern_fn(
            instance_of = crate::num::sig::wrapping_mul,
            effect = pure,
            law(associative, commutative, identity = 1)
        )]
        pub fn wrapping_mul(a: $t, b: $t) -> $t {
            a.wrapping_mul(b)
        }

        #[extern_fn(instance_of = crate::num::sig::wrapping_div, effect = pure)]
        pub fn wrapping_div(a: $t, b: $t) -> $t {
            assert!(b != 0, "wrapping_div: divisor is zero");
            a.wrapping_div(b)
        }

        #[extern_fn(instance_of = crate::num::sig::wrapping_rem, effect = pure)]
        pub fn wrapping_rem(a: $t, b: $t) -> $t {
            assert!(b != 0, "wrapping_rem: divisor is zero");
            a.wrapping_rem(b)
        }

        #[extern_fn(instance_of = crate::num::sig::wrapping_neg, effect = pure)]
        pub fn wrapping_neg(a: $t) -> $t {
            a.wrapping_neg()
        }

        #[extern_fn(instance_of = crate::num::sig::wrapping_pow, effect = pure)]
        pub fn wrapping_pow(base: $t, exp: u32) -> $t {
            base.wrapping_pow(exp)
        }

        // -- saturating -------------------------------------------------

        #[extern_fn(instance_of = crate::num::sig::saturating_add, effect = pure)]
        pub fn saturating_add(a: $t, b: $t) -> $t {
            a.saturating_add(b)
        }

        #[extern_fn(instance_of = crate::num::sig::saturating_sub, effect = pure)]
        pub fn saturating_sub(a: $t, b: $t) -> $t {
            a.saturating_sub(b)
        }

        #[extern_fn(instance_of = crate::num::sig::saturating_mul, effect = pure)]
        pub fn saturating_mul(a: $t, b: $t) -> $t {
            a.saturating_mul(b)
        }

        #[extern_fn(instance_of = crate::num::sig::saturating_div, effect = pure)]
        pub fn saturating_div(a: $t, b: $t) -> $t {
            assert!(b != 0, "saturating_div: divisor is zero");
            a.saturating_div(b)
        }

        #[extern_fn(instance_of = crate::num::sig::saturating_pow, effect = pure)]
        pub fn saturating_pow(base: $t, exp: u32) -> $t {
            base.saturating_pow(exp)
        }

        // -- euclidean division -----------------------------------------

        #[extern_fn(instance_of = crate::num::sig::div_euclid, effect = pure)]
        pub fn div_euclid(a: $t, b: $t) -> $t {
            assert!(b != 0, "div_euclid: divisor is zero");
            a.wrapping_div_euclid(b)
        }

        #[extern_fn(instance_of = crate::num::sig::rem_euclid, effect = pure)]
        pub fn rem_euclid(a: $t, b: $t) -> $t {
            assert!(b != 0, "rem_euclid: divisor is zero");
            a.wrapping_rem_euclid(b)
        }

        // -- bits -------------------------------------------------------

        #[extern_fn(instance_of = crate::num::sig::leading_zeros, effect = pure)]
        pub fn leading_zeros(a: $t) -> u32 {
            a.leading_zeros()
        }

        #[extern_fn(instance_of = crate::num::sig::trailing_zeros, effect = pure)]
        pub fn trailing_zeros(a: $t) -> u32 {
            a.trailing_zeros()
        }

        #[extern_fn(instance_of = crate::num::sig::count_ones, effect = pure)]
        pub fn count_ones(a: $t) -> u32 {
            a.count_ones()
        }

        #[extern_fn(instance_of = crate::num::sig::swap_bytes, effect = pure)]
        pub fn swap_bytes(a: $t) -> $t {
            a.swap_bytes()
        }

        #[extern_fn(instance_of = crate::num::sig::to_be, effect = pure)]
        pub fn to_be(a: $t) -> $t {
            a.to_be()
        }

        #[extern_fn(instance_of = crate::num::sig::to_le, effect = pure)]
        pub fn to_le(a: $t) -> $t {
            a.to_le()
        }

        // -- ordering ---------------------------------------------------

        #[extern_fn(
            instance_of = crate::num::sig::min,
            effect = pure,
            law(associative, commutative, identity = $t::MAX)
        )]
        pub fn min(a: $t, b: $t) -> $t {
            a.min(b)
        }

        #[extern_fn(
            instance_of = crate::num::sig::max,
            effect = pure,
            law(associative, commutative, identity = $t::MIN)
        )]
        pub fn max(a: $t, b: $t) -> $t {
            a.max(b)
        }

        #[extern_fn(instance_of = crate::num::sig::clamp, effect = pure)]
        pub fn clamp(x: $t, lo: $t, hi: $t) -> $t {
            assert!(
                lo <= hi,
                "clamp: lower bound {lo} is above upper bound {hi}"
            );
            x.clamp(lo, hi)
        }

        #[extern_fn(instance_of = crate::num::sig::abs_diff, effect = pure)]
        pub fn abs_diff(a: $t, b: $t) -> u64 {
            u64::from(a.abs_diff(b))
        }

        // -- text -------------------------------------------------------

        #[extern_fn(instance_of = crate::num::sig::to_hex, effect = pure)]
        pub fn to_hex(a: $t) -> String {
            crate::num::hex_text(a)
        }

        #[extern_fn(instance_of = crate::num::sig::to_binary, effect = pure)]
        pub fn to_binary(a: $t) -> String {
            crate::num::binary_text(a)
        }

        #[extern_fn(instance_of = crate::num::sig::to_octal, effect = pure)]
        pub fn to_octal(a: $t) -> String {
            crate::num::octal_text(a)
        }
    };
}

macro_rules! pow_at_width {
    ($t:ty, $base:expr, $exp:expr) => {{
        let Ok(exp32) = u32::try_from($exp) else {
            return wrapping_pow_wide!($t, $base, $exp);
        };
        $base.wrapping_pow(exp32)
    }};
}

/// `num::pow` at one signed width. Rust's exponent is a `u32` and the
/// signature's is the base's own type, so a negative exponent is a trap
/// and an exponent past `u32::MAX` runs the wide square-and-multiply.
macro_rules! int_pow_signed {
    ($t:ty) => {
        #[extern_fn(instance_of = crate::num::sig::pow, effect = pure)]
        pub fn pow(base: $t, exp: $t) -> $t {
            assert!(exp >= 0, "pow: negative exponent {exp} on an integer base");
            pow_at_width!($t, base, exp)
        }
    };
}

/// `num::pow` at an unsigned width inside `u32`, where every exponent the
/// type can hold is a `u32` and there is no wide path to take.
macro_rules! int_pow_narrow {
    ($t:ty) => {
        #[extern_fn(instance_of = crate::num::sig::pow, effect = pure)]
        pub fn pow(base: $t, exp: $t) -> $t {
            base.wrapping_pow(u32::from(exp))
        }
    };
}

macro_rules! int_pow_unsigned {
    ($t:ty) => {
        #[extern_fn(instance_of = crate::num::sig::pow, effect = pure)]
        pub fn pow(base: $t, exp: $t) -> $t {
            pow_at_width!($t, base, exp)
        }
    };
}

macro_rules! int_signed {
    () => {
        #[extern_fn(instance_of = crate::num::sig::abs, effect = pure)]
        pub fn abs(a: Width) -> Width {
            a.wrapping_abs()
        }

        #[extern_fn(instance_of = crate::num::sig::signum, effect = pure)]
        pub fn signum(a: Width) -> Width {
            a.signum()
        }

        #[extern_fn(instance_of = crate::num::sig::saturating_neg, effect = pure)]
        pub fn saturating_neg(a: Width) -> Width {
            a.saturating_neg()
        }

        #[extern_fn(instance_of = crate::num::sig::isqrt, effect = pure)]
        pub fn isqrt(a: Width) -> Width {
            assert!(a >= 0, "isqrt: negative argument {a}");
            a.isqrt()
        }
    };
}

macro_rules! int_unsigned {
    () => {
        #[extern_fn(instance_of = crate::num::sig::isqrt, effect = pure)]
        pub fn isqrt(a: Width) -> Width {
            a.isqrt()
        }

        #[extern_fn(instance_of = crate::num::sig::is_power_of_two, effect = pure)]
        pub fn is_power_of_two(a: Width) -> bool {
            a.is_power_of_two()
        }

        /// Rust's `next_power_of_two` panics on overflow in a debug build
        /// and wraps to `0` in a release one. A script's behaviour may not
        /// depend on how the host was compiled, so this traps at both.
        #[extern_fn(instance_of = crate::num::sig::next_power_of_two, effect = pure)]
        pub fn next_power_of_two(a: Width) -> Width {
            let Some(next) = a.checked_next_power_of_two() else {
                panic!("next_power_of_two: {a} has no next power of two in its width")
            };
            next
        }
    };
}

/// One width, one registry. An instance's own name is its Rust name, so
/// eight handlers all called `checked_add` need eight registries to be
/// eight declarations; the signature they instantiate collects them into
/// one language name when the registries combine (RFC-0019).
macro_rules! signed_width {
    ($m:ident: $t:ident) => {
        pub mod $m {
            use acvus_extern::{Registry, Runtime, extern_registry};

            type Width = $t;
            int_common!($t);
            int_pow_signed!($t);
            int_signed!();

            pub fn registry<R>() -> Registry<R>
            where
                R: Runtime,
            {
                extern_registry! {
                    ns: "num",
                    fns: [
                    checked_add, checked_sub, checked_mul,
                    checked_div, checked_rem, checked_neg, checked_pow,
                    wrapping_add, wrapping_sub, wrapping_mul,
                    wrapping_div, wrapping_rem, wrapping_neg, wrapping_pow,
                    saturating_add, saturating_sub, saturating_mul,
                    saturating_div, saturating_pow,
                    div_euclid, rem_euclid,
                    leading_zeros, trailing_zeros, count_ones,
                    swap_bytes, to_be, to_le,
                    min, max, clamp, abs_diff,
                    to_hex, to_binary, to_octal,
                    pow,
                        abs, signum, saturating_neg, isqrt,
                    ],
                }
            }
        }
    };
}

macro_rules! unsigned_width {
    ($m:ident: $t:ident, pow = $pow:ident) => {
        pub mod $m {
            use acvus_extern::{Registry, Runtime, extern_registry};

            type Width = $t;
            int_common!($t);
            $pow!($t);
            int_unsigned!();

            pub fn registry<R>() -> Registry<R>
            where
                R: Runtime,
            {
                extern_registry! {
                    ns: "num",
                    fns: [
                    checked_add, checked_sub, checked_mul,
                    checked_div, checked_rem, checked_neg, checked_pow,
                    wrapping_add, wrapping_sub, wrapping_mul,
                    wrapping_div, wrapping_rem, wrapping_neg, wrapping_pow,
                    saturating_add, saturating_sub, saturating_mul,
                    saturating_div, saturating_pow,
                    div_euclid, rem_euclid,
                    leading_zeros, trailing_zeros, count_ones,
                    swap_bytes, to_be, to_le,
                    min, max, clamp, abs_diff,
                    to_hex, to_binary, to_octal,
                    pow,
                        isqrt, is_power_of_two, next_power_of_two,
                    ],
                }
            }
        }
    };
}

signed_width!(i8s: i8);
signed_width!(i16s: i16);
signed_width!(i32s: i32);
signed_width!(i64s: i64);
unsigned_width!(u8s: u8, pow = int_pow_narrow);
unsigned_width!(u16s: u16, pow = int_pow_narrow);
unsigned_width!(u32s: u32, pow = int_pow_narrow);
unsigned_width!(u64s: u64, pow = int_pow_unsigned);

// -- f64 ----------------------------------------------------------------

#[extern_fn(instance_of = sig::abs, effect = pure)]
fn abs_float(a: f64) -> f64 {
    a.abs()
}

#[extern_fn(instance_of = sig::signum, effect = pure)]
fn signum_float(a: f64) -> f64 {
    a.signum()
}

#[extern_fn(instance_of = sig::min, effect = pure)]
fn min_float(a: f64, b: f64) -> f64 {
    a.min(b)
}

#[extern_fn(instance_of = sig::max, effect = pure)]
fn max_float(a: f64, b: f64) -> f64 {
    a.max(b)
}

#[extern_fn(instance_of = sig::clamp, effect = pure)]
fn clamp_float(x: f64, lo: f64, hi: f64) -> f64 {
    let Some(Ordering::Less | Ordering::Equal) = lo.partial_cmp(&hi) else {
        panic!("clamp: bounds {lo} and {hi} are not ordered")
    };
    x.clamp(lo, hi)
}

#[extern_fn(instance_of = sig::pow, effect = pure)]
fn pow_float(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

macro_rules! float_unary {
    ($($name:ident),* $(,)?) => {$(
        #[extern_fn(effect = pure)]
        fn $name(a: f64) -> f64 {
            a.$name()
        }
    )*};
}

float_unary! {
    floor, ceil, round, trunc, fract, sqrt, cbrt, exp, exp2, ln, log10, log2,
    sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
    to_degrees, to_radians, recip,
}

macro_rules! float_binary {
    ($($name:ident),* $(,)?) => {$(
        #[extern_fn(effect = pure)]
        fn $name(a: f64, b: f64) -> f64 {
            a.$name(b)
        }
    )*};
}

float_binary! {
    powf, log, atan2, hypot, copysign,
}

macro_rules! float_predicate {
    ($($name:ident),* $(,)?) => {$(
        #[extern_fn(effect = pure)]
        fn $name(a: f64) -> bool {
            a.$name()
        }
    )*};
}

float_predicate! {
    is_nan, is_finite, is_infinite, is_normal, is_sign_negative, is_sign_positive,
}

#[extern_fn(effect = pure)]
fn powi(a: f64, n: i32) -> f64 {
    a.powi(n)
}

#[extern_fn(effect = pure)]
fn mul_add(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// `f64::total_cmp` as the three values an `Ordering` has: Rust's total
/// order over every `f64`, `NaN` included, where `==` is a partial one.
#[extern_fn(effect = pure)]
fn total_cmp(a: f64, b: f64) -> i64 {
    match a.total_cmp(&b) {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

#[extern_fn(effect = pure)]
fn to_bits(a: f64) -> u64 {
    a.to_bits()
}

// -- the associated constants -------------------------------------------

macro_rules! int_constants {
    ($m:ident: $t:ty, ns = $ns:literal) => {
        mod $m {
            use acvus_extern::{Registry, Runtime, extern_fn, extern_registry};

            #[extern_fn(name = "MIN", effect = pure)]
            fn min_value() -> $t {
                <$t>::MIN
            }

            #[extern_fn(name = "MAX", effect = pure)]
            fn max_value() -> $t {
                <$t>::MAX
            }

            #[extern_fn(name = "BITS", effect = pure)]
            fn bits() -> u32 {
                <$t>::BITS
            }

            pub fn registry<R>() -> Registry<R>
            where
                R: Runtime,
            {
                extern_registry! {
                    ns: $ns,
                    fns: [min_value, max_value, bits],
                }
            }
        }
    };
}

int_constants!(i8_consts: i8, ns = "i8");
int_constants!(i16_consts: i16, ns = "i16");
int_constants!(i32_consts: i32, ns = "i32");
int_constants!(i64_consts: i64, ns = "i64");
int_constants!(u8_consts: u8, ns = "u8");
int_constants!(u16_consts: u16, ns = "u16");
int_constants!(u32_consts: u32, ns = "u32");
int_constants!(u64_consts: u64, ns = "u64");

mod f64_consts {
    use acvus_extern::{Registry, Runtime, extern_fn, extern_registry};

    #[extern_fn(name = "MIN", effect = pure)]
    fn min_value() -> f64 {
        f64::MIN
    }

    #[extern_fn(name = "MAX", effect = pure)]
    fn max_value() -> f64 {
        f64::MAX
    }

    #[extern_fn(name = "EPSILON", effect = pure)]
    fn epsilon() -> f64 {
        f64::EPSILON
    }

    #[extern_fn(name = "INFINITY", effect = pure)]
    fn infinity() -> f64 {
        f64::INFINITY
    }

    #[extern_fn(name = "NEG_INFINITY", effect = pure)]
    fn neg_infinity() -> f64 {
        f64::NEG_INFINITY
    }

    #[extern_fn(name = "NAN", effect = pure)]
    fn nan() -> f64 {
        f64::NAN
    }

    #[extern_fn(effect = pure)]
    fn from_bits(bits: u64) -> f64 {
        f64::from_bits(bits)
    }

    pub fn registry<R>() -> Registry<R>
    where
        R: Runtime,
    {
        extern_registry! {
            ns: "f64",
            fns: [min_value, max_value, epsilon, infinity, neg_infinity, nan, from_bits],
        }
    }
}

pub fn num_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "num",
        signatures: [
            sig::abs, sig::signum, sig::min, sig::max, sig::clamp, sig::pow,
            sig::wrapping_add, sig::wrapping_sub, sig::wrapping_mul,
            sig::wrapping_div, sig::wrapping_rem, sig::wrapping_neg,
            sig::wrapping_pow,
            sig::saturating_add, sig::saturating_sub, sig::saturating_mul,
            sig::saturating_div, sig::saturating_neg, sig::saturating_pow,
            sig::checked_add, sig::checked_sub, sig::checked_mul,
            sig::checked_div, sig::checked_rem, sig::checked_neg,
            sig::checked_pow,
            sig::div_euclid, sig::rem_euclid,
            sig::leading_zeros, sig::trailing_zeros, sig::count_ones,
            sig::swap_bytes, sig::to_be, sig::to_le,
            sig::isqrt, sig::is_power_of_two, sig::next_power_of_two,
            sig::abs_diff,
            sig::to_hex, sig::to_binary, sig::to_octal,
        ],
        fns: [
            abs_float, signum_float, min_float, max_float, clamp_float, pow_float,
            floor, ceil, round, trunc, fract, sqrt, cbrt, exp, exp2, ln, log10, log2,
            sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
            to_degrees, to_radians, recip,
            powf, log, atan2, hypot, copysign,
            is_nan, is_finite, is_infinite, is_normal,
            is_sign_negative, is_sign_positive,
            powi, mul_add, total_cmp, to_bits,
        ],
    }
}

/// One registry per integer width, beside the signatures `num_registry`
/// declares.
pub fn num_width_registries<R>() -> Vec<Registry<R>>
where
    R: Runtime,
{
    vec![
        i8s::registry(),
        i16s::registry(),
        i32s::registry(),
        i64s::registry(),
        u8s::registry(),
        u16s::registry(),
        u32s::registry(),
        u64s::registry(),
    ]
}

/// `MIN`, `MAX` and `BITS` live under their type's own namespace, one
/// registry each, because a zero-argument function has no argument for a
/// shared signature to resolve on.
pub fn num_constant_registries<R>() -> Vec<Registry<R>>
where
    R: Runtime,
{
    vec![
        i8_consts::registry(),
        i16_consts::registry(),
        i32_consts::registry(),
        i64_consts::registry(),
        u8_consts::registry(),
        u16_consts::registry(),
        u32_consts::registry(),
        u64_consts::registry(),
        f64_consts::registry(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{Externs, Interner, TypesOnly};

    /// The names `docs/std/num.md` tabulates. Its table folds the eight
    /// integer namespaces of `MIN`, `MAX` and `BITS` into one row each
    /// and is otherwise a row per name, so a name added here without a
    /// row there moves this count.
    #[test]
    fn the_module_offers_the_names_the_table_lists() {
        let i = Interner::new();
        let mut registries = vec![num_registry::<TypesOnly>()];
        registries.extend(num_width_registries());
        registries.extend(num_constant_registries());
        let reg = Externs::combine(registries, &i).expect("registries combine");
        let core = Externs::<TypesOnly>::combine(vec![], &i).expect("core combines");

        let shared_signatures = 41;
        let float_fns = 39;
        let int_constants = 8 * 3;
        let float_constants = 6 + 1;
        assert_eq!(
            reg.functions.len() - core.functions.len(),
            shared_signatures + float_fns + int_constants + float_constants
        );
        assert_eq!(
            reg.handlers.len() - core.handlers.len(),
            shared_signatures + float_fns + int_constants + float_constants
        );
    }

    /// A fixed-seed linear congruential sequence (Knuth's MMIX constants),
    /// so a failing law names the same inputs on every run.
    struct Samples(u64);

    const SEED: u64 = 0x5eed_1a55_0c1a_7e00;
    const SAMPLED: usize = 64;

    impl Samples {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
    }

    struct Declared<T> {
        name: &'static str,
        f: fn(T, T) -> T,
        identity: T,
    }

    /// Each declared law of `min`, `max`, `wrapping_add` and `wrapping_mul`
    /// at one width, over the width's edges and a sample of its words.
    macro_rules! laws_hold_at {
        ($test:ident, $m:ident: $t:ident) => {
            #[test]
            fn $test() {
                let edges: [$t; 5] = [$t::MIN, $t::MAX, 0, 1, $t::MAX / 2];
                let mut samples = Samples(SEED);
                let words: Vec<$t> = edges
                    .into_iter()
                    .chain((0..SAMPLED).map(|_| samples.next() as $t))
                    .collect();
                let declared: [Declared<$t>; 4] = [
                    Declared { name: "min", f: $m::min, identity: $t::MAX },
                    Declared { name: "max", f: $m::max, identity: $t::MIN },
                    Declared { name: "wrapping_add", f: $m::wrapping_add, identity: 0 },
                    Declared { name: "wrapping_mul", f: $m::wrapping_mul, identity: 1 },
                ];
                for Declared { name, f, identity } in declared {
                    for &a in &words {
                        assert_eq!(f(a, identity), a, "{name}: identity on the right of {a}");
                        assert_eq!(f(identity, a), a, "{name}: identity on the left of {a}");
                        for &b in &words {
                            assert_eq!(f(a, b), f(b, a), "{name}: commutes at {a}, {b}");
                            for &c in words.iter().take(16) {
                                assert_eq!(
                                    f(f(a, b), c),
                                    f(a, f(b, c)),
                                    "{name}: associates at {a}, {b}, {c}"
                                );
                            }
                        }
                    }
                }
            }
        };
    }

    laws_hold_at!(declared_laws_hold_at_i8, i8s: i8);
    laws_hold_at!(declared_laws_hold_at_i16, i16s: i16);
    laws_hold_at!(declared_laws_hold_at_i32, i32s: i32);
    laws_hold_at!(declared_laws_hold_at_i64, i64s: i64);
    laws_hold_at!(declared_laws_hold_at_u8, u8s: u8);
    laws_hold_at!(declared_laws_hold_at_u16, u16s: u16);
    laws_hold_at!(declared_laws_hold_at_u32, u32s: u32);
    laws_hold_at!(declared_laws_hold_at_u64, u64s: u64);

    /// The identities the laws name resolve to the constants the property
    /// tests above sample against.
    #[test]
    fn min_and_max_name_their_width_s_bounds_as_identities() {
        use acvus_extern::{BinaryLaws, FnKind, Identity, Laws, QualifiedRef};
        let i = Interner::new();
        let mut registries = vec![num_registry::<TypesOnly>()];
        registries.extend(num_width_registries());
        registries.extend(num_constant_registries());
        let reg = Externs::combine(registries, &i).expect("registries combine");
        let named = |ns: &str, name: &str| QualifiedRef::qualified(i.intern(ns), i.intern(name));
        for (sig, bound) in [("min", "MAX"), ("max", "MIN")] {
            let function = reg
                .functions
                .iter()
                .find(|f| f.qref == named("num", sig))
                .expect("the signature is declared");
            let FnKind::Extern { instances, .. } = &function.kind else {
                panic!("{sig} is an extern")
            };
            let mut widths: Vec<&str> = Vec::new();
            for instance in &instances.concrete {
                let acvus_extern::PolyTy::Fn { ret, .. } = &instance.ty else {
                    panic!("an instance is a function")
                };
                let width = match &**ret {
                    acvus_extern::PolyTy::Int(width) => width.name(),
                    acvus_extern::PolyTy::Float => {
                        assert_eq!(instance.laws, Laws::None, "{sig} over f64 declares no law");
                        continue;
                    }
                    other => panic!("{sig} at {other:?}"),
                };
                assert_eq!(
                    instance.laws,
                    Laws::Binary(BinaryLaws {
                        associative: true,
                        commutative: true,
                        identity: Some(Identity::Extern(named(width, bound))),
                    }),
                    "{sig} at {width}"
                );
                widths.push(width);
            }
            assert_eq!(widths.len(), 8, "{sig}: {widths:?}");
        }
    }
}
