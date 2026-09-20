//! Numbers under Rust `std`'s names and contracts: the shared `num::`
//! signatures `abs`, `min`, `max`, `clamp`, `pow`, `signum` with an `i64`
//! and an `f64` instance each, and the `f64`-only functions.

use std::cmp::Ordering;

use acvus_extern::{Registry, Runtime, extern_fn, extern_registry};

pub mod sig {
    use acvus_extern::extern_signature;

    extern_signature! {
        ns: "num",
        fn abs<T>(a: T) -> T
        where
            T: Var<kind::Type>;
    }

    extern_signature! {
        ns: "num",
        fn min<T>(a: T, b: T) -> T
        where
            T: Var<kind::Type>;
    }

    extern_signature! {
        ns: "num",
        fn max<T>(a: T, b: T) -> T
        where
            T: Var<kind::Type>;
    }

    extern_signature! {
        ns: "num",
        fn clamp<T>(x: T, lo: T, hi: T) -> T
        where
            T: Var<kind::Type>;
    }

    extern_signature! {
        ns: "num",
        fn pow<T>(base: T, exp: T) -> T
        where
            T: Var<kind::Type>;
    }

    extern_signature! {
        ns: "num",
        fn signum<T>(a: T) -> T
        where
            T: Var<kind::Type>;
    }
}

// -- abs ----------------------------------------------------------------

#[extern_fn(instance_of = sig::abs, effect = pure)]
fn abs_int(a: i64) -> i64 {
    a.wrapping_abs()
}

#[extern_fn(instance_of = sig::abs, effect = pure)]
fn abs_float(a: f64) -> f64 {
    a.abs()
}

// -- min / max ----------------------------------------------------------

#[extern_fn(instance_of = sig::min, effect = pure)]
fn min_int(a: i64, b: i64) -> i64 {
    a.min(b)
}

#[extern_fn(instance_of = sig::min, effect = pure)]
fn min_float(a: f64, b: f64) -> f64 {
    a.min(b)
}

#[extern_fn(instance_of = sig::max, effect = pure)]
fn max_int(a: i64, b: i64) -> i64 {
    a.max(b)
}

#[extern_fn(instance_of = sig::max, effect = pure)]
fn max_float(a: f64, b: f64) -> f64 {
    a.max(b)
}

// -- clamp --------------------------------------------------------------

#[extern_fn(instance_of = sig::clamp, effect = pure)]
fn clamp_int(x: i64, lo: i64, hi: i64) -> i64 {
    assert!(
        lo <= hi,
        "clamp: lower bound {lo} is above upper bound {hi}"
    );
    x.clamp(lo, hi)
}

#[extern_fn(instance_of = sig::clamp, effect = pure)]
fn clamp_float(x: f64, lo: f64, hi: f64) -> f64 {
    let Some(Ordering::Less | Ordering::Equal) = lo.partial_cmp(&hi) else {
        panic!("clamp: bounds {lo} and {hi} are not ordered")
    };
    x.clamp(lo, hi)
}

// -- pow ----------------------------------------------------------------

#[extern_fn(instance_of = sig::pow, effect = pure)]
fn pow_int(base: i64, exp: i64) -> i64 {
    assert!(exp >= 0, "pow: negative exponent {exp} on an integer base");
    let Ok(exp32) = u32::try_from(exp) else {
        return wrapping_pow_wide(base, exp);
    };
    base.wrapping_pow(exp32)
}

/// `wrapping_pow` beyond the `u32` exponent Rust's own signature admits:
/// the same square-and-multiply, over an `i64` exponent. An even base
/// reaches `0` on its own here, since the product's factor of two outruns
/// the width long before the exponent is spent.
fn wrapping_pow_wide(base: i64, exp: i64) -> i64 {
    let mut acc: i64 = 1;
    let mut square = base;
    let mut rest = exp;
    while rest > 0 {
        if rest & 1 == 1 {
            acc = acc.wrapping_mul(square);
        }
        square = square.wrapping_mul(square);
        rest >>= 1;
    }
    acc
}

#[extern_fn(instance_of = sig::pow, effect = pure)]
fn pow_float(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

// -- signum -------------------------------------------------------------

#[extern_fn(instance_of = sig::signum, effect = pure)]
fn signum_int(a: i64) -> i64 {
    a.signum()
}

#[extern_fn(instance_of = sig::signum, effect = pure)]
fn signum_float(a: f64) -> f64 {
    a.signum()
}

// -- f64 only -----------------------------------------------------------

macro_rules! float_unary {
    ($($name:ident),* $(,)?) => {$(
        #[extern_fn(effect = pure)]
        fn $name(a: f64) -> f64 {
            a.$name()
        }
    )*};
}

float_unary! {
    floor, ceil, round, trunc, sqrt, cbrt, exp, ln, log10, log2, sin, cos, tan,
}

#[extern_fn(effect = pure)]
fn is_nan(a: f64) -> bool {
    a.is_nan()
}

#[extern_fn(effect = pure)]
fn is_finite(a: f64) -> bool {
    a.is_finite()
}

pub fn num_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "num",
        signatures: [sig::abs, sig::min, sig::max, sig::clamp, sig::pow, sig::signum],
        fns: [
            abs_int, abs_float,
            min_int, min_float,
            max_int, max_float,
            clamp_int, clamp_float,
            pow_int, pow_float,
            signum_int, signum_float,
            floor, ceil, round, trunc, sqrt, cbrt, exp, ln, log10, log2, sin, cos, tan,
            is_nan, is_finite,
        ],
    }
}
