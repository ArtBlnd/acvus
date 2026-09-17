//! Numbers under Rust `std`'s names and contracts: the shared `num::`
//! signatures `abs`, `min`, `max`, `clamp`, `pow`, `signum` with an `i64`
//! and an `f64` instance each, and the `f64`-only functions.

use std::cmp::Ordering;

use acvus_extern::{Registry, Runtime, Trap, extern_fn, extern_registry};

pub mod sig {
    use acvus_extern::extern_signature;

    extern_signature! {
        ns: "num",
        fn abs<T>(a: T) -> T
        where
            T: TyVar;
    }

    extern_signature! {
        ns: "num",
        fn min<T>(a: T, b: T) -> T
        where
            T: TyVar;
    }

    extern_signature! {
        ns: "num",
        fn max<T>(a: T, b: T) -> T
        where
            T: TyVar;
    }

    extern_signature! {
        ns: "num",
        fn clamp<T>(x: T, lo: T, hi: T) -> T
        where
            T: TyVar;
    }

    extern_signature! {
        ns: "num",
        fn pow<T>(base: T, exp: T) -> T
        where
            T: TyVar;
    }

    extern_signature! {
        ns: "num",
        fn signum<T>(a: T) -> T
        where
            T: TyVar;
    }
}

// -- abs ----------------------------------------------------------------

#[extern_fn(instance_of = sig::abs, effect = pure)]
fn abs_int(a: i64) -> Result<i64, Trap> {
    a.checked_abs()
        .ok_or_else(|| Trap::call("abs", "the absolute value of i64::MIN does not fit in i64"))
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
fn clamp_int(x: i64, lo: i64, hi: i64) -> Result<i64, Trap> {
    if lo > hi {
        return Err(Trap::call(
            "clamp",
            format!("lower bound {lo} is above upper bound {hi}"),
        ));
    }
    Ok(x.clamp(lo, hi))
}

#[extern_fn(instance_of = sig::clamp, effect = pure)]
fn clamp_float(x: f64, lo: f64, hi: f64) -> Result<f64, Trap> {
    let Some(Ordering::Less | Ordering::Equal) = lo.partial_cmp(&hi) else {
        return Err(Trap::call(
            "clamp",
            format!("bounds {lo} and {hi} are not ordered"),
        ));
    };
    Ok(x.clamp(lo, hi))
}

// -- pow ----------------------------------------------------------------

#[extern_fn(instance_of = sig::pow, effect = pure)]
fn pow_int(base: i64, exp: i64) -> Result<i64, Trap> {
    if exp < 0 {
        return Err(Trap::call(
            "pow",
            format!("negative exponent {exp} on an integer base"),
        ));
    }
    let Ok(exp32) = u32::try_from(exp) else {
        return pow_with_exponent_beyond_u32(base, exp);
    };
    base.checked_pow(exp32)
        .ok_or_else(|| pow_overflow(base, exp))
}

fn pow_with_exponent_beyond_u32(base: i64, exp: i64) -> Result<i64, Trap> {
    match base {
        0 => Ok(0),
        1 => Ok(1),
        -1 if exp % 2 == 0 => Ok(1),
        -1 => Ok(-1),
        _ => Err(pow_overflow(base, exp)),
    }
}

fn pow_overflow(base: i64, exp: i64) -> Trap {
    Trap::call("pow", format!("{base}^{exp} overflows i64"))
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
