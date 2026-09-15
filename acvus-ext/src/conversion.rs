//! Type conversions. All pure.
//!
//! `core::to_string` and `core::to_int` are shared signatures (RFC-0019) with
//! one instance per scalar: Int, Float, Bool, Byte, String.

use acvus_extern::{ExternError, Registry, Runtime, extern_fn, extern_registry};

pub mod sig {
    use acvus_extern::extern_signature;

    extern_signature! {
        ns: "core",
        fn to_string<T>(a: &T) -> String
        where
            T: TyVar;
    }

    extern_signature! {
        ns: "core",
        fn to_int<T>(a: &T) -> i64
        where
            T: TyVar;
    }
}

// -- to_string ----------------------------------------------------------

#[extern_fn(instance_of = sig::to_string, effect = pure)]
fn to_string_int<R>(_: &R, a: &i64) -> String
where
    R: Runtime,
{
    a.to_string()
}

#[extern_fn(instance_of = sig::to_string, effect = pure)]
fn to_string_float<R>(_: &R, a: &f64) -> String
where
    R: Runtime,
{
    a.to_string()
}

#[extern_fn(instance_of = sig::to_string, effect = pure)]
fn to_string_bool<R>(_: &R, a: &bool) -> String
where
    R: Runtime,
{
    a.to_string()
}

#[extern_fn(instance_of = sig::to_string, effect = pure)]
fn to_string_byte<R>(_: &R, a: &u8) -> String
where
    R: Runtime,
{
    a.to_string()
}

#[extern_fn(instance_of = sig::to_string, effect = pure)]
fn to_string_string<R>(_: &R, a: &String) -> String
where
    R: Runtime,
{
    a.clone()
}

// -- to_int -------------------------------------------------------------

#[extern_fn(instance_of = sig::to_int, effect = pure)]
fn to_int_int<R>(_: &R, a: &i64) -> Result<i64, ExternError>
where
    R: Runtime,
{
    Ok(*a)
}

#[extern_fn(instance_of = sig::to_int, effect = pure)]
fn to_int_float<R>(_: &R, a: &f64) -> Result<i64, ExternError>
where
    R: Runtime,
{
    Ok(*a as i64)
}

#[extern_fn(instance_of = sig::to_int, effect = pure)]
fn to_int_bool<R>(_: &R, a: &bool) -> Result<i64, ExternError>
where
    R: Runtime,
{
    Ok(i64::from(*a))
}

#[extern_fn(instance_of = sig::to_int, effect = pure)]
fn to_int_byte<R>(_: &R, a: &u8) -> Result<i64, ExternError>
where
    R: Runtime,
{
    Ok(i64::from(*a))
}

#[extern_fn(instance_of = sig::to_int, effect = pure)]
fn to_int_string<R>(_: &R, a: &String) -> Result<i64, ExternError>
where
    R: Runtime,
{
    a.parse::<i64>()
        .map_err(|e| ExternError::call("to_int", format!("cannot parse string: {e}")))
}

// -- the rest -----------------------------------------------------------

#[extern_fn(effect = pure)]
fn to_float<R>(_: &R, n: i64) -> f64
where
    R: Runtime,
{
    n as f64
}

#[extern_fn(effect = pure)]
fn char_to_int<R>(_: &R, s: String) -> Result<i64, ExternError>
where
    R: Runtime,
{
    match s.chars().next() {
        Some(c) => Ok(c as i64),
        None => Err(ExternError::call("char_to_int", "empty string")),
    }
}

#[extern_fn(effect = pure)]
fn int_to_char<R>(_: &R, n: i64) -> Result<String, ExternError>
where
    R: Runtime,
{
    let code = u32::try_from(n)
        .map_err(|_| ExternError::call("int_to_char", format!("{n} is not a code point")))?;
    match char::from_u32(code) {
        Some(c) => Ok(c.to_string()),
        None => Err(ExternError::call(
            "int_to_char",
            format!("{n} is not a code point"),
        )),
    }
}

pub fn conversion_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        signatures: [sig::to_string, sig::to_int],
        fns: [
            to_string_int, to_string_float, to_string_bool, to_string_byte, to_string_string,
            to_int_int, to_int_float, to_int_bool, to_int_byte, to_int_string,
            to_float, char_to_int, int_to_char,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{Externs, Interner, TypesOnly};

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg = Externs::combine(vec![conversion_registry::<TypesOnly>()], &i)
            .expect("registry combines");
        let core = Externs::<TypesOnly>::combine(vec![], &i).expect("core combines");
        let signatures = 2;
        let plain_fns = 3;
        assert_eq!(
            reg.functions.len() - core.functions.len(),
            signatures + plain_fns
        );
        assert_eq!(
            reg.handlers.len() - core.handlers.len(),
            signatures + plain_fns
        );
    }
}
