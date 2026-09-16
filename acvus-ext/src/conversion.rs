//! Type conversions. All pure.
//!
//! `core::to_string` and `core::to_int` are shared signatures (RFC-0019) with
//! one instance per scalar: Int, Float, Bool, Byte, String.

use acvus_extern::{Registry, Runtime, Trap, extern_fn, extern_registry};

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

macro_rules! to_string_ints {
    ($($name:ident: $t:ty),* $(,)?) => {$(
        #[extern_fn(instance_of = sig::to_string, effect = pure)]
        fn $name(a: &$t) -> String {
            a.to_string()
        }
    )*};
}

to_string_ints! {
    to_string_i8: i8, to_string_i16: i16, to_string_i32: i32, to_string_int: i64,
    to_string_u8: u8, to_string_u16: u16, to_string_u32: u32, to_string_u64: u64,
}

#[extern_fn(instance_of = sig::to_string, effect = pure)]
fn to_string_float(a: &f64) -> String {
    a.to_string()
}

#[extern_fn(instance_of = sig::to_string, effect = pure)]
fn to_string_bool(a: &bool) -> String {
    a.to_string()
}

#[extern_fn(instance_of = sig::to_string, effect = pure)]
fn to_string_string(a: &String) -> String {
    a.clone()
}

// -- to_int -------------------------------------------------------------

#[extern_fn(instance_of = sig::to_int, effect = pure)]
fn to_int_int(a: &i64) -> Result<i64, Trap> {
    Ok(*a)
}

#[extern_fn(instance_of = sig::to_int, effect = pure)]
fn to_int_float(a: &f64) -> Result<i64, Trap> {
    Ok(*a as i64)
}

#[extern_fn(instance_of = sig::to_int, effect = pure)]
fn to_int_bool(a: &bool) -> Result<i64, Trap> {
    Ok(i64::from(*a))
}

macro_rules! to_int_widens {
    ($($name:ident: $t:ty),* $(,)?) => {$(
        #[extern_fn(instance_of = sig::to_int, effect = pure)]
        fn $name(a: &$t) -> Result<i64, Trap> {
            Ok(i64::from(*a))
        }
    )*};
}

to_int_widens! {
    to_int_i8: i8, to_int_i16: i16, to_int_i32: i32,
    to_int_byte: u8, to_int_u16: u16, to_int_u32: u32,
}

#[extern_fn(instance_of = sig::to_int, effect = pure)]
fn to_int_u64(a: &u64) -> Result<i64, Trap> {
    i64::try_from(*a).map_err(|_| Trap::call("to_int", format!("{a} does not fit i64")))
}

#[extern_fn(instance_of = sig::to_int, effect = pure)]
fn to_int_string(a: &String) -> Result<i64, Trap> {
    a.parse::<i64>()
        .map_err(|e| Trap::call("to_int", format!("cannot parse string: {e}")))
}

// -- the rest -----------------------------------------------------------

#[extern_fn(effect = pure)]
fn to_float(n: i64) -> f64 {
    n as f64
}

#[extern_fn(effect = pure)]
fn char_to_int(s: String) -> Result<i64, Trap> {
    match s.chars().next() {
        Some(c) => Ok(c as i64),
        None => Err(Trap::call("char_to_int", "empty string")),
    }
}

#[extern_fn(effect = pure)]
fn int_to_char(n: i64) -> Result<String, Trap> {
    let code = u32::try_from(n)
        .map_err(|_| Trap::call("int_to_char", format!("{n} is not a code point")))?;
    match char::from_u32(code) {
        Some(c) => Ok(c.to_string()),
        None => Err(Trap::call(
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
            to_string_i8, to_string_i16, to_string_i32, to_string_int,
            to_string_u8, to_string_u16, to_string_u32, to_string_u64,
            to_string_float, to_string_bool, to_string_string,
            to_int_i8, to_int_i16, to_int_i32, to_int_int,
            to_int_byte, to_int_u16, to_int_u32, to_int_u64,
            to_int_float, to_int_bool, to_int_string,
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
