//! Type conversions. All pure.
//!
//! `core::to_string` and `core::to_int` are shared signatures (RFC-0019).
//! `to_int` converts a `Bool` and nothing else: every number-to-number
//! conversion is `expr as T` in the language (RFC-0049), which is total,
//! reaches every width in both directions, and is a chain leaf rather than
//! a call. Parsing text is `i64::from_str(s)`, one `from_str` under each
//! integer type's namespace, and it returns a `Result` (RFC-0038).

use std::num::IntErrorKind;

use acvus_extern::{Registry, Runtime, TyArg, extern_fn, extern_registry};

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
fn to_int_bool(a: &bool) -> i64 {
    i64::from(*a)
}

// -- from_str -----------------------------------------------------------

#[derive(TyArg)]
pub enum ParseIntError {
    Invalid(String),
    OutOfRange(String),
}

fn parse_int_error(text: String, e: &std::num::ParseIntError) -> ParseIntError {
    match e.kind() {
        IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => ParseIntError::OutOfRange(text),
        _ => ParseIntError::Invalid(text),
    }
}

macro_rules! from_str_ints {
    ($($name:ident: $t:ty as $ns:literal => $registry:ident),* $(,)?) => {$(
        #[extern_fn(name = "from_str", effect = pure)]
        fn $name(text: String) -> Result<$t, ParseIntError> {
            text.parse::<$t>().map_err(|e| parse_int_error(text, &e))
        }

        fn $registry<R>() -> Registry<R>
        where
            R: Runtime,
        {
            extern_registry! {
                ns: $ns,
                fns: [$name],
            }
        }
    )*};
}

from_str_ints! {
    from_str_i8: i8 as "i8" => i8_registry,
    from_str_i16: i16 as "i16" => i16_registry,
    from_str_i32: i32 as "i32" => i32_registry,
    from_str_i64: i64 as "i64" => i64_registry,
    from_str_u8: u8 as "u8" => u8_registry,
    from_str_u16: u16 as "u16" => u16_registry,
    from_str_u32: u32 as "u32" => u32_registry,
    from_str_u64: u64 as "u64" => u64_registry,
}

pub fn from_str_registries<R>() -> Vec<Registry<R>>
where
    R: Runtime,
{
    vec![
        i8_registry(),
        i16_registry(),
        i32_registry(),
        i64_registry(),
        u8_registry(),
        u16_registry(),
        u32_registry(),
        u64_registry(),
    ]
}

// -- the rest -----------------------------------------------------------

#[derive(TyArg)]
pub enum CharError {
    NotOneChar(String),
    NotAChar(i64),
}

#[extern_fn(effect = pure)]
fn char_to_int(s: String) -> Result<i64, CharError> {
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Ok(i64::from(u32::from(c))),
        _ => Err(CharError::NotOneChar(s)),
    }
}

#[extern_fn(effect = pure)]
fn int_to_char(n: i64) -> Result<String, CharError> {
    let Ok(code) = u32::try_from(n) else {
        return Err(CharError::NotAChar(n));
    };
    match char::from_u32(code) {
        Some(c) => Ok(c.to_string()),
        None => Err(CharError::NotAChar(n)),
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
            to_int_bool,
            char_to_int, int_to_char,
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
        let plain_fns = 2;
        assert_eq!(
            reg.functions.len() - core.functions.len(),
            signatures + plain_fns
        );
        assert_eq!(
            reg.handlers.len() - core.handlers.len(),
            signatures + plain_fns
        );
    }

    #[test]
    fn every_integer_type_has_its_own_from_str() {
        let i = Interner::new();
        let reg =
            Externs::combine(from_str_registries::<TypesOnly>(), &i).expect("registries combine");
        let core = Externs::<TypesOnly>::combine(vec![], &i).expect("core combines");
        assert_eq!(reg.functions.len() - core.functions.len(), 8);
        assert_eq!(reg.handlers.len() - core.handlers.len(), 8);
        for ns in ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"] {
            let qref = acvus_extern::QualifiedRef::qualified(i.intern(ns), i.intern("from_str"));
            assert!(
                reg.handlers.contains_key(&qref),
                "{ns}::from_str is registered"
            );
        }
    }
}
