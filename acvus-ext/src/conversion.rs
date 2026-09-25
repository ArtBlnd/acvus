//! Type conversions. All pure.
//!
//! `core::to_int` is a shared signature (RFC-0019); `core::display` is
//! declared beside the other core signatures in `acvus_extern::core` and
//! its instances at the language's own types are here, with the one
//! generic `to_string` that requires it (RFC-0070 rule 5). `display` has no
//! instance at `str` or at `String`: the owned copy of text, a `&str` or a
//! `String` lent as one, is `string::to_string`.
//! `to_int` converts a `Bool` and nothing else: every number-to-number
//! conversion is `expr as T` in the language (RFC-0049), which is total,
//! reaches every width in both directions, and is a chain leaf rather than
//! a call. Parsing text is `i64::from_str(s)`, one `from_str` under each
//! integer type's namespace, and it returns a `Result` (RFC-0038).
//!
//! Rust writes the same parse as `s.parse::<i64>()`. The language has no
//! turbofish, so the target type is written as the namespace instead and
//! `parse` is registered beside `from_str` under it: `i64::parse(s)` is
//! `i64::from_str(s)`, and both are what Rust's `FromStr for i64` does.

use std::num::IntErrorKind;

use std::ops::Deref;

use acvus_extern::{
    Borrowable, Ctx, InstanceOf, Registry, Runtime, TyArg, Var, core, extern_fn, extern_registry,
    kind,
};

pub mod sig {
    use acvus_extern::extern_signature;

    extern_signature! {
        ns: "core",
        fn to_int<T>(a: &T) -> i64
        where
            T: Var<kind::Type>;
    }
}

// -- display ------------------------------------------------------------

/// Appends `a`'s `Display` text to `out` in place, so a text built of many
/// parts grows the one buffer it is written into (RFC-0070 rule 5).
fn append(out: &mut String, a: &dyn std::fmt::Display) {
    use std::fmt::Write;
    // `String`'s `write_str` returns `Ok` on every path, and none of the
    // `Display` impls this module reaches returns an error of its own, so
    // the one failure `write!` can report does not arise here.
    write!(out, "{a}").expect("a String's fmt::Write does not fail");
}

macro_rules! display_ints {
    ($($name:ident: $t:ty),* $(,)?) => {$(
        #[extern_fn(instance_of = acvus_extern::core::display, effect = pure)]
        fn $name(a: &$t, out: &mut String) {
            append(out, a);
        }
    )*};
}

display_ints! {
    display_i8: i8, display_i16: i16, display_i32: i32, display_int: i64,
    display_u8: u8, display_u16: u16, display_u32: u32, display_u64: u64,
}

#[extern_fn(instance_of = acvus_extern::core::display, effect = pure)]
fn display_float(a: &f64, out: &mut String) {
    append(out, a);
}

#[extern_fn(instance_of = acvus_extern::core::display, effect = pure)]
fn display_char(a: &char, out: &mut String) {
    out.push(*a);
}

#[extern_fn(instance_of = acvus_extern::core::display, effect = pure)]
fn display_bool(a: &bool, out: &mut String) {
    append(out, a);
}

/// The standard registry's one `to_string`: the text `display` appends at
/// `T`, into an empty `String`. Every type with a `display` instance has it,
/// and a type's text has one source (RFC-0070 rule 5). Its `T` ranges over
/// the types `display` stands at, which hold neither `str` nor `String`, so
/// `"…".to_string()` and `s.to_string()` reach `string::to_string` alone
/// (RFC-0043).
#[extern_fn(effect = pure)]
fn to_string<T, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    a: &T,
    display: InstanceOf<'_, core::display<T, Rt>, T, Rt>,
) -> String
where
    T: Var<kind::Type> + Borrowable<Rt> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    let mut out = String::new();
    display.call(ctx, a, (&mut out,));
    out
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
    ($($from_str:ident / $parse:ident / $radix:ident: $t:ty as $ns:literal => $registry:ident),* $(,)?) => {$(
        /// `returns`: `str::parse` reads each byte of `text` once and
        /// answers `Ok` or `Err`, and `parse_int_error` only copies `text`.
        #[extern_fn(name = "from_str", effect = pure, returns)]
        fn $from_str(text: &str) -> Result<$t, ParseIntError> {
            text.parse::<$t>()
                .map_err(|e| parse_int_error(text.to_owned(), &e))
        }

        #[extern_fn(name = "parse", effect = pure)]
        fn $parse(text: &str) -> Result<$t, ParseIntError> {
            text.parse::<$t>()
                .map_err(|e| parse_int_error(text.to_owned(), &e))
        }

        /// `<int>::from_str_radix`. A radix outside Rust's `2..=36` is a
        /// trap and not an `Err`: Rust panics on it too, because it is the
        /// caller's constant and not the parsed text.
        #[extern_fn(name = "from_str_radix", effect = pure)]
        fn $radix(text: &str, radix: u32) -> Result<$t, ParseIntError> {
            assert!(
                (2..=36).contains(&radix),
                "from_str_radix: radix {radix} is outside 2..=36"
            );
            <$t>::from_str_radix(text, radix)
                .map_err(|e| parse_int_error(text.to_owned(), &e))
        }

        fn $registry<R>() -> Registry<R>
        where
            R: Runtime,
        {
            extern_registry! {
                ns: $ns,
                fns: [$from_str, $parse, $radix],
            }
        }
    )*};
}

from_str_ints! {
    from_str_i8 / parse_i8 / from_str_radix_i8: i8 as "i8" => i8_registry,
    from_str_i16 / parse_i16 / from_str_radix_i16: i16 as "i16" => i16_registry,
    from_str_i32 / parse_i32 / from_str_radix_i32: i32 as "i32" => i32_registry,
    from_str_i64 / parse_i64 / from_str_radix_i64: i64 as "i64" => i64_registry,
    from_str_u8 / parse_u8 / from_str_radix_u8: u8 as "u8" => u8_registry,
    from_str_u16 / parse_u16 / from_str_radix_u16: u16 as "u16" => u16_registry,
    from_str_u32 / parse_u32 / from_str_radix_u32: u32 as "u32" => u32_registry,
    from_str_u64 / parse_u64 / from_str_radix_u64: u64 as "u64" => u64_registry,
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

/// What an integer was not: a Unicode scalar value.
#[derive(TyArg)]
pub enum CharError {
    NotAChar(i64),
}

/// `char::from_u32`: the checked half of the `char` conversions.
///
/// `as` has the other half and does not cover this one. RFC-0049's `as` is
/// total, and Rust admits only `u8 as char`, which reaches U+00FF and no
/// further; every code point above it arrives through a check that can
/// fail, so this returns a `Result` (RFC-0038) and stays.
#[extern_fn(effect = pure)]
fn int_to_char(n: i64) -> Result<char, CharError> {
    let Ok(code) = u32::try_from(n) else {
        return Err(CharError::NotAChar(n));
    };
    char::from_u32(code).ok_or(CharError::NotAChar(n))
}

pub fn conversion_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        signatures: [sig::to_int],
        fns: [
            display_i8, display_i16, display_i32, display_int,
            display_u8, display_u16, display_u32, display_u64,
            display_float, display_char, display_bool,
            to_string,
            to_int_bool,
            int_to_char,
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
        let signatures = 1;
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

    /// Each instance appends to what `out` already holds, and its text is
    /// the text Rust's `to_string` gives the same value.
    #[test]
    fn display_appends_the_text_to_string_gives() {
        let mut out = String::from(">");
        display_i8(&-1, &mut out);
        display_int(&42, &mut out);
        display_u64(&u64::MAX, &mut out);
        display_float(&1.5, &mut out);
        display_char(&'c', &mut out);
        display_bool(&true, &mut out);
        let expected = [
            ">".to_string(),
            (-1i8).to_string(),
            42i64.to_string(),
            u64::MAX.to_string(),
            1.5f64.to_string(),
            'c'.to_string(),
            true.to_string(),
        ]
        .concat();
        assert_eq!(out, expected);
    }

    /// RFC-0070 rule 5: `display` stands at no `str` and at no `String`;
    /// the owned copy of text is `string::to_string`, and `std::to_string`
    /// is generic.
    #[test]
    fn display_stands_at_no_text() {
        let i = Interner::new();
        let reg = Externs::combine(vec![conversion_registry::<TypesOnly>()], &i)
            .expect("registry combines");
        let qref = acvus_extern::QualifiedRef::qualified(i.intern("core"), i.intern("display"));
        let function = reg
            .functions
            .iter()
            .find(|f| f.qref == qref)
            .expect("the signature is declared");
        let acvus_extern::FnKind::Extern { instances, .. } = &function.kind else {
            panic!("display is an extern")
        };
        let at_text = instances.concrete.iter().filter(|instance| {
            let acvus_extern::PolyTy::Fn { params, .. } = &instance.ty else {
                panic!("an instance is a function")
            };
            matches!(
                &params[0].ty,
                acvus_extern::PolyTy::Ref(_, lent)
                    if matches!(*lent.ty(), acvus_extern::PolyTy::Str | acvus_extern::PolyTy::String)
            )
        });
        assert_eq!(at_text.count(), 0);
        assert_eq!(instances.concrete.len(), 11);
    }

    #[test]
    fn every_integer_type_has_its_own_from_str_and_parse() {
        let i = Interner::new();
        let reg =
            Externs::combine(from_str_registries::<TypesOnly>(), &i).expect("registries combine");
        let core = Externs::<TypesOnly>::combine(vec![], &i).expect("core combines");
        assert_eq!(reg.functions.len() - core.functions.len(), 24);
        assert_eq!(reg.handlers.len() - core.handlers.len(), 24);
        for ns in ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"] {
            for name in ["from_str", "parse", "from_str_radix"] {
                let qref = acvus_extern::QualifiedRef::qualified(i.intern(ns), i.intern(name));
                assert!(
                    reg.handlers.contains_key(&qref),
                    "{ns}::{name} is registered"
                );
            }
        }
    }
}
