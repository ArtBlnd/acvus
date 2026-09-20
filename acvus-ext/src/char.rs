//! `char` under Rust's names, the pure part. A `char` is one Unicode scalar
//! value and an inline word (RFC-0058), so every function here takes and
//! returns it by value.
//!
//! `to_digit` and `from_digit` are the two Rust functions that take a radix.
//! Rust panics on a radix above 36; a radix is a program's own constant far
//! more often than a datum, so an out-of-range radix is refused here rather
//! than folded into the `None` that "this character is not a digit" already
//! means.

use acvus_extern::{Registry, Runtime, extern_fn, extern_registry};

/// Rust's `char::MAX_RADIX`, which `std` does not export.
const MAX_RADIX: u32 = 36;

fn checked_radix(what: &str, radix: u32) -> u32 {
    if radix > MAX_RADIX {
        panic!("{what}: radix {radix} is above {MAX_RADIX}")
    }
    radix
}

#[extern_fn(effect = pure)]
fn is_alphabetic(c: char) -> bool {
    c.is_alphabetic()
}

#[extern_fn(effect = pure)]
fn is_numeric(c: char) -> bool {
    c.is_numeric()
}

#[extern_fn(effect = pure)]
fn is_alphanumeric(c: char) -> bool {
    c.is_alphanumeric()
}

#[extern_fn(effect = pure)]
fn is_whitespace(c: char) -> bool {
    c.is_whitespace()
}

#[extern_fn(effect = pure)]
fn is_uppercase(c: char) -> bool {
    c.is_uppercase()
}

#[extern_fn(effect = pure)]
fn is_lowercase(c: char) -> bool {
    c.is_lowercase()
}

#[extern_fn(effect = pure)]
fn is_ascii(c: char) -> bool {
    c.is_ascii()
}

#[extern_fn(effect = pure)]
fn is_ascii_digit(c: char) -> bool {
    c.is_ascii_digit()
}

#[extern_fn(effect = pure)]
fn is_ascii_alphabetic(c: char) -> bool {
    c.is_ascii_alphabetic()
}

#[extern_fn(effect = pure)]
fn to_ascii_uppercase(c: char) -> char {
    c.to_ascii_uppercase()
}

#[extern_fn(effect = pure)]
fn to_ascii_lowercase(c: char) -> char {
    c.to_ascii_lowercase()
}

/// The value `c` carries as a digit in base `radix`, or `None` where it
/// carries none. A `radix` above 36 is refused.
#[extern_fn(effect = pure)]
fn to_digit(c: char, radix: u32) -> Option<u32> {
    c.to_digit(checked_radix("to_digit", radix))
}

/// The character that writes `digit` in base `radix`, or `None` where
/// `digit` is not a digit of that base. A `radix` above 36 is refused.
#[extern_fn(effect = pure)]
fn from_digit(digit: u32, radix: u32) -> Option<char> {
    char::from_digit(digit, checked_radix("from_digit", radix))
}

/// How many bytes `c` occupies in UTF-8.
#[extern_fn(effect = pure)]
fn len_utf8(c: char) -> u64 {
    c.len_utf8() as u64
}

pub fn char_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "char",
        fns: [
            is_alphabetic, is_numeric, is_alphanumeric, is_whitespace,
            is_uppercase, is_lowercase, is_ascii, is_ascii_digit, is_ascii_alphabetic,
            to_ascii_uppercase, to_ascii_lowercase,
            to_digit, from_digit, len_utf8,
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
        let reg =
            Externs::combine(vec![char_registry::<TypesOnly>()], &i).expect("registry combines");
        let core = Externs::<TypesOnly>::combine(vec![], &i).expect("core combines");
        assert_eq!(reg.functions.len() - core.functions.len(), 14);
        assert_eq!(reg.handlers.len() - core.handlers.len(), 14);
    }
}
