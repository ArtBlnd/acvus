//! A law over borrowed views is stated on `f(a: &str, b: &str) -> String`
//! (RFC-0082 rule 2): `str` is the view a `String` lends, and a result of
//! another type is refused at the declaration.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, law(associative))]
fn width(a: &str, b: &str) -> u64 {
    (a.len() + b.len()) as u64
}

fn main() {}
