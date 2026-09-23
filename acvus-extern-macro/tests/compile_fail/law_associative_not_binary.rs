//! `associative` is stated over `f(a: T, b: T) -> T` (RFC-0082 rule 2):
//! a function whose operands differ in type is refused at the declaration.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, law(associative))]
fn scale(a: i64, b: f64) -> i64 {
    a * b as i64
}

fn main() {}
