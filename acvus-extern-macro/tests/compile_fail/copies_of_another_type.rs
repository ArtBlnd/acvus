//! `copies(x)` is stated over `f(.., x: &T, ..) -> T` (RFC-0082 rule 10): a
//! result of another type than what `x` lends is refused.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, copies(a))]
fn widen(a: &i64) -> String {
    a.to_string()
}

fn main() {}
