//! `inverse` is stated over `f(s: &mut S) -> Option<X>` (RFC-0082 rule 3):
//! a function taking its state by value is refused.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, law(inverse = put))]
fn take(s: Vec<i64>) -> Option<i64> {
    s.last().copied()
}

fn main() {}
