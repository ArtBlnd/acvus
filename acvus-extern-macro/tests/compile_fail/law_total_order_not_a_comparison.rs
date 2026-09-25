//! `total_order` is stated over `f(a: &T, b: &T) -> i64` (RFC-0082 rule 10):
//! a comparison that takes its operands by value is refused at the
//! declaration.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, law(total_order))]
fn order(a: i64, b: i64) -> i64 {
    (a - b).signum()
}

fn main() {}
