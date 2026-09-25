//! `total_order` is a comparison's law, and `associative` a combining
//! function's: a declaration states one form (RFC-0082 rule 10).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, law(total_order, associative))]
fn order(a: &i64, b: &i64) -> i64 {
    (a - b).signum()
}

fn main() {}
