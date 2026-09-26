//! A term calls registered externs, never a parameter (RFC-0104 rule 1).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, means(a(b)))]
fn applied(a: i64, b: i64) -> i64 {
    a + b
}

fn main() {}
