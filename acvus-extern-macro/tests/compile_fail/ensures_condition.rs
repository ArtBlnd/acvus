//! A postcondition has no condition (RFC-0082 rule 4).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, ensures(ret <= a if a < b))]
fn lesser(a: u64, b: u64) -> u64 {
    a.min(b)
}

fn main() {}
