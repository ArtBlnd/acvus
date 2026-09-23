//! A term reads the declaration's own parameters (RFC-0082 rule 4).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, ensures(ret <= b))]
fn same(a: u64) -> u64 {
    a
}

fn main() {}
