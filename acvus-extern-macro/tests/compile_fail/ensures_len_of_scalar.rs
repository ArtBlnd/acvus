//! `len` reads a slice or a container, and an integer is neither
//! (RFC-0082 rule 4).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, ensures(ret = len(n)))]
fn same(n: u64) -> u64 {
    n
}

fn main() {}
