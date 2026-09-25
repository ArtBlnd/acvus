//! A call reaches a place only through a reference argument (RFC-0082 rule 7).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, reaches(c[i]))]
fn first(c: u64, i: u64) -> u64 {
    c + i
}

fn main() {}
