//! An element is named by a `u64` index (RFC-0082 rule 7).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, reaches(a[i]))]
fn at(a: &Vec<u64>, i: i64) -> u64 {
    a[i as usize]
}

fn main() {}
