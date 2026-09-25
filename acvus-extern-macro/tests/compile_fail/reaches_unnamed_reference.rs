//! A declaration of the places a call reaches names every reference
//! parameter (RFC-0082 rule 7).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, reaches(a[i]))]
fn both(a: &Vec<u64>, b: &Vec<u64>, i: u64) -> u64 {
    a[i as usize] + b[i as usize]
}

fn main() {}
