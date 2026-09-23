//! The postcondition vocabulary is closed (RFC-0082 rule 4): a function it
//! does not hold is refused.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, ensures(ret <= min(a, b)))]
fn lesser(a: u64, b: u64) -> u64 {
    a.min(b)
}

fn main() {}
