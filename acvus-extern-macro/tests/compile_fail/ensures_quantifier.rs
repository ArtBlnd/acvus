//! A postcondition has no quantifier (RFC-0082 rule 4).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, ensures(forall i: ret <= i))]
fn least(a: u64) -> u64 {
    a
}

fn main() {}
