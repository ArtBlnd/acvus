//! A step is stated on a declaration that pulls one stream (RFC-0099 rule 1).
use acvus_extern::{extern_fn};

#[extern_fn(effect = pure, step(yield x))]
fn plain(x: i64) -> i64 {
    x
}

fn main() {}
