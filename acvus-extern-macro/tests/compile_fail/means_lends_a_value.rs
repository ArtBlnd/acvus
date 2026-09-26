//! `*x` reads what a reference parameter lends (RFC-0104 rule 1): a
//! parameter taken by value lends nothing.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, means(*a))]
fn same(a: i64) -> i64 {
    a
}

fn main() {}
