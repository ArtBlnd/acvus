//! A term names the declaration's parameters and its own bindings
//! (RFC-0104 rule 1).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, means(*b))]
fn same(a: &i64) -> i64 {
    *a
}

fn main() {}
