//! `copies(x)` is `means(*x)` (RFC-0104 rule 5), and the keyword is gone.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, copies(a))]
fn same(a: &i64) -> i64 {
    *a
}

fn main() {}
