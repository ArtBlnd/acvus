//! `copies(x)` names a parameter of the declaration (RFC-0082 rule 10).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, copies(b))]
fn same(a: &i64) -> i64 {
    *a
}

fn main() {}
