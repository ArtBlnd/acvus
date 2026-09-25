//! `copies(x)` names a reference parameter (RFC-0082 rule 10): one taken by
//! value is refused.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, copies(a))]
fn same(a: i64) -> i64 {
    a
}

fn main() {}
