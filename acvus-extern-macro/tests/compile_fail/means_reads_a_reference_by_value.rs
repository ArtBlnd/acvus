//! A term reads a reference parameter as `*x` (RFC-0104 rule 1): the
//! parameter's name alone is refused.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, means(a))]
fn same(a: &i64) -> i64 {
    *a
}

fn main() {}
