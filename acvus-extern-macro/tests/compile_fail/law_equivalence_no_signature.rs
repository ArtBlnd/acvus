//! `law(equivalence)` is stated on a `core::eq` instance (RFC-0082 rule 11):
//! a declaration that is no signature's instance is refused.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, law(equivalence))]
fn same(a: &i64, b: &i64) -> bool {
    a == b
}

fn main() {}
