//! An `Option` has no storage of its own type, so no parameter borrows one
//! (RFC-0039): `Borrowable` is the bound, and its absence is the refusal.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure)]
fn is_some(v: &Option<i64>) -> bool {
    v.is_some()
}

fn main() {}
