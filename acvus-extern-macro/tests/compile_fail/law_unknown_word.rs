//! The law vocabulary is closed (RFC-0082 rule 1): a word it does not
//! hold is refused.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, law(associative, idempotent))]
fn join(a: i64, b: i64) -> i64 {
    a.max(b)
}

fn main() {}
