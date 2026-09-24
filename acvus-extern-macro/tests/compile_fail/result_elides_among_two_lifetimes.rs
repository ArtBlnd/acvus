//! A result that elides its lifetime among two parameter lifetimes: Rust's
//! elision gives it none, and the macro says which to name (RFC-0079 rule 6).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure)]
fn either(a: &str, b: &str) -> &str {
    if a.len() > b.len() { a } else { b }
}

fn main() {}
