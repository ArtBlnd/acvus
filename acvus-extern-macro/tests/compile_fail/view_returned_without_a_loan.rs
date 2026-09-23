//! A view is a projection of a parameter the caller kept (RFC-0047 rule 3): a
//! declaration whose every parameter is taken by value has no storage the
//! result's bytes could belong to.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure)]
fn sign(n: i64) -> &'static str {
    if n < 0 { "-" } else { "+" }
}

fn main() {}
