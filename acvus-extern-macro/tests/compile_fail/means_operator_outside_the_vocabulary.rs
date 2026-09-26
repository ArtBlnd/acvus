//! A term's operators are the language's (RFC-0104 rule 1), and `^` is
//! none of them.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, means(a ^ b))]
fn add(a: i64, b: i64) -> i64 {
    a ^ b
}

fn main() {}
