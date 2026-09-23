//! `fold` is stated over `f(s: &mut S, x: X)` returning nothing (RFC-0082
//! rule 3): a function taking its state by value is refused.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, law(fold(combine = join, identity = empty)))]
fn put(s: Vec<i64>, x: i64) -> Vec<i64> {
    let mut s = s;
    s.push(x);
    s
}

fn main() {}
