//! `old(t)` reads the call's start, where there is no `ret` (RFC-0082
//! rule 4).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, ensures(len(c) = old(ret)))]
fn popped(c: &mut Vec<u64>) -> u64 {
    c.pop();
    c.len() as u64
}

fn main() {}
