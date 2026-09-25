//! `old(t)` reads a `&mut` parameter's state (RFC-0082 rule 4): a shared
//! parameter's state is the same at the call's start and its return.
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, ensures(ret = old(len(c))))]
fn counted(c: &Vec<u64>) -> u64 {
    c.len() as u64
}

fn main() {}
