//! A Rust slice is the language's `&[T]` only where its element is stored as
//! one of the runtime's values (`T: TransparentOver<Rt>`): a variable's fill
//! or an `Erased<Rt, i64>`. A slice of bare `i64` names no storage the
//! language has, and both borrow modes are refused by the same missing impl
//! (RFC-0047, RFC-0068 rule 4).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure)]
fn first(v: &[i64]) -> i64 {
    v[0]
}

#[extern_fn(effect = pure)]
fn clear(v: &mut [i64]) {
    v[0] = 0;
}

fn main() {}
