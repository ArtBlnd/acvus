//! A Rust slice is not one of the language's types: `Slice<T, Shared, Rt>` is
//! (RFC-0047). Both borrow modes are refused by the same missing impl.
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
