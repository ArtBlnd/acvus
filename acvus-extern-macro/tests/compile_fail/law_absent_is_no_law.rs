//! `law(absent = v)` is no law: what a call does to an entry is its
//! `means(..)` (RFC-0104 rule 5).
use acvus_extern::extern_fn;

#[extern_fn(effect = pure, law(absent = value))]
fn first(xs: &mut Vec<i64>, value: i64) -> &mut i64 {
    xs.push(value);
    &mut xs[0]
}

fn main() {}
