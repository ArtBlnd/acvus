//! A term names one entry `x[k]` (RFC-0104 rule 1).
use acvus_extern::{Var, extern_fn, kind};

#[extern_fn(effect = pure, reaches(xs[a], xs[b]), means(let first = xs[a]; xs[b]))]
fn both<K>(xs: &Vec<K>, a: K, b: K) -> bool
where
    K: Var<kind::Type>,
{
    drop((xs, a, b));
    true
}

fn main() {}
