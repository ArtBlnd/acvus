//! A term writes an entry only through a `&mut` table (RFC-0104 rule 1).
use acvus_extern::{Var, extern_fn, kind};

#[extern_fn(effect = pure, reaches(xs[key]), means(xs[key] = None; true))]
fn clears<K>(xs: &Vec<K>, key: K) -> bool
where
    K: Var<kind::Type>,
{
    drop((xs, key));
    true
}

fn main() {}
