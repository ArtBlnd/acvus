//! A type variable takes no lifetime bound in a `where` clause either
//! (RFC-0079 rule 8).
#![forbid(unsafe_code)]
use acvus_extern::{Var, extern_fn, kind};

#[extern_fn(effect = pure)]
fn first<'a, T>(v: &'a Vec<T>) -> i64
where
    T: Var<kind::Type>,
    T: 'a,
{
    v.len() as i64
}

fn main() {}
