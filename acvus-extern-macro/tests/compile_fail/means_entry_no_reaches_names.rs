//! A term reads and writes only an entry its declaration's `reaches` names
//! (RFC-0104 rule 1).
use acvus_extern::{Var, extern_fn, kind};

#[extern_fn(effect = pure, means(&xs[key]))]
fn first<'a, K>(xs: &'a Vec<K>, key: &K) -> Option<&'a K>
where
    K: Var<kind::Type>,
{
    drop(key);
    xs.first()
}

fn main() {}
