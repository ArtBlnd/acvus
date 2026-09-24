//! `Erased` derefs to the runtime's word, which is `Copy`; making an
//! `Owned` of a copy would release the word twice, and one kept past the
//! call would come back holding a loan that ended. `Owned::from_value` is
//! `unsafe` (RFC-0080 rule 1).
use acvus_extern::{Erased, Owned, Runtime, extern_fn};

#[extern_fn(effect = opaque)]
fn free_word<Rt>(s: &Erased<Rt, String>) -> i64
where
    Rt: Runtime,
{
    drop(Owned::<Rt>::from_value(**s));
    0
}

fn main() {}
