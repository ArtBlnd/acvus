//! A receiver's storage derefs to the runtime's word, which is `Copy`;
//! making an `Owned` of a copy would release the word twice, and one kept
//! past the call would come back holding a loan that ended. Holding a bare
//! word is the runtime's: `Owned::from_value` and `Owned::vacant` take its
//! `Holding`, which a handler is never handed (RFC-0080 rule 2).
use std::ops::Deref;

use acvus_extern::{Owned, Runtime, Var, extern_fn, kind};

#[extern_fn(effect = opaque)]
fn free_word<I, Rt>(it: &I) -> i64
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    drop(unsafe { Owned::<Rt>::from_value(**it) });
    drop(Owned::<Rt>::vacant());
    0
}

fn main() {}
