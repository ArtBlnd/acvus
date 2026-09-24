//! An `Erased<R, T>` is read as its `T` and at no other type: it does not
//! deref to the runtime's word, and naming its storage at another `Erased`'s
//! type takes the runtime's `Holding` (RFC-0076 rule 1).
use acvus_extern::{Erased, InPlaceElement, Owned, Runtime, extern_fn};

#[extern_fn(effect = opaque)]
fn peek<Rt>(s: &Erased<Rt, String>) -> i64
where
    Rt: Runtime,
{
    let _word: &Rt::Value = &**s;
    0
}

fn retype<Rt>(words: &Vec<Owned<Rt>>) -> &Vec<Erased<Rt, String>>
where
    Rt: Runtime,
{
    <Erased<Rt, String> as InPlaceElement<Rt>>::in_place(words)
}

fn main() {}
