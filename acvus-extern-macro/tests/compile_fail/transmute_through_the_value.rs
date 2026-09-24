//! `T -> Value -> U` is a transmute. Safe code writes neither half: each
//! crossing takes the glue's `Crossing` and each retyping of a holder the
//! runtime's `Holding`, and safe code holds neither.
#![forbid(unsafe_code)]
use acvus_extern::{Erased, OneValue, Owned, Runtime, Stored};

fn through_the_crossing<Rt>(rt: &Rt, x: i64) -> Erased<Rt, String>
where
    Rt: Runtime,
{
    let word = x.erase(rt);
    <Erased<Rt, String> as OneValue<Rt>>::materialize(rt, word)
}

fn through_a_holder<Rt>(rt: &Rt, x: i64) -> String
where
    Rt: Runtime,
{
    let held: Owned<Rt> = Owned::erased(rt, x);
    let retyped = <Erased<Rt, String> as Stored<Rt>>::from_payload(&held);
    retyped.as_ref(rt).clone()
}

fn main() {}
