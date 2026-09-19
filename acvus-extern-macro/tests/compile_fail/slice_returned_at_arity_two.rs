//! RFC-0047 §3 admits a slice result from a declaration of one parameter:
//! the run it names is the caller's loan, and every other arity returns one
//! value.
use acvus_extern::{Ref, Runtime, Slice, TyVar, extern_fn};

#[extern_fn(effect = pure)]
fn tail<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Rt>, from: u64) -> Slice<T, Rt>
where
    T: TyVar,
    Rt: Runtime,
{
    let _ = from;
    Slice::of(c.elements(rt))
}

fn main() {}
