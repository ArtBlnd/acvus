//! An `async fn` declaration owns a copy of its argument run across every
//! await point, and the elements a slice names belong to a frame that does
//! not wait for it (RFC-0047 rule 6).
use acvus_extern::{Runtime, Slice, TyVar, extern_fn};

#[extern_fn(effect = pure)]
async fn count<T, Rt>(rt: &Rt, s: Slice<T, Rt>) -> u64
where
    T: TyVar,
    Rt: Runtime,
{
    let _ = rt;
    s.into_elements().len() as u64
}

fn main() {}
