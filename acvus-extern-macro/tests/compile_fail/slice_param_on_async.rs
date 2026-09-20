//! An `async fn` declaration owns a copy of its argument run across every
//! await point, and the elements a slice names belong to a frame that does
//! not wait for it (RFC-0047 rule 6).
use acvus_extern::{Ctx, Runtime, Shared, Slice, Var, extern_fn, kind};

#[extern_fn(effect = pure)]
async fn count<T, Rt>(ctx: &mut Ctx<'_, Rt>, s: Slice<T, Shared, Rt>) -> u64
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    let _ = ctx;
    s.into_elements().len() as u64
}

fn main() {}
