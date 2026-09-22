//! A `heavy` declaration is offloaded and awaited, so its arguments were laid
//! on a frame that is gone when it resumes; a slice is a borrow of that
//! frame's registers (RFC-0047 rule 6).
use acvus_extern::{Ctx, Runtime, Shared, Slice, Var, extern_fn, kind};

#[extern_fn(effect = pure, heavy)]
fn count<T, Rt>(ctx: &mut Ctx<'_, Rt>, s: Slice<T, Shared, Rt>) -> u64
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    let _ = ctx;
    s.len() as u64
}

fn main() {}
