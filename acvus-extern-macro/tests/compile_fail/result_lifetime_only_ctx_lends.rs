//! A result lifetime only `ctx` names: no acvus parameter lends it, so no
//! flow can say what the result holds there (RFC-0079 rule 6).
use acvus_extern::{Ctx, Runtime, extern_fn};

#[extern_fn(effect = pure)]
fn from_ctx<'c, Rt>(ctx: &mut Ctx<'c, Rt>, s: &str) -> &'c str
where
    Rt: Runtime,
{
    let _ = (ctx, s);
    ""
}

fn main() {}
