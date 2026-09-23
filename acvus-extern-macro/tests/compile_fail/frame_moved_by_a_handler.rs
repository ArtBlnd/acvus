//! The frame a handler's `Ctx` holds is the machine's own, and the machine
//! writes its cells after the call. Safe code neither reaches the frame nor
//! lends itself a second `Ctx` to exchange it with (RFC-0080 rule 2).
#![forbid(unsafe_code)]
use acvus_extern::{Ctx, Runtime, extern_fn};

#[extern_fn(effect = pure)]
fn swap_frame<Rt>(ctx: &mut Ctx<'_, Rt>) -> i64
where
    Rt: Runtime,
{
    let mut rooted = ctx.rt.rooted();
    let other = Rt::ctx_of(&mut rooted);
    std::mem::swap(&mut ctx.frame, &mut other.frame);
    7
}

#[extern_fn(effect = pure)]
fn swap_ctx<Rt>(ctx: &mut Ctx<'_, Rt>) -> i64
where
    Rt: Runtime,
{
    let mut rooted = ctx.rt.rooted();
    std::mem::swap(ctx, Rt::ctx_of(&mut rooted));
    7
}

fn main() {}
