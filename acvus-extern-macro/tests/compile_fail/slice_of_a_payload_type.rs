//! A `Slice<X, _, Rt>` reads the run as `[X]`, which holds only where `X` is
//! the runtime's value under another name. An extension type is stored as
//! its payload, so the declaration is refused and the diagnostic names the
//! slice whose elements do read as `X`.
use acvus_extern::{ExternType, Mut, Runtime, Shared, Slice, extern_fn};

#[derive(ExternType)]
#[repr(transparent)]
pub struct Rendered(());

#[extern_fn(effect = pure)]
fn count_rendered<Rt>(xs: Slice<'_, Rendered, Shared, Rt>) -> u64
where
    Rt: Runtime,
{
    xs.len() as u64
}

#[extern_fn(effect = pure)]
fn count_rendered_mut<Rt>(xs: Slice<'_, Rendered, Mut, Rt>) -> u64
where
    Rt: Runtime,
{
    xs.len() as u64
}

fn main() {}
