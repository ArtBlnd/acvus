//! A `heavy` declaration is awaited, so its result outlives the frame the
//! call ran on; a slice is a borrow of that frame's loan (RFC-0047 §3).
use acvus_extern::{Ref, Runtime, Shared, Slice, Var, extern_fn, kind};

#[extern_fn(effect = pure, heavy)]
fn elements<T, Rt>(rt: &Rt, c: Ref<Vec<T>, Shared, Rt>) -> Slice<T, Shared, Rt>
where
    T: Var<kind::Type>,
    Rt: Runtime,
{
    Slice::of(c.elements(rt))
}

fn main() {}
