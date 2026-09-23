//! A `heavy` declaration is awaited, so its result outlives the frame the
//! call ran on; a slice is a borrow of that frame's loan (RFC-0023 rule 6).
use acvus_extern::{Runtime, TransparentOver, Var, extern_fn, kind};

#[extern_fn(effect = pure, heavy)]
fn elements<T, Rt>(c: &Vec<T>) -> &[T]
where
    T: Var<kind::Type> + TransparentOver<Rt>,
    Rt: Runtime,
{
    c
}

fn main() {}
