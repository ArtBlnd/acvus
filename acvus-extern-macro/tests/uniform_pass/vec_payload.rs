//! A `Vec<T>` payload holds its uniform parameter only as elements, which
//! `UniformPayload` proves, so the derive asks for no attribute.
use acvus_extern::{ExternType, Var, kind};

#[derive(ExternType)]
#[repr(transparent)]
#[extern_type(name = "Bag")]
struct Bag<T>(Vec<T>)
where
    T: Var<kind::Type>;

fn main() {}
