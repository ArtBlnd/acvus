//! A payload of the author's own struct is proved once that struct derives
//! `UniformPayload`.
use acvus_extern::{ExternType, UniformPayload, Var, kind};

#[derive(UniformPayload)]
struct Pair<T> {
    first: T,
    rest: Vec<T>,
    count: usize,
}

#[derive(ExternType)]
#[repr(transparent)]
#[extern_type(name = "Paired")]
struct Paired<T>(Pair<T>)
where
    T: Var<kind::Type>;

fn main() {}
