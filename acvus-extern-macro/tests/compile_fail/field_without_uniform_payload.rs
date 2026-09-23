//! A payload of the author's own struct that does not derive
//! `UniformPayload` is refused, naming that struct and the derive.
use acvus_extern::{ExternType, Var, kind};

struct Pair<T> {
    first: T,
    rest: Vec<T>,
}

#[derive(ExternType)]
#[repr(transparent)]
#[extern_type(name = "Paired")]
struct Paired<T>(Pair<T>)
where
    T: Var<kind::Type>;

fn main() {}
