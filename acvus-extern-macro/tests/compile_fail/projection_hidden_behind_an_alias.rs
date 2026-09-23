//! A projection through a uniform parameter behind a type alias, whose
//! associated type is declared `UniformPayload` at every marker, is refused:
//! the payload is proved at a marker no bound the struct writes can name.
use acvus_extern::{ExternType, UniformPayload, Var, kind};

pub trait Holds {
    type Held<M>: UniformPayload<M> + Send + Sync + 'static;
}

type HeldBy<T> = <T as Holds>::Held<()>;

#[derive(ExternType)]
#[repr(transparent)]
#[extern_type(name = "Held")]
struct Held<T>(HeldBy<T>)
where
    T: Var<kind::Type> + Holds;

fn main() {}
