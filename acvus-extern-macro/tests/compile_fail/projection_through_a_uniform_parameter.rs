//! A payload that projects through a uniform type parameter has whatever
//! layout a trait impl for that parameter chooses, so no one box serves
//! every instantiation: the derive refuses it, with or without
//! `unsafe(uniform_payload)` (RFC-0076).
use acvus_extern::{ExternType, Var, kind};

pub trait Holds {
    type Held: Send + Sync + 'static;
}

#[derive(ExternType)]
#[repr(transparent)]
#[extern_type(name = "Held")]
#[extern_type(unsafe(uniform_payload))]
struct Held<T>(<T as Holds>::Held)
where
    T: Var<kind::Type> + Holds;

fn main() {}
