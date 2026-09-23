//! A `Chosen` type parameter keys each instance's box at its own Rust type,
//! so no one set of space hooks reads every instance: the derive refuses
//! `space` on such a type (RFC-0033).
use acvus_extern::{Chosen, ExternType, Var, kind};

#[derive(ExternType)]
#[extern_type(name = "Picked", space)]
#[repr(transparent)]
struct Picked<T>(Vec<T>)
where
    T: Var<kind::Type> + Chosen;

fn main() {}
