//! A derived type's box is keyed by its payload with each uniform type
//! parameter at its canonical form, and read at the parameter's own
//! (RFC-0076). That read is sound only if the payload's layout reaches no
//! uniform parameter through a trait, which the author asserts with
//! `unsafe(uniform_payload)`; without it the derive refuses.
use acvus_extern::{ExternType, Var, kind};

#[derive(ExternType)]
#[repr(transparent)]
#[extern_type(name = "Bag")]
struct Bag<T>(Vec<T>)
where
    T: Var<kind::Type>;

fn main() {}
