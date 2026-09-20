//! A value carries no pointer (RFC-0067, step 3 second half). What fills a
//! bounded type variable while the handler runs is a carrier, and a
//! carrier's Rust type belongs to the declaration that wrote the bound — so
//! a payload storing one is read back by any other declaration as a type it
//! does not have.
use std::marker::PhantomData;

use acvus_extern::{ExternType, Runtime, Var, kind};

#[derive(ExternType)]
#[extern_type(name = "Doubled")]
#[repr(transparent)]
struct Doubled<I, Rt>(I, PhantomData<Rt>)
where
    I: Var<kind::Type>,
    Rt: Runtime;

fn main() {}
