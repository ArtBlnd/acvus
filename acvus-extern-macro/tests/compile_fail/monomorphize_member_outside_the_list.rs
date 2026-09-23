//! A `Monomorphize` parameter compiles one handler per member type, and the
//! members are the list `mono_member!` names. A type outside the list
//! cannot say it is one (RFC-0080 rule 2).
#![forbid(unsafe_code)]
use acvus_extern::{ExternType, Monomorphize};

#[derive(ExternType, Clone, Copy, Debug)]
#[extern_type(name = "Id")]
#[repr(transparent)]
pub struct Id(i64);

impl<Types> Monomorphize<Types> for Id {}

fn main() {}
