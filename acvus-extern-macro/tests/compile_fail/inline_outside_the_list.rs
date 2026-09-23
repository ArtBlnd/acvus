//! `Inline` says a type lives in the runtime's value word, and
//! `Erased::get` reads the word as that type on the strength of it. The
//! list `for_each_inline!` names is the whole of it: a type outside the
//! list, derived or not, cannot say so (RFC-0080 rule 2).
#![forbid(unsafe_code)]
use acvus_extern::{ExternType, Inline};

#[derive(ExternType, Clone, Copy, Debug)]
#[extern_type(name = "Id")]
#[repr(transparent)]
pub struct Id(i64);

impl Inline for Id {}

fn main() {}
