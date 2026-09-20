//! A struct variant's payload is an object the enum writes and no Rust type
//! names, so `#[projection]` on such an enum is refused at the derive
//! (RFC-0050 rule 6).
use acvus_extern::TyArg;

#[derive(TyArg)]
#[projection]
pub enum Shape {
    Empty,
    At { x: i64 },
}

fn main() {}
