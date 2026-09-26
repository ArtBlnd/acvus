//! `Value::inline` checks its kind only in a debug build, so it is the
//! runtime's alone: a tooling crate makes an inline value by its type.
use acvus_interpreter::{Kind, Value};

fn main() {
    let _ = Value::inline(Kind::Large, 8);
}
