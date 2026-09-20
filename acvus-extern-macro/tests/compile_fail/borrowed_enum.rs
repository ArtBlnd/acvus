//! A borrowed enum crosses as its projection, never as `&E` (RFC-0050 rule
//! 6): `BorrowedWhole` has no impl, so the `Borrowable` the derive emits for
//! a projected enum never holds, and the message names `ShapeRef<'_>`.
use acvus_extern::{TyArg, extern_fn};

#[derive(TyArg)]
#[projection]
pub enum Shape {
    Empty,
    Count(i64),
}

#[extern_fn(effect = pure)]
fn is_empty(s: &Shape) -> bool {
    matches!(s, Shape::Empty)
}

fn main() {}
