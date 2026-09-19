//! A borrowed aggregate crosses as its projection, never as `&S` (RFC-0050
//! rule 6): `BorrowedWhole` has no impl, so the `Borrowable` the derive emits
//! for a projected struct never holds, and the message names `PointRef<'_>`.
use acvus_extern::{TyArg, extern_fn};

#[derive(TyArg)]
#[projection]
pub struct Point {
    pub x: i64,
    pub label: String,
}

#[extern_fn(effect = pure)]
fn point_x(p: &Point) -> i64 {
    p.x
}

fn main() {}
