//! A derived struct without a projection is kept as an object, which holds
//! no `Point` to read in place, so it has no `Borrowable` and `&Point` is
//! refused at the declaration. It was once accepted and panicked at its
//! first call.
use acvus_extern::{TyArg, extern_fn};

#[derive(TyArg)]
pub struct Point {
    pub x: i64,
    pub label: String,
}

#[extern_fn(effect = pure)]
fn point_x(p: &Point) -> i64 {
    p.x
}

fn main() {}
