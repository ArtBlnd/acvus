//! A payload borrows its type in place, so it needs a `Borrowed` impl — the
//! same requirement the projected struct's fields carry. A `Vec` has none:
//! the language stores it as `Vec<Owned<Rt>>` and only a borrow naming the
//! runtime could reach it (RFC-0050 rule 6).
use acvus_extern::TyArg;

#[derive(TyArg)]
#[projection]
pub enum Shape {
    Empty,
    Many(Vec<i64>),
}

fn main() {}
