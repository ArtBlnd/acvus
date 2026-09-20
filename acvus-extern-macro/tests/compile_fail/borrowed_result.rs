//! A `Result` is a heap variant, not Rust's `Result<Owned, Owned>` (RFC-0050
//! rule 8), so no parameter borrows one: `Borrowable` carries the refusal for a
//! concrete parameter and `BorrowableSpecialized` for a monomorphized one. Both
//! markers are absent for `Result` and both crossings cross it by value, which
//! is what `acvus-interpreter-test`'s `mono_result.rs` runs.
use acvus_extern::{Monomorphize, Runtime, extern_fn};

#[extern_fn(effect = pure)]
fn succeeded(r: &Result<i64, String>) -> bool {
    r.is_ok()
}

#[extern_fn(effect = pure)]
fn succeeded_at<A, Rt>(r: &Result<A, String>) -> bool
where
    A: Monomorphize<(i64, String)>,
    Rt: Runtime,
{
    r.is_ok()
}

fn main() {}
