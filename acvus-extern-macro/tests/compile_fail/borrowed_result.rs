//! A `Result` is a heap variant, not Rust's `Result<Owned, Owned>` (RFC-0050
//! rule 8), so no parameter borrows one: `Borrowable` carries the refusal for a
//! concrete parameter. A monomorphized one is refused wider than that, and
//! deliberately so — `CrossSpecialized` is the only bound `obj.rs` can drop
//! without reaching into `handler.rs`, and dropping it takes the family's
//! by-value crossing with the by-reference one. Restoring a narrower refusal
//! means a bound of its own on `ByRef<_, Specialized>`.
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
