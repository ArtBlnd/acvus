//! A `Finished` is branded with its call's lifetime, invariantly: one built
//! by another call's `Output` is no result of this one, even where that call
//! outlives this one (RFC-0097 rule 3).
use acvus_extern::{Finished, Output, Runtime};

fn swapped<'a, 'b: 'a, T, R>(_: Output<'a, T, R>, other: Output<'b, T, R>) -> Finished<'a, T, R>
where
    R: Runtime,
{
    other.finish()
}

fn main() {}
