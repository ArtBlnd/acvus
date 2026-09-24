//! A carrier written without its lifetime hides the lifetime the flows are
//! read from: `elided_lifetimes_in_paths` refuses it (RFC-0079 rule 6).
use acvus_extern::{Ref, Runtime, Shared, extern_fn};

#[extern_fn(effect = pure)]
fn peek<Rt>(r: Ref<String, Shared, Rt>) -> i64
where
    Rt: Runtime,
{
    let _ = r;
    0
}

fn main() {}
