//! A `RustFn`'s function type carries the effect its extern declares, and its
//! body runs at `Task::Sync`: an extern that declares a task above `Sync`
//! would type a body that suspends (RFC-0097 rule 2).
use acvus_extern::{Runtime, RustFn, extern_fn};

#[extern_fn(effect = pure)]
async fn adder<R>(k: i64) -> RustFn<(i64,), i64, R>
where
    R: Runtime,
{
    RustFn::new(move |_, args| args.with(0, |n: &i64| n + k).unwrap_or(k))
}

fn main() {}
