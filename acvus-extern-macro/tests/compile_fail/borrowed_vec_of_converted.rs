//! A borrowed `Vec<T>` or `Arr<T, N>` is read in place, and its storage is
//! the runtime's `Vec<Owned<Rt>>`: only an element whose `Vec` that storage is
//! can be borrowed that way (`InPlaceElement`). An extension type and a scalar
//! are each stored as their own value, so a container of them is not a `Vec`
//! of them, and the declaration is refused where it was once accepted and
//! failed at its first call. `Erased<Rt, X>` is refused too: its layout is
//! `Owned<Rt>`'s, but two instantiations of `Vec` share none Rust promises.
use acvus_extern::{Arr, Erased, ExternType, Runtime, Var, extern_fn, kind};

#[derive(ExternType)]
#[repr(transparent)]
pub struct Rendered(());

#[extern_fn(effect = pure)]
async fn count_rendered(xs: &Vec<Rendered>) -> u64 {
    xs.len() as u64
}

#[extern_fn(effect = pure)]
fn count_ints(xs: &Vec<i64>) -> u64 {
    xs.len() as u64
}

#[extern_fn(effect = pure)]
fn clear_ints(xs: &mut Vec<i64>) {
    xs.clear();
}

#[extern_fn(effect = pure)]
fn count_array<N>(xs: &Arr<i64, N>) -> u64
where
    N: Var<kind::Length>,
{
    xs.0.len() as u64
}

#[extern_fn(effect = pure)]
fn count_erased<Rt>(xs: &Vec<Erased<Rt, Rendered>>) -> u64
where
    Rt: Runtime,
{
    xs.len() as u64
}

fn main() {}
