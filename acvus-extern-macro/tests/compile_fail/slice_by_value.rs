//! A slice crosses as the two registers it occupies and is no value, so a
//! by-value parameter that names one has no `OneValue` crossing to take it
//! with (RFC-0047 amended).
use acvus_extern::{Runtime, Slice, TyVar, extern_fn};

#[extern_fn(effect = pure)]
fn head<T, Rt>(rt: &Rt, s: Slice<T, Rt>) -> u64
where
    T: TyVar,
    Rt: Runtime,
{
    let _ = rt;
    s.into_elements().len() as u64
}

fn main() {}
