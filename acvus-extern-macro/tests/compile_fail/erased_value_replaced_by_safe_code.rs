//! An `Erased<R, T>` names the Rust type its value was erased from, so safe
//! code neither conjures one (`Default`) nor writes a value into it
//! (`DerefMut`): both are `Owned`'s alone, as inherent methods (RFC-0076).
use acvus_extern::{Erased, Runtime};

fn conjure<Rt>() -> Erased<Rt, String>
where
    Rt: Runtime,
{
    Erased::<Rt, String>::default()
}

fn overwrite<Rt>(mut erased: Erased<Rt, String>, value: Rt::Value) -> Erased<Rt, String>
where
    Rt: Runtime,
{
    *erased = value;
    erased
}

fn main() {}
