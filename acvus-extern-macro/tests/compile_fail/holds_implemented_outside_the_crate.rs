//! An `Instance` names its receiver through `Holds`, whose impls are the
//! value at the signature's receiver type and the `&mut` of it its requirer
//! was lent. Another crate adds no third: `Holds` is sealed, so a type that
//! is no receiver cannot be named as one (RFC-0067 rule 1).
use acvus_extern::{Holds, Runtime};

struct Forged<Rt>(Rt::Value)
where
    Rt: Runtime;

impl<Rt> Holds<Rt, Forged<Rt>> for Forged<Rt>
where
    Rt: Runtime,
{
    fn word(&self) -> &Rt::Value {
        &self.0
    }
}

fn main() {}
