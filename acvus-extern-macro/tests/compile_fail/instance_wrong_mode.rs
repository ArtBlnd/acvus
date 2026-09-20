//! The mode of a shared signature's first parameter reaches a requiring
//! handler through `Signature::Recv` and nowhere else, so a call written in
//! the wrong mode is refused at the handler's own `I::call` (RFC-0067
//! Decision 1). `step` takes `&mut I`; `drive` lends it `&it`.
use acvus_extern::{Carrier, InstanceOf, Runtime, Var, extern_fn, extern_signature, kind};

extern_signature! { ns: "probe", fn step<I>(it: &mut I) -> i64 where I: Var<kind::Type>; }

#[extern_fn(effect = pure)]
fn drive<I, Rt>(rt: &Rt, frame: &mut Rt::Frame<'_>, it: I) -> i64
where
    I: Var<kind::Type> + Carrier<Rt> + InstanceOf<step<I, Rt>, Rt>,
    Rt: Runtime,
{
    I::call(&it, rt, frame, ())
}

fn main() {}
