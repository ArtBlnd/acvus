//! RFC-0067 step 3, second half, rule 7. `Monomorphize` compiles the body
//! once per member, with the variable standing for that member's Rust type;
//! a required instance fills the same variable with the carrier, which has
//! none of the member's methods. The two cannot hold at once, so the
//! declaration is refused rather than emitted for the trait solver to fail
//! on.
use acvus_extern::{Carrier, InstanceOf, Monomorphize, Runtime, Var, extern_fn, extern_signature, kind};

extern_signature! { ns: "probe", fn eq<T>(a: &T, b: &T) -> bool where T: Var<kind::Type>; }

#[extern_fn(effect = pure)]
fn same_member<T, Rt>(rt: &Rt, frame: &mut Rt::Frame<'_>, a: T, b: T) -> bool
where
    T: Monomorphize<(i64, f64)> + Carrier<Rt> + InstanceOf<eq<T, Rt>, Rt>,
    Rt: Runtime,
{
    T::call(&a, rt, frame, (&b,))
}

fn main() {}
