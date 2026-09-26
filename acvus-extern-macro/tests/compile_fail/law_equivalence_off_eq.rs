//! `law(equivalence)` is stated on a `core::eq` instance (RFC-0082 rule 11):
//! an instance of another signature of the same shape is refused.
use acvus_extern::{extern_fn, extern_signature};

extern_signature! { ns: "t", fn same<T>(a: &T, b: &T) -> bool where T: acvus_extern::Var<acvus_extern::kind::Type>; }

#[extern_fn(instance_of = same, effect = pure, law(equivalence))]
fn same_int(a: &i64, b: &i64) -> bool {
    a == b
}

fn main() {}
