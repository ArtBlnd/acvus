//! A signature's effect is one of its own effect variables (RFC-0067 rule 2).
use acvus_extern::extern_signature;

extern_signature! {
    ns: "q",
    effect = opaque,
    fn drain<S>(it: S) -> i64
    where
        S: acvus_extern::Var<acvus_extern::kind::Type>;
}

fn main() {}
