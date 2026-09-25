//! An `Instance` pairs a receiver with the word the checker chose for its
//! type at the site the receiver was passed to, and only the glue knows
//! that pairing. Code outside `acvus-extern` builds none: `own` is the
//! crate's (RFC-0068 rule 2).
#![forbid(unsafe_code)]
use acvus_extern::{Instance, Now, Owned, Pure, Runtime, extern_signature};

extern_signature! {
    ns: "q",
    effect = E,
    fn next<I, T, E, Rt>(it: &mut I) -> Option<T>
    where
        I: Var<kind::Type>,
        T: Var<kind::Type>,
        E: Var<kind::Effect>,
        Rt: Runtime;
}

type Pipeline<Rt> = Instance<'static, next<Owned<Rt>, i64, Pure, Rt>, Owned<Rt>, Rt, Now>;

fn through_the_constructor<Rt>(recv: Owned<Rt>, word: Rt::Value) -> Pipeline<Rt>
where
    Rt: Runtime,
{
    Pipeline::<Rt>::own(recv, word)
}

fn main() {}
