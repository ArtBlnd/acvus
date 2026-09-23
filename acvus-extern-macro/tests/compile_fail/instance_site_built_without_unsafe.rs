//! `CallSite::new` is the one way a requirement word reaches a call site,
//! and it is `unsafe`: its caller asserts the words are the entries chosen
//! for the requirements of the declaration it is handed to (RFC-0059
//! rule 8).
#![forbid(unsafe_code)]
use acvus_extern::{CallSite, Now, Owned, Pure, Required, Runtime, Sited, extern_signature};

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

type Forge<Rt> = Required<next<Owned<Rt>, i64, Pure, Rt>, Owned<Rt>, Now, 0>;

fn through_the_constructor<Rt>(words: &[Rt::Value])
where
    Rt: Runtime,
{
    let site = CallSite::<Rt>::new(&[], words);
    let _ = <Forge<Rt> as Sited<Rt>>::site(&site, 0);
}

fn main() {}
