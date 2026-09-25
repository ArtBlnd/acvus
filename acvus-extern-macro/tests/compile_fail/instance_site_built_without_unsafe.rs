//! `CallSite::new` is the one way a requirement word reaches a call site,
//! and it is `unsafe`: its caller asserts the words are the entries chosen
//! for the requirements of the declaration it is handed to (RFC-0059
//! rule 8).
#![forbid(unsafe_code)]
use acvus_extern::{CallSite, Now, Owned, Arg, Required, Runtime, extern_signature};

extern_signature! { ns: "q", fn eq<T>(a: &T, b: &T) -> bool where T: Var<kind::Type>; }

type Forge<Rt> = Required<eq<Owned<Rt>, Rt>, Owned<Rt>, Now, 0>;

fn through_the_constructor<Rt>(words: &[Rt::Value])
where
    Rt: Runtime,
{
    let site = CallSite::<Rt>::new(&[], words);
    let _ = <Forge<Rt> as Arg<Rt>>::site(&site, 0);
}

fn main() {}
