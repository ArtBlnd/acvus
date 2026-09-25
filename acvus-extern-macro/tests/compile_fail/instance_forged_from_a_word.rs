//! A `Required` parameter's site makes an `InstanceOf` out of the word its
//! call site holds, and calling the instance runs whatever that word
//! addresses. The words are `prepare`'s, so safe code builds no call site
//! that holds them, whether a bare word or one `instance_value` made from an
//! entry of its own (RFC-0080 rule 2).
#![forbid(unsafe_code)]
use acvus_extern::{
    CallSite, InstanceEntry, InstanceRun, Now, Owned, Required, Runtime, Arg,
    extern_signature,
};

extern_signature! { ns: "q", fn eq<T>(a: &T, b: &T) -> bool where T: Var<kind::Type>; }

type Forge<Rt> = Required<eq<Owned<Rt>, Rt>, Owned<Rt>, Now, 0>;

fn from_a_word<Rt>(words: &[Rt::Value])
where
    Rt: Runtime,
{
    let site = CallSite::<Rt> {
        args: &[],
        ret: None,
        requires: words,
    };
    let _ = <Forge<Rt> as Arg<Rt>>::site(&site, 0);
}

fn from_an_entry<Rt>(run: InstanceRun)
where
    Rt: Runtime,
{
    let entry = InstanceEntry::<Rt> {
        run,
        requires: Box::new([Rt::Value::default()]),
    };
    let words = [Rt::instance_value(&entry)];
    let site = CallSite::<Rt> {
        args: &[],
        ret: None,
        requires: &words,
    };
    let _ = <Forge<Rt> as Arg<Rt>>::site(&site, 0);
}

fn main() {}
