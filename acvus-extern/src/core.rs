//! The shared signatures the compiler names (RFC-0019, RFC-0020).

use crate::{Registry, Runtime, extern_fn, extern_registry, extern_signature};

extern_signature! { ns: "core", fn clone<T>(a: &T) -> T where T: crate::TyVar; }
extern_signature! { ns: "core", fn eq<T>(a: &T, b: &T) -> bool where T: crate::TyVar; }

#[extern_fn(instance_of = clone, effect = pure)]
fn clone_string<R>(_: &R, a: &String) -> String
where
    R: Runtime,
{
    a.clone()
}

#[extern_fn(instance_of = eq, effect = pure)]
fn eq_string<R>(_: &R, a: &String, b: &String) -> bool
where
    R: Runtime,
{
    a == b
}

pub fn core_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "core",
        signatures: [clone, eq],
        fns: [clone_string, eq_string],
    }
}
