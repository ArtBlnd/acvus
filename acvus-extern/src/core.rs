//! The shared signatures the compiler names (RFC-0019, RFC-0020).

use crate::{Registry, Runtime, extern_registry, extern_signature};

extern_signature! { ns: "core", fn clone<T>(a: &T) -> T where T: crate::TyVar; }
extern_signature! { ns: "core", fn eq<T>(a: &T, b: &T) -> bool where T: crate::TyVar; }

// Every instance of `hash` is bound to the `eq` instance at the same type:
// two values `eq` holds equal hash equal. No structure here can hold that
// bond, because the instances live in other crates and each pair of them
// is two functions.
extern_signature! { ns: "core", fn hash<T>(a: &T) -> i64 where T: crate::TyVar; }

pub fn core_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "core",
        signatures: [clone, eq, hash],
        fns: [],
    }
}
