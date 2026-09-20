//! The shared signatures the compiler names (RFC-0019, RFC-0020).

use crate::str::StrView;
use crate::{Registry, Runtime, extern_fn, extern_registry, extern_signature};

extern_signature! { ns: "core", fn clone<T>(a: &T) -> T where T: crate::Var<crate::kind::Type>; }
extern_signature! { ns: "core", fn eq<T>(a: &T, b: &T) -> bool where T: crate::Var<crate::kind::Type>; }

// Every instance of `hash` is bound to the `eq` instance at the same type:
// two values `eq` holds equal hash equal. No structure here can hold that
// bond, because the instances live in other crates and each pair of them
// is two functions.
extern_signature! { ns: "core", fn hash<T>(a: &T) -> i64 where T: crate::Var<crate::kind::Type>; }

/// An obligation across artifacts. `acvus-mir`'s `slice_coercion` takes
/// this declaration out of the environment's machine set to lower a
/// `&String` argument at a `&str` parameter (RFC-0062 Decision 3), so the
/// registry that carries it is the registry in which a `&str` parameter is
/// reachable; it is core because the instruction is the language's
/// (RFC-0039).
#[extern_fn(effect = pure)]
#[extern_view]
fn as_str(s: &String) -> StrView {
    StrView::of(s)
}

pub fn core_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "core",
        signatures: [clone, eq, hash],
        fns: [as_str],
    }
}
