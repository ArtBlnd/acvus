//! The shared signatures the compiler names (RFC-0019, RFC-0020, RFC-0070 rule
//! 5).

use crate::{Registry, Runtime, extern_fn, extern_registry, extern_signature};

extern_signature! { ns: "core", fn clone<T>(a: &T) -> T where T: crate::Var<crate::kind::Type>; }
extern_signature! { ns: "core", fn eq<T>(a: &T, b: &T) -> bool where T: crate::Var<crate::kind::Type>; }

// An obligation across artifacts: an instance answers `-1`, `0` or `1`,
// and `acvus-mir` lowers `<`, `<=`, `>`, `>=` on an extension type to the
// sign of that answer. An `Ordering` type is not declared: this integer
// is the protocol `string::cmp`, `num::total_cmp` and the `sort_by`
// comparator already speak.
extern_signature! { ns: "core", fn cmp<T>(a: &T, b: &T) -> i64 where T: crate::Var<crate::kind::Type>; }

// An obligation across artifacts: two values an instance of `eq` holds
// equal hash equal, and the registry declaring the pair pins that with a
// test per type.
extern_signature! { ns: "core", fn hash<T>(a: &T) -> u64 where T: crate::Var<crate::kind::Type>; }

extern_signature! { ns: "core", fn to_string<T>(a: &T) -> String where T: crate::Var<crate::kind::Type>; }

/// An obligation across artifacts. `acvus-mir`'s `slice_coercion` takes
/// this declaration out of the environment's machine set to lower a
/// `&String` argument at a `&str` parameter (RFC-0062 rule 3), so the
/// registry that carries it is the registry in which a `&str` parameter is
/// reachable; it is core because the instruction is the language's
/// (RFC-0039).
#[extern_fn(effect = pure)]
#[extern_view]
fn as_str(s: &String) -> &str {
    s
}

pub fn core_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "core",
        signatures: [clone, eq, cmp, hash, to_string],
        fns: [as_str],
    }
}
