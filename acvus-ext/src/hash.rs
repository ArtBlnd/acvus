use std::hash::{DefaultHasher, Hash, Hasher};

use acvus_extern::{Registry, Runtime, extern_fn, extern_registry};

#[extern_fn(instance_of = acvus_extern::core::hash, effect = pure)]
fn hash_int(a: &i64) -> i64 {
    *a
}

/// The interpreter's `==` on a `Float` compares bit patterns
/// (`acvus-interpreter/src/interpreter.rs`, `BinOp::Eq`), so `0.0` and
/// `-0.0` are unequal there and hash apart here. A hash over the numeric
/// value would break the bond `core::hash` states.
#[extern_fn(instance_of = acvus_extern::core::hash, effect = pure)]
fn hash_float(a: &f64) -> i64 {
    i64::from_ne_bytes(a.to_bits().to_ne_bytes())
}

#[extern_fn(instance_of = acvus_extern::core::hash, effect = pure)]
fn hash_bool(a: &bool) -> i64 {
    i64::from(*a)
}

#[extern_fn(instance_of = acvus_extern::core::hash, effect = pure)]
fn hash_string(a: &String) -> i64 {
    let mut hasher = DefaultHasher::new();
    a.hash(&mut hasher);
    i64::from_ne_bytes(hasher.finish().to_ne_bytes())
}

pub fn hash_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "core",
        fns: [hash_int, hash_float, hash_bool, hash_string],
    }
}
