//! The functions on `Vec<T>`; the type itself is declared in `acvus_extern`.

use acvus_extern::{
    Arr, LenVar, Registry, Runtime, TyVar, extern_fn, extern_registry, extern_signature,
};

// A container demotes to a vec (RFC-0027).
extern_signature! {
    ns: "std",
    fn vec<C, T>(items: C) -> Vec<T>
    where
        C: TyVar,
        T: TyVar;
}

#[extern_fn(effect = pure)]
fn reverse<T>(mut items: Vec<T>) -> Vec<T>
where
    T: TyVar,
{
    items.reverse();
    items
}

#[extern_fn(instance_of = vec, effect = pure)]
#[extern_cast]
fn vec_array<T, N>(items: Arr<T, N>) -> Vec<T>
where
    T: TyVar,
    N: LenVar,
{
    items.0
}

pub fn vec_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        types: [Vec<_>],
        signatures: [vec],
        fns: [reverse, vec_array],
    }
}
