//! Iterator builtins (sync/lazy constructors and combinators).
//!
//! Async iterator consumers (find, reduce, fold, any, all, collect, next,
//! first_iter, last_iter, contains_iter, join, join_iter, next_seq)
//! remain in the interpreter.

use std::marker::PhantomData;

use acvus_mir::builtins::BuiltinId;

use super::handler::{sync, BuiltinExecute};
use super::types::{E, Fun1, Iter, T};
use crate::iter::IterHandle;
use crate::value::{TypedValue, Value};

// ── Lazy combinators (Iter → Iter) ─────────────────────────────────

fn map_iter(
    iter: Iter<T<0>, E<0>>,
    f: Fun1<T<0>, T<1>, E<0>>,
) -> Iter<T<1>, E<0>> {
    Iter(iter.0.map(f.0), PhantomData)
}

fn pmap_iter(
    iter: Iter<T<0>, E<0>>,
    f: Fun1<T<0>, T<1>, E<0>>,
) -> Iter<T<1>, E<0>> {
    // pmap has same runtime behavior as map; parallelism is a scheduling concern.
    Iter(iter.0.map(f.0), PhantomData)
}

fn filter_iter(
    iter: Iter<T<0>, E<0>>,
    f: Fun1<T<0>, bool, E<0>>,
) -> Iter<T<0>, E<0>> {
    Iter(iter.0.filter(f.0), PhantomData)
}

fn take_iter(iter: Iter<T<0>, E<0>>, n: i64) -> Iter<T<0>, E<0>> {
    Iter(iter.0.take(n.max(0) as usize), PhantomData)
}

fn skip_iter(iter: Iter<T<0>, E<0>>, n: i64) -> Iter<T<0>, E<0>> {
    Iter(iter.0.skip(n.max(0) as usize), PhantomData)
}

fn chain_iter(
    a: Iter<T<0>, E<0>>,
    b: Iter<T<0>, E<0>>,
) -> Iter<T<0>, E<0>> {
    Iter(a.0.chain(b.0), PhantomData)
}

fn flatten_iter(iter: Iter<T<0>, E<0>>) -> Iter<T<0>, E<0>> {
    Iter(iter.0.flatten(), PhantomData)
}

fn flat_map_fn(
    iter: Iter<T<0>, E<0>>,
    f: Fun1<T<0>, T<1>, E<0>>,
) -> Iter<T<1>, E<0>> {
    Iter(iter.0.flat_map(f.0), PhantomData)
}

// ── Constructors (List → Iter) — polymorphic, manual ───────────────

fn iter_from_list(args: Vec<TypedValue>) -> Result<TypedValue, crate::error::RuntimeError> {
    use super::types::FromTyped;
    use super::types::IntoTyped;
    use acvus_mir::ty::Effect;

    let mut it = args.into_iter();
    let tv = it.next().expect("missing arg 0");
    let effect = match tv.ty() {
        acvus_mir::ty::Ty::Iterator(_, e) => *e,
        _ => Effect::Pure,
    };
    let items = Vec::<Value>::from_typed(tv)?;
    Ok(IterHandle::from_list(items, effect).into_typed())
}

fn rev_iter_from_list(args: Vec<TypedValue>) -> Result<TypedValue, crate::error::RuntimeError> {
    use super::types::FromTyped;
    use super::types::IntoTyped;
    use acvus_mir::ty::Effect;

    let mut it = args.into_iter();
    let tv = it.next().expect("missing arg 0");
    let effect = match tv.ty() {
        acvus_mir::ty::Ty::Iterator(_, e) => *e,
        _ => Effect::Pure,
    };
    let mut items = Vec::<Value>::from_typed(tv)?;
    items.reverse();
    Ok(IterHandle::from_list(items, effect).into_typed())
}

// ── Registration ───────────────────────────────────────────────────

pub fn entries() -> Vec<(BuiltinId, BuiltinExecute)> {
    vec![
        // Constructors (polymorphic — manual)
        (BuiltinId::Iter,         Box::new(iter_from_list)),
        (BuiltinId::RevIter,      Box::new(rev_iter_from_list)),
        // Lazy combinators (typed — sync)
        (BuiltinId::Map,          sync(map_iter)),
        (BuiltinId::Pmap,         sync(pmap_iter)),
        (BuiltinId::Filter,       sync(filter_iter)),
        (BuiltinId::Take,         sync(take_iter)),
        (BuiltinId::Skip,         sync(skip_iter)),
        (BuiltinId::Chain,        sync(chain_iter)),
        (BuiltinId::Flatten,      sync(flatten_iter)),
        (BuiltinId::FlatMap,      sync(flat_map_fn)),
    ]
}
