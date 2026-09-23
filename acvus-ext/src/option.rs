//! `Option`'s pure methods under Rust `std`'s names and contracts.
//!
//! Every method takes the `Option` by value where Rust's takes `&self` or
//! `&mut self`. An `Option` is the value's own kind (RFC-0039), so its
//! storage is the runtime's value and not a Rust `Option<T>`: `Ref<Option<T>,
//! M, Rt>` has nothing to dereference, which is why `as_ref`, `take`,
//! `replace`, `insert`, `get_or_insert` and `get_or_insert_with` are not
//! here.
//!
//! `map` and `filter` are absent from this module and from `result`, and
//! both wait on `step_signature` in `acvus-mir/src/solver.rs`. It narrows an overloaded call by the
//! candidates the call shape still takes; where the receiver's own type is
//! not yet settled, two candidates that each take a closure settle only
//! after that closure has been typed from its body alone, and the type
//! which yields belongs to neither. Registered beside `iter::map`,
//! `option::map` refuses `acvus-ext/tests/e2e.rs`'s and
//! `examples/grades`'s `as_iter() | map(|p| -> p.x)` with ``no `map` takes
//! a call of type Fn(Iterator<&Pt{label: String, x: i64}, Pure>,
//! Fn({x: _}) -> _) -> Iterator<i64, Pure>``; registered beside
//! `iter::filter`, `option::filter` refuses
//! `acvus-interpreter-test/tests/extern_call_forms.rs`'s
//! `a | filter(|x| -> x > 1) | count` with ``no `filter` takes a call of
//! type Fn(_, Fn(i64) -> Bool) -> Iterator<_, Pure>``. Each was measured
//! one variable apart: unregistering that one name alone makes that one
//! test pass. `flatten`, which takes no closure, coexists, and
//! `acvus-interpreter-test/tests/option_methods.rs` pins that. Both rows
//! return when the solver pins a closure argument from the candidates
//! rather than after them.
//!
//! Not here either, each for a reason the framework states: `zip` and
//! `unzip` yield a tuple, and no tuple implements `Cross`; `unwrap_or_default`
//! needs a generic `default<T>`, which nothing declares; `copied` and
//! `cloned` need a handler to require `core::clone` of its own type
//! parameter, which a handler cannot state.

use acvus_extern::{
    Closure, ClosureFn, Cross, OneValue, Registry, Runtime, Var, extern_fn, extern_registry, kind,
};

use crate::iter::Items;
use acvus_extern::Ctx;

#[extern_fn(effect = pure)]
fn unwrap<T>(val: Option<T>) -> T
where
    T: Var<kind::Type>,
{
    val.expect("unwrap: called on None")
}

#[extern_fn(effect = pure)]
fn expect<T>(val: Option<T>, message: String) -> T
where
    T: Var<kind::Type>,
{
    let Some(inner) = val else {
        panic!("{message}")
    };
    inner
}

#[extern_fn(effect = pure)]
fn unwrap_or<T>(val: Option<T>, default: T) -> T
where
    T: Var<kind::Type>,
{
    val.unwrap_or(default)
}

#[extern_fn(effect = pure)]
fn is_some<T>(val: Option<T>) -> bool
where
    T: Var<kind::Type>,
{
    val.is_some()
}

#[extern_fn(effect = pure)]
fn is_none<T>(val: Option<T>) -> bool
where
    T: Var<kind::Type>,
{
    val.is_none()
}

#[extern_fn(effect = pure)]
fn and<T, U>(val: Option<T>, other: Option<U>) -> Option<U>
where
    T: Var<kind::Type>,
    U: Var<kind::Type>,
{
    val.and(other)
}

#[extern_fn(effect = pure)]
fn or<T>(val: Option<T>, other: Option<T>) -> Option<T>
where
    T: Var<kind::Type>,
{
    val.or(other)
}

#[extern_fn(effect = pure)]
fn xor<T>(val: Option<T>, other: Option<T>) -> Option<T>
where
    T: Var<kind::Type>,
{
    val.xor(other)
}

#[extern_fn(effect = pure)]
fn flatten<T>(val: Option<Option<T>>) -> Option<T>
where
    T: Var<kind::Type>,
{
    val.flatten()
}

#[extern_fn(effect = pure)]
fn ok_or<T, Er>(val: Option<T>, error: Er) -> Result<T, Er>
where
    T: Var<kind::Type>,
    Er: Var<kind::Type>,
{
    val.ok_or(error)
}

#[extern_fn(instance_of = crate::iter::sig::into_iter, effect = pure)]
fn into_iter_option<T, I, Rt>(val: Option<T>) -> Items<T, I, Rt>
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(val.into_iter().collect::<Vec<T>>())
}

fn is_some_and_now<T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    f: Closure<'_, (T,), bool, E, Rt>,
) -> bool
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call_now(ctx, (inner,)),
        None => false,
    }
}

#[extern_fn(effect = E, sync = is_some_and_now)]
async fn is_some_and<T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    f: Closure<'_, (T,), bool, E, Rt>,
) -> bool
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call(ctx, (inner,)).await,
        None => false,
    }
}

fn is_none_or_now<T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    f: Closure<'_, (T,), bool, E, Rt>,
) -> bool
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call_now(ctx, (inner,)),
        None => true,
    }
}

#[extern_fn(effect = E, sync = is_none_or_now)]
async fn is_none_or<T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    f: Closure<'_, (T,), bool, E, Rt>,
) -> bool
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call(ctx, (inner,)).await,
        None => true,
    }
}

fn unwrap_or_else_now<T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    f: Closure<'_, (), T, E, Rt>,
) -> T
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => inner,
        None => f.call_now(ctx, ()),
    }
}

#[extern_fn(effect = E, sync = unwrap_or_else_now)]
async fn unwrap_or_else<T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    f: Closure<'_, (), T, E, Rt>,
) -> T
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => inner,
        None => f.call(ctx, ()).await,
    }
}

fn map_or_now<T, U, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    default: U,
    f: Closure<'_, (T,), U, E, Rt>,
) -> U
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call_now(ctx, (inner,)),
        None => default,
    }
}

#[extern_fn(effect = E, sync = map_or_now)]
async fn map_or<T, U, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    default: U,
    f: Closure<'_, (T,), U, E, Rt>,
) -> U
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call(ctx, (inner,)).await,
        None => default,
    }
}

fn map_or_else_now<T, U, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    default: Closure<'_, (), U, E, Rt>,
    f: Closure<'_, (T,), U, E, Rt>,
) -> U
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call_now(ctx, (inner,)),
        None => default.call_now(ctx, ()),
    }
}

#[extern_fn(effect = E, sync = map_or_else_now)]
async fn map_or_else<T, U, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    default: Closure<'_, (), U, E, Rt>,
    f: Closure<'_, (T,), U, E, Rt>,
) -> U
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call(ctx, (inner,)).await,
        None => default.call(ctx, ()).await,
    }
}

fn and_then_now<T, U, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    f: Closure<'_, (T,), Option<U>, E, Rt>,
) -> Option<U>
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    val.and_then(|inner| f.call_now(ctx, (inner,)))
}

#[extern_fn(effect = E, sync = and_then_now)]
async fn and_then<T, U, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    f: Closure<'_, (T,), Option<U>, E, Rt>,
) -> Option<U>
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call(ctx, (inner,)).await,
        None => None,
    }
}

fn or_else_now<T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    f: Closure<'_, (), Option<T>, E, Rt>,
) -> Option<T>
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => Some(inner),
        None => f.call_now(ctx, ()),
    }
}

#[extern_fn(effect = E, sync = or_else_now)]
async fn or_else<T, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    f: Closure<'_, (), Option<T>, E, Rt>,
) -> Option<T>
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => Some(inner),
        None => f.call(ctx, ()).await,
    }
}

fn ok_or_else_now<T, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    f: Closure<'_, (), Er, E, Rt>,
) -> Result<T, Er>
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => Ok(inner),
        None => Err(f.call_now(ctx, ())),
    }
}

#[extern_fn(effect = E, sync = ok_or_else_now)]
async fn ok_or_else<T, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Option<T>,
    f: Closure<'_, (), Er, E, Rt>,
) -> Result<T, Er>
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => Ok(inner),
        None => Err(f.call(ctx, ()).await),
    }
}

pub fn option_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        fns: [
            unwrap, expect, unwrap_or, unwrap_or_else, is_some, is_none, is_some_and,
            is_none_or, map_or, map_or_else, and, and_then, or, or_else, xor,
            flatten, ok_or, ok_or_else, into_iter_option,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{Externs, Interner, TypesOnly};

    /// Eighteen entries for nineteen registered functions: `into_iter_option`
    /// is an instance of `iter::into_iter`, whose entry the baseline registry
    /// already holds.
    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg = Externs::combine(
            vec![
                crate::iterator_registry::<TypesOnly>(),
                option_registry::<TypesOnly>(),
            ],
            &i,
        )
        .expect("registry combines");
        let base = Externs::combine(vec![crate::iterator_registry::<TypesOnly>()], &i)
            .expect("the baseline combines");
        assert_eq!(reg.functions.len() - base.functions.len(), 18);
        assert_eq!(reg.handlers.len() - base.handlers.len(), 18);
    }
}
