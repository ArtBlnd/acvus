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
//! that is a decision two measurements forced. A method whose name `Iter`
//! also carries and which takes a closure cannot be a second entry: with
//! `map` registered in any namespace and in either registration order,
//! `examples/grades`'s `@students.as_iter() | map(|s| -> s.score)` stops
//! compiling, and with `filter` registered, `a | filter(|x| -> x > 1)` in
//! `acvus-interpreter-test/tests/extern_call_forms.rs` stops compiling.
//! `step_signature` in `acvus-mir/src/solver.rs` narrows an overloaded call
//! by the candidates the call shape still takes: one candidate settles at
//! once and hands the closure its parameter type, two settle only after the
//! closure has been typed from its body alone, and the type that yields
//! belongs to neither. Folding the three `map`s into one
//! `extern_signature!` does not reach it — an instance whose return
//! constructor differs from the signature's is refused by
//! `Externs::combine`. `flatten`, which takes no closure, coexists. Both
//! rows return when the solver pins a closure argument from the candidates
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

use crate::iter::Iter;

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

#[extern_fn(instance_of = crate::iterator::sig::into_iter, effect = pure)]
#[extern_cast]
fn into_iter_option<T, E, I, Rt>(val: Option<T>) -> Iter<T, E, I, Rt>
where
    T: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Iter::from_items(val.into_iter().collect::<Vec<T>>())
}

fn is_some_and_now<T, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    f: Closure<(T,), bool, E, Rt>,
) -> bool
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call_now(rt, frame, (inner,)),
        None => false,
    }
}

#[extern_fn(effect = E, sync = is_some_and_now)]
async fn is_some_and<T, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    f: Closure<(T,), bool, E, Rt>,
) -> bool
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call(rt, frame, (inner,)).await,
        None => false,
    }
}

fn is_none_or_now<T, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    f: Closure<(T,), bool, E, Rt>,
) -> bool
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call_now(rt, frame, (inner,)),
        None => true,
    }
}

#[extern_fn(effect = E, sync = is_none_or_now)]
async fn is_none_or<T, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    f: Closure<(T,), bool, E, Rt>,
) -> bool
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call(rt, frame, (inner,)).await,
        None => true,
    }
}

fn unwrap_or_else_now<T, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    f: Closure<(), T, E, Rt>,
) -> T
where
    T: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => inner,
        None => f.call_now(rt, frame, ()),
    }
}

#[extern_fn(effect = E, sync = unwrap_or_else_now)]
async fn unwrap_or_else<T, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    f: Closure<(), T, E, Rt>,
) -> T
where
    T: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => inner,
        None => f.call(rt, frame, ()).await,
    }
}

fn map_or_now<T, U, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    default: U,
    f: Closure<(T,), U, E, Rt>,
) -> U
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt>,
    U: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call_now(rt, frame, (inner,)),
        None => default,
    }
}

#[extern_fn(effect = E, sync = map_or_now)]
async fn map_or<T, U, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    default: U,
    f: Closure<(T,), U, E, Rt>,
) -> U
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt>,
    U: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call(rt, frame, (inner,)).await,
        None => default,
    }
}

fn map_or_else_now<T, U, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    default: Closure<(), U, E, Rt>,
    f: Closure<(T,), U, E, Rt>,
) -> U
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt>,
    U: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call_now(rt, frame, (inner,)),
        None => default.call_now(rt, frame, ()),
    }
}

#[extern_fn(effect = E, sync = map_or_else_now)]
async fn map_or_else<T, U, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    default: Closure<(), U, E, Rt>,
    f: Closure<(T,), U, E, Rt>,
) -> U
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt>,
    U: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call(rt, frame, (inner,)).await,
        None => default.call(rt, frame, ()).await,
    }
}

fn and_then_now<T, U, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    f: Closure<(T,), Option<U>, E, Rt>,
) -> Option<U>
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt>,
    U: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    val.and_then(|inner| f.call_now(rt, frame, (inner,)))
}

#[extern_fn(effect = E, sync = and_then_now)]
async fn and_then<T, U, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    f: Closure<(T,), Option<U>, E, Rt>,
) -> Option<U>
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt>,
    U: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => f.call(rt, frame, (inner,)).await,
        None => None,
    }
}

fn or_else_now<T, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    f: Closure<(), Option<T>, E, Rt>,
) -> Option<T>
where
    T: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => Some(inner),
        None => f.call_now(rt, frame, ()),
    }
}

#[extern_fn(effect = E, sync = or_else_now)]
async fn or_else<T, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    f: Closure<(), Option<T>, E, Rt>,
) -> Option<T>
where
    T: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => Some(inner),
        None => f.call(rt, frame, ()).await,
    }
}

fn ok_or_else_now<T, Er, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    f: Closure<(), Er, E, Rt>,
) -> Result<T, Er>
where
    T: Var<kind::Type> + OneValue<Rt>,
    Er: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => Ok(inner),
        None => Err(f.call_now(rt, frame, ())),
    }
}

#[extern_fn(effect = E, sync = ok_or_else_now)]
async fn ok_or_else<T, Er, E, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    val: Option<T>,
    f: Closure<(), Er, E, Rt>,
) -> Result<T, Er>
where
    T: Var<kind::Type> + OneValue<Rt>,
    Er: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Some(inner) => Ok(inner),
        None => Err(f.call(rt, frame, ()).await),
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
