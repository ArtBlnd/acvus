//! `Result`'s pure methods under Rust `std`'s names and contracts, in the
//! `result` namespace: a `Result` method and an `Option` method of the same
//! name are two entries, resolved by the argument's type, exactly as
//! `string::len` and `vec::len` are.
//!
//! Every method takes the `Result` by value where Rust's takes `&self`. Rust
//! renders the other side's payload into the panic text of `unwrap`,
//! `unwrap_err` and their kin through `Debug`; a language value has no
//! `Debug`, so the text names the method and the arm it met.
//!
//! `map` is absent for the reason the `option` module states: a method whose
//! name `Iter` also carries and which takes a closure cannot be a second
//! entry.
//!
//! Not here, each for a reason the framework states: `unwrap_or_default`
//! needs a generic `default<T>`, which nothing declares; `copied` and
//! `cloned` need a handler to require `core::clone` of its own type
//! parameter, which a handler cannot state.

use acvus_extern::{
    Closure, ClosureFn, Cross, OneValue, Registry, Runtime, Var, extern_fn, extern_registry, kind,
};

use crate::iter::Items;
use acvus_extern::Ctx;

#[extern_fn(effect = pure)]
fn is_ok<T, Er>(val: Result<T, Er>) -> bool
where
    T: Var<kind::Type>,
    Er: Var<kind::Type>,
{
    val.is_ok()
}

#[extern_fn(effect = pure)]
fn is_err<T, Er>(val: Result<T, Er>) -> bool
where
    T: Var<kind::Type>,
    Er: Var<kind::Type>,
{
    val.is_err()
}

#[extern_fn(effect = pure)]
fn ok<T, Er>(val: Result<T, Er>) -> Option<T>
where
    T: Var<kind::Type>,
    Er: Var<kind::Type>,
{
    val.ok()
}

#[extern_fn(effect = pure)]
fn err<T, Er>(val: Result<T, Er>) -> Option<Er>
where
    T: Var<kind::Type>,
    Er: Var<kind::Type>,
{
    val.err()
}

#[extern_fn(effect = pure)]
fn unwrap<T, Er>(val: Result<T, Er>) -> T
where
    T: Var<kind::Type>,
    Er: Var<kind::Type>,
{
    let Ok(inner) = val else {
        panic!("unwrap: called on Err")
    };
    inner
}

#[extern_fn(effect = pure)]
fn unwrap_err<T, Er>(val: Result<T, Er>) -> Er
where
    T: Var<kind::Type>,
    Er: Var<kind::Type>,
{
    let Err(error) = val else {
        panic!("unwrap_err: called on Ok")
    };
    error
}

#[extern_fn(effect = pure)]
fn expect<T, Er>(val: Result<T, Er>, message: String) -> T
where
    T: Var<kind::Type>,
    Er: Var<kind::Type>,
{
    let Ok(inner) = val else { panic!("{message}") };
    inner
}

#[extern_fn(effect = pure)]
fn expect_err<T, Er>(val: Result<T, Er>, message: String) -> Er
where
    T: Var<kind::Type>,
    Er: Var<kind::Type>,
{
    let Err(error) = val else { panic!("{message}") };
    error
}

#[extern_fn(effect = pure)]
fn unwrap_or<T, Er>(val: Result<T, Er>, default: T) -> T
where
    T: Var<kind::Type>,
    Er: Var<kind::Type>,
{
    val.unwrap_or(default)
}

#[extern_fn(effect = pure)]
fn and<T, U, Er>(val: Result<T, Er>, other: Result<U, Er>) -> Result<U, Er>
where
    T: Var<kind::Type>,
    U: Var<kind::Type>,
    Er: Var<kind::Type>,
{
    val.and(other)
}

#[extern_fn(effect = pure)]
fn or<T, Er, F>(val: Result<T, Er>, other: Result<T, F>) -> Result<T, F>
where
    T: Var<kind::Type>,
    Er: Var<kind::Type>,
    F: Var<kind::Type>,
{
    val.or(other)
}

#[extern_fn(effect = pure)]
fn flatten<T, Er>(val: Result<Result<T, Er>, Er>) -> Result<T, Er>
where
    T: Var<kind::Type>,
    Er: Var<kind::Type>,
{
    val.and_then(|inner| inner)
}

#[extern_fn(effect = pure)]
fn transpose<T, Er>(val: Result<Option<T>, Er>) -> Option<Result<T, Er>>
where
    T: Var<kind::Type>,
    Er: Var<kind::Type>,
{
    val.transpose()
}

#[extern_fn(instance_of = crate::iter::sig::into_iter, effect = pure)]
fn into_iter_result<T, Er, I, Rt>(val: Result<T, Er>) -> Items<T, I, Rt>
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(val.into_iter().collect::<Vec<T>>())
}

fn is_ok_and_now<T, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    f: Closure<'_, (T,), bool, E, Rt>,
) -> bool
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    Er: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Ok(inner) => f.call_now(ctx, (inner,)),
        Err(_) => false,
    }
}

#[extern_fn(effect = E, sync = is_ok_and_now)]
async fn is_ok_and<T, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    f: Closure<'_, (T,), bool, E, Rt>,
) -> bool
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    Er: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Ok(inner) => f.call(ctx, (inner,)).await,
        Err(_) => false,
    }
}

fn is_err_and_now<T, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    f: Closure<'_, (Er,), bool, E, Rt>,
) -> bool
where
    T: Var<kind::Type>,
    Er: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Ok(_) => false,
        Err(error) => f.call_now(ctx, (error,)),
    }
}

#[extern_fn(effect = E, sync = is_err_and_now)]
async fn is_err_and<T, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    f: Closure<'_, (Er,), bool, E, Rt>,
) -> bool
where
    T: Var<kind::Type>,
    Er: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Ok(_) => false,
        Err(error) => f.call(ctx, (error,)).await,
    }
}

fn map_err_now<T, Er, F, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    f: Closure<'_, (Er,), F, E, Rt>,
) -> Result<T, F>
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    F: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    val.map_err(|error| f.call_now(ctx, (error,)))
}

#[extern_fn(effect = E, sync = map_err_now)]
async fn map_err<T, Er, F, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    f: Closure<'_, (Er,), F, E, Rt>,
) -> Result<T, F>
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    F: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Ok(inner) => Ok(inner),
        Err(error) => Err(f.call(ctx, (error,)).await),
    }
}

fn map_or_now<T, U, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    default: U,
    f: Closure<'_, (T,), U, E, Rt>,
) -> U
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Ok(inner) => f.call_now(ctx, (inner,)),
        Err(_) => default,
    }
}

#[extern_fn(effect = E, sync = map_or_now)]
async fn map_or<T, U, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    default: U,
    f: Closure<'_, (T,), U, E, Rt>,
) -> U
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Ok(inner) => f.call(ctx, (inner,)).await,
        Err(_) => default,
    }
}

fn map_or_else_now<T, U, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    default: Closure<'_, (Er,), U, E, Rt>,
    f: Closure<'_, (T,), U, E, Rt>,
) -> U
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Ok(inner) => f.call_now(ctx, (inner,)),
        Err(error) => default.call_now(ctx, (error,)),
    }
}

#[extern_fn(effect = E, sync = map_or_else_now)]
async fn map_or_else<T, U, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    default: Closure<'_, (Er,), U, E, Rt>,
    f: Closure<'_, (T,), U, E, Rt>,
) -> U
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Ok(inner) => f.call(ctx, (inner,)).await,
        Err(error) => default.call(ctx, (error,)).await,
    }
}

fn and_then_now<T, U, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    f: Closure<'_, (T,), Result<U, Er>, E, Rt>,
) -> Result<U, Er>
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    val.and_then(|inner| f.call_now(ctx, (inner,)))
}

#[extern_fn(effect = E, sync = and_then_now)]
async fn and_then<T, U, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    f: Closure<'_, (T,), Result<U, Er>, E, Rt>,
) -> Result<U, Er>
where
    T: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    U: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Ok(inner) => f.call(ctx, (inner,)).await,
        Err(error) => Err(error),
    }
}

fn or_else_now<T, Er, F, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    f: Closure<'_, (Er,), Result<T, F>, E, Rt>,
) -> Result<T, F>
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    F: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    val.or_else(|error| f.call_now(ctx, (error,)))
}

#[extern_fn(effect = E, sync = or_else_now)]
async fn or_else<T, Er, F, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    f: Closure<'_, (Er,), Result<T, F>, E, Rt>,
) -> Result<T, F>
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    F: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Ok(inner) => Ok(inner),
        Err(error) => f.call(ctx, (error,)).await,
    }
}

fn unwrap_or_else_now<T, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    f: Closure<'_, (Er,), T, E, Rt>,
) -> T
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    val.unwrap_or_else(|error| f.call_now(ctx, (error,)))
}

#[extern_fn(effect = E, sync = unwrap_or_else_now)]
async fn unwrap_or_else<T, Er, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    val: Result<T, Er>,
    f: Closure<'_, (Er,), T, E, Rt>,
) -> T
where
    T: Var<kind::Type> + OneValue<Rt> + acvus_extern::Unbranded,
    Er: Var<kind::Type> + OneValue<Rt> + Cross<Rt> + acvus_extern::PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    match val {
        Ok(inner) => inner,
        Err(error) => f.call(ctx, (error,)).await,
    }
}

pub fn result_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "result",
        fns: [
            is_ok, is_err, is_ok_and, is_err_and, ok, err, unwrap, unwrap_err, unwrap_or,
            unwrap_or_else, expect, expect_err, map_err, map_or, map_or_else, and,
            and_then, or, or_else, flatten, transpose, into_iter_result,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{Externs, Interner, TypesOnly};

    /// Twenty-one entries for twenty-two registered functions:
    /// `into_iter_result` is an instance of `iter::into_iter`, whose entry
    /// the baseline registry already holds.
    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg = Externs::combine(
            vec![
                crate::iterator_registry::<TypesOnly>(),
                result_registry::<TypesOnly>(),
            ],
            &i,
        )
        .expect("registry combines");
        let base = Externs::combine(vec![crate::iterator_registry::<TypesOnly>()], &i)
            .expect("the baseline combines");
        assert_eq!(reg.functions.len() - base.functions.len(), 21);
        assert_eq!(reg.handlers.len() - base.handlers.len(), 21);
    }
}
