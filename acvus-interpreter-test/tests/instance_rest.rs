//! RFC-0067 rule 6: a rest position standing at the signature's own
//! type variable.
//!
//! `eq(a: &T, b: &T)` is the shape the iterator spike never reached — its
//! `next` has an empty rest — and it is the shape where the instance's own
//! Rust types and the requiring handler's stand-in for `T` differ. The
//! position crosses as the runtime's value and the mono glue materializes
//! the instance's type back out of it, so `same` reads what the script
//! wrote rather than the stand-in's bytes.

use std::ops::Deref;
use std::sync::Arc;

use acvus_extern::{
    Ctx, Instance, Owned, Registry, Runtime, Var, extern_fn, extern_registry, kind,
};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::*;
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

mod sig {
    use acvus_extern::extern_signature;

    extern_signature! {
        ns: "rest",
        fn eq<T>(a: &T, b: &T) -> bool
        where
            T: Var<kind::Type>;
    }

    extern_signature! {
        ns: "rest",
        fn tally<T>(a: &T, b: Option<T>) -> i64
        where
            T: Var<kind::Type>;
    }
}

#[extern_fn(instance_of = sig::eq, effect = pure)]
fn eq_int(a: &i64, b: &i64) -> bool {
    a == b
}

#[extern_fn(instance_of = sig::eq, effect = pure)]
fn eq_text(a: &String, b: &String) -> bool {
    a == b
}

#[extern_fn(instance_of = sig::tally, effect = pure)]
fn tally_int(a: &i64, b: Option<i64>) -> i64 {
    match b {
        Some(x) => a + x,
        None => *a,
    }
}

/// The customer: a handler generic in `T` calling the `eq` of its `T`,
/// with the second argument standing at that same variable.
#[extern_fn(effect = pure)]
fn same<T, Rt>(ctx: &mut Ctx<'_, Rt>, a: T, b: T, eq: Instance<sig::eq<T, Rt>, T, Rt>) -> bool
where
    T: Var<kind::Type> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    let mut a = a;
    // SAFETY: `Externs::combine` met this parameter's requirement with the
    // instance of `rest::eq` at the ground type `T` was filled with, and
    // both `a` and `b` are values of that type.
    eq.call(ctx, &mut a, (&*b,))
}

/// The same crossing where the rest position is a pattern over the
/// variable: one value, `Value::some`'s own.
#[extern_fn(effect = pure)]
fn tally_of<T, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    a: T,
    b: Option<T>,
    tally: Instance<'_, sig::tally<T, Rt>, T, Rt>,
) -> i64
where
    T: Var<kind::Type> + Deref<Target = Rt::Value> + Into<Owned<Rt>>,
    Rt: Runtime,
{
    let mut a = a;
    let rest = match b {
        Some(x) => Owned::from_value(ctx.rt.some(x.into().into_value())),
        None => Owned::from_value(ctx.rt.none()),
    };
    // SAFETY: as `same`'s, at the option the signature writes there.
    tally.call(ctx, &mut a, (rest,))
}

fn rest_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "rest",
        signatures: [sig::eq, sig::tally],
        fns: [eq_int, eq_text, tally_int, same, tally_of],
    }
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut rs = acvus_ext::std_registries::<AcvusRuntime>();
    rs.push(rest_registry());
    rs
}

async fn run_i64_at(source: &str, opt: Opt) -> i64 {
    let i = Interner::new();
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse error"));
    let cr = check_source(
        &i,
        ast,
        &FxHashMap::default(),
        registries(),
        Ty::I64,
        opt,
        |_| {},
    )
    .unwrap_or_else(|refusal| panic!("compile failed: {}", refusal.messages.join("; ")));
    let (_, mut interp) = execute_compiled(
        &i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(acvus_interpreter::SequentialExecutor),
    );
    interp.execute().await.as_int()
}

const EQUAL_INTS: &str = "if same(7, 7) { 1 } else { 0 }";
const UNEQUAL_INTS: &str = "if same(7, 8) { 1 } else { 0 }";
const EQUAL_TEXT: &str = "if same(\"abc\".to_string(), \"abc\".to_string()) { 1 } else { 0 }";
const UNEQUAL_TEXT: &str = "if same(\"abc\".to_string(), \"abd\".to_string()) { 1 } else { 0 }";

/// The defect this file pins: before the rest position crossed uniformly,
/// `same(7, 7)` read the requiring handler's `Owned<Rt>` as an `i64` and
/// answered `false`.
#[tokio::test]
async fn a_rest_position_at_the_variable_reads_the_value_the_script_wrote() {
    for opt in [Opt::None, Opt::Full] {
        assert_eq!(
            run_i64_at(EQUAL_INTS, opt).await,
            1,
            "same(7, 7) at {opt:?}"
        );
        assert_eq!(
            run_i64_at(UNEQUAL_INTS, opt).await,
            0,
            "same(7, 8) at {opt:?}"
        );
    }
}

/// The same handler at a `Large` type: the instance's glue materializes a
/// `&String` from the same one value an `i64`'s materializes an `&i64`.
#[tokio::test]
async fn the_same_handler_serves_a_large_type() {
    for opt in [Opt::None, Opt::Full] {
        assert_eq!(
            run_i64_at(EQUAL_TEXT, opt).await,
            1,
            "equal text at {opt:?}"
        );
        assert_eq!(
            run_i64_at(UNEQUAL_TEXT, opt).await,
            0,
            "unequal text at {opt:?}"
        );
    }
}

const TALLY_SOME: &str = "tally_of(3, Some(4))";
const TALLY_NONE: &str = "tally_of(3, None)";

#[tokio::test]
async fn a_rest_position_that_is_a_pattern_over_the_variable_crosses_as_one_value() {
    for opt in [Opt::None, Opt::Full] {
        assert_eq!(run_i64_at(TALLY_SOME, opt).await, 7, "Some at {opt:?}");
        assert_eq!(run_i64_at(TALLY_NONE, opt).await, 3, "None at {opt:?}");
    }
}
