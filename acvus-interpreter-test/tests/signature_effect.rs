//! RFC-0067 rule 2: a shared signature declares its effect, so a consumer generic
//! in its effect can be one signature with an instance per input length.
//!
//! `Pipe` stands in for `Iter`, its first parameter for the list of element
//! types the pipeline passed through as nested pairs, `step` for every
//! adaptor, `drain` for every consumer that takes a closure, and `tally`
//! for every consumer that takes none.

use std::marker::PhantomData;
use std::sync::Arc;

use acvus_extern::{
    Closure, Erased, ExternType, Registry, Runtime, Var, extern_fn, extern_registry, kind,
};
use acvus_interpreter::code::Body;
use acvus_interpreter::{AcvusRuntime, PrepareCtx, prepare_module};
use acvus_interpreter_test::*;
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

#[derive(ExternType)]
#[extern_type(name = "Pipe")]
#[repr(transparent)]
pub struct Pipe<Ts, O, E, I, Rt>(i64, PhantomData<(Ts, O, E, I, Rt)>)
where
    Ts: Var<kind::Type>,
    O: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime;

impl<Ts, O, E, I, Rt> Pipe<Ts, O, E, I, Rt>
where
    Ts: Var<kind::Type>,
    O: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    fn stages(&self) -> i64 {
        self.0
    }
}

#[extern_fn(effect = pure)]
fn src<E, I, Rt>() -> Pipe<(), Erased<Rt, i64>, E, I, Rt>
where
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Pipe(0, PhantomData)
}

/// Not pure, so `optimize::spawn_split` turns a call of it into a `Spawn`
/// and an `Eval`: a closure whose body calls it suspends.
#[extern_fn(effect = opaque)]
fn seen(x: i64) -> i64 {
    x
}

mod sig {
    use acvus_extern::{Closure, extern_signature};

    use super::Pipe;

    extern_signature! {
        ns: "q",
        fn step<S, R, T, U, E, Rt>(it: S, f: Closure<'_, (T,), U, E, Rt>) -> R
        where
            S: Var<kind::Type>,
            R: Var<kind::Type>,
            T: Var<kind::Type>,
            U: Var<kind::Type>,
            E: Var<kind::Effect>,
            Rt: Runtime;
    }

    extern_signature! {
        ns: "q",
        effect = E,
        fn drain<S, T, U, E, I, Rt>(it: S, f: Closure<'_, (T,), U, E, Rt>) -> i64
        where
            S: Var<kind::Type>,
            T: Var<kind::Type>,
            U: Var<kind::Type>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime;
    }

    extern_signature! {
        ns: "q",
        effect = E,
        fn tally<Ts, O, E, I, Rt>(it: Pipe<Ts, O, E, I, Rt>) -> i64
        where
            Ts: Var<kind::Type> + Chosen,
            O: Var<kind::Type>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime;
    }

    extern_signature! {
        ns: "q",
        effect = E,
        fn bare_tally<S, E, I, Rt>(it: S) -> i64
        where
            S: Var<kind::Type>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime;
    }
}

/// The bodies do nothing with the closure and only count the stage, as the
/// answer this file reads is the length the types carried, not a computation
/// over elements.
macro_rules! step_instance {
    ($name:ident, [$($v:ident),*], $($ts:tt)+) => {
        #[extern_fn(instance_of = sig::step, effect = pure)]
        fn $name<$($v,)* T, U, E, I, Rt>(
            it: Pipe<$($ts)+, T, E, I, Rt>,
            f: Closure<'_, (T,), U, E, Rt>,
        ) -> Pipe<(T, $($ts)+), U, E, I, Rt>
        where
            $($v: Var<kind::Type>,)*
            T: Var<kind::Type>,
            U: Var<kind::Type>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime,
        {
            drop(f);
            Pipe(it.stages() + 1, PhantomData)
        }
    };
}

step_instance!(step_0, [], ());
step_instance!(step_1, [A], (A, ()));
step_instance!(step_2, [A, B], (A, (B, ())));

macro_rules! drain_instance {
    ($name:ident, $now:ident, [$($v:ident),*], $($ts:tt)+) => {
        fn $now<$($v,)* T, U, E, I, Rt>(
            it: Pipe<$($ts)+, T, E, I, Rt>,
            f: Closure<'_, (T,), U, E, Rt>,
        ) -> i64
        where
            $($v: Var<kind::Type>,)*
            T: Var<kind::Type>,
            U: Var<kind::Type>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime,
        {
            drop(f);
            it.stages()
        }

        #[extern_fn(instance_of = sig::drain, effect = E, sync = $now)]
        async fn $name<$($v,)* T, U, E, I, Rt>(
            it: Pipe<$($ts)+, T, E, I, Rt>,
            f: Closure<'_, (T,), U, E, Rt>,
        ) -> i64
        where
            $($v: Var<kind::Type>,)*
            T: Var<kind::Type>,
            U: Var<kind::Type>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime,
        {
            drop(f);
            it.stages()
        }
    };
}

drain_instance!(drain_0, drain_0_now, [], ());
drain_instance!(drain_1, drain_1_now, [A], (A, ()));

#[extern_fn(instance_of = sig::drain, effect = pure)]
fn drain_2<A, B, T, U, E, I, Rt>(
    it: Pipe<(A, (B, ())), T, E, I, Rt>,
    f: Closure<'_, (T,), U, E, Rt>,
) -> i64
where
    A: Var<kind::Type>,
    B: Var<kind::Type>,
    T: Var<kind::Type>,
    U: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    drop(f);
    it.stages()
}

macro_rules! tally_instance {
    ($name:ident, $now:ident, $sig:ident, [$($v:ident),*], $($ts:tt)+) => {
        fn $now<$($v,)* T, E, I, Rt>(it: Pipe<$($ts)+, T, E, I, Rt>) -> i64
        where
            $($v: Var<kind::Type>,)*
            T: Var<kind::Type>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime,
        {
            it.stages()
        }

        #[extern_fn(instance_of = sig::$sig, effect = E, sync = $now)]
        async fn $name<$($v,)* T, E, I, Rt>(it: Pipe<$($ts)+, T, E, I, Rt>) -> i64
        where
            $($v: Var<kind::Type>,)*
            T: Var<kind::Type>,
            E: Var<kind::Effect>,
            I: Var<kind::Identity>,
            Rt: Runtime,
        {
            it.stages()
        }
    };
}

tally_instance!(tally_0, tally_0_now, tally, [], ());
tally_instance!(tally_1, tally_1_now, tally, [A], (A, ()));
tally_instance!(bare_tally_0, bare_tally_0_now, bare_tally, [], ());
tally_instance!(bare_tally_1, bare_tally_1_now, bare_tally, [A], (A, ()));

fn depth_now<A, O, E, I, Rt>(it: Pipe<(A, ()), O, E, I, Rt>) -> i64
where
    A: Var<kind::Type>,
    O: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    it.stages()
}

/// A plain declaration at the one length its programs build: a pipeline's
/// stage list is its box's Rust type, so a declaration reads one length, and
/// the per-length reading is a signature's (`tally`).
#[extern_fn(effect = E, sync = depth_now)]
async fn depth<A, O, E, I, Rt>(it: Pipe<(A, ()), O, E, I, Rt>) -> i64
where
    A: Var<kind::Type>,
    O: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    it.stages()
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "q",
        types: [Pipe<_, _, _, _, AcvusRuntime>],
        signatures: [sig::step, sig::drain, sig::tally, sig::bare_tally],
        fns: [
            src, seen, depth,
            tally_0, tally_1, bare_tally_0, bare_tally_1,
            step_0, step_1, step_2,
            drain_0, drain_1, drain_2,
        ],
    }
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut rs = acvus_ext::std_registries::<AcvusRuntime>();
    rs.push(registry());
    rs
}

async fn run_i64(source: &str) -> i64 {
    let i = Interner::new();
    run_script_mode_with_externs(&i, source, Context::default(), registries(), Ty::I64)
        .await
        .value
        .as_int()
}

fn prepared_entry(source: &str) -> Body {
    let i = Interner::new();
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse error"));
    let cr = compile_source_with_externs(&i, ast, &FxHashMap::default(), registries(), Ty::I64);
    let module = cr.modules.get(&cr.entry_qref).expect("the entry module");
    let ctx = PrepareCtx {
        interner: &i,
        externs: &cr.extern_executables,
        context_names: &cr.context_names,
        instances: &cr.instances,
        access: acvus_mir::graph::Access::Sync,
    };
    let prepared = prepare_module(module, &ctx)
        .unwrap_or_else(|refused| panic!("the body is refused: {refused}"));
    Arc::try_unwrap(prepared.main).unwrap_or_else(|_| panic!("one reference to main"))
}

struct AtLength {
    source: &'static str,
    stages: i64,
}

const fn at_length(source: &'static str, stages: i64) -> AtLength {
    AtLength { source, stages }
}

const EVERY_LENGTH: [AtLength; 3] = [
    at_length("src() | drain(|x| -> x + 1)", 0),
    at_length("src() | step(|x| -> x + 1) | drain(|x| -> x + 1)", 1),
    at_length(
        "src() | step(|x| -> x + 1) | step(|x| -> x + 1) | drain(|x| -> x + 1)",
        2,
    ),
];

#[tokio::test]
async fn one_signature_answers_at_every_length_it_has_an_instance_for() {
    for AtLength { source, stages } in EVERY_LENGTH {
        assert_eq!(run_i64(source).await, stages, "{source}");
    }
}

#[tokio::test]
async fn the_signatures_call_is_sync_over_a_sync_closure_and_async_over_an_async_one() {
    let over_sync = "src() | step(|x| -> x + 1) | drain(|x| -> x + 1)";
    let over_async = "src() | step(|x| -> x + 1) | drain(|x| -> seen(x))";
    assert!(
        !prepared_entry(over_sync).may_suspend,
        "every call in the body is synchronous, so the body is"
    );
    assert!(
        prepared_entry(over_async).may_suspend,
        "the closure the call was handed suspends, and the signature's \
         effect variable is that closure's effect"
    );
    assert_eq!(run_i64(over_sync).await, 1);
    assert_eq!(run_i64(over_async).await, 1);
}

#[tokio::test]
async fn the_pure_instance_is_reached_over_an_async_pipeline() {
    let source = "src() | step(|x| -> seen(x)) | step(|x| -> x + 1) | drain(|x| -> x + 1)";
    assert_eq!(run_i64(source).await, 2);
}

#[tokio::test]
async fn a_declaration_whose_parameter_names_the_effect_is_async_over_an_async_stage() {
    let over_sync = "src() | step(|x| -> x + 1) | depth()";
    let over_async = "src() | step(|x| -> seen(x)) | depth()";
    assert!(!prepared_entry(over_sync).may_suspend);
    assert!(
        prepared_entry(over_async).may_suspend,
        "the stage's closure suspends, and the parameter type `Pipe<(A, ()), O, E, I, Rt>` \
         makes the pipeline's effect the call's own"
    );
    assert_eq!(run_i64(over_sync).await, 1);
    assert_eq!(run_i64(over_async).await, 1);
}

#[tokio::test]
async fn a_signature_that_names_the_pipeline_is_async_over_an_async_stage() {
    let over_sync = "src() | step(|x| -> x + 1) | tally()";
    let over_async = "src() | step(|x| -> seen(x)) | tally()";
    assert!(!prepared_entry(over_sync).may_suspend);
    assert!(
        prepared_entry(over_async).may_suspend,
        "the stage's closure suspends, and the signature's parameter type \
         `Pipe<Ts, O, E, I, Rt>` makes the pipeline's effect the call's own"
    );
    assert_eq!(run_i64(over_sync).await, 1);
    assert_eq!(run_i64(over_async).await, 1);
}

/// A call runs its instance, so it has the instance's effect whether or
/// not the signature's parameter names the pipeline: `bare_tally`'s is the
/// bare variable its per-length instances are matched by, and the call over
/// an async stage is `Async` as `tally`'s is.
#[tokio::test]
async fn a_signature_that_hides_the_pipeline_still_takes_its_instances_task() {
    let over_sync = "src() | step(|x| -> x + 1) | bare_tally()";
    let over_async = "src() | step(|x| -> seen(x)) | bare_tally()";
    assert!(!prepared_entry(over_sync).may_suspend);
    assert!(prepared_entry(over_async).may_suspend);
    assert_eq!(run_i64(over_sync).await, 1);
    assert_eq!(run_i64(over_async).await, 1);
}
