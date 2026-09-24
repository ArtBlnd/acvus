//! RFC-0067, spike: an iterator is a type with a `next`.
//!
//! A source, two adaptors standing at a pattern and a consumer over one
//! shared signature `nit::next` — no `Iterator` type and no `Box<dyn>`
//! chain, so that per element the machine runs one plain function pointer
//! per stage, taken out of the instance the call site resolved.
//!
//! These names sit in their own namespace and the registered `iter::`
//! surface is left alone, because the other half of this spike is
//! `acvus-interpreter-test/benches/accum.rs`, which measures the two
//! designs against each other inside one binary. Rename anything here and
//! those bench rows stop compiling.

use std::ops::Deref;
use std::sync::Arc;

use acvus_extern::{
    Closure, ClosureFn, Cross, Ctx, ExternType, Instance, Later, PassedByValue, Pure, Ref,
    Registry, Runtime, Shared, Stored, TransparentOver, Var, extern_fn, extern_registry, kind,
};
use acvus_interpreter::code::Body;
use acvus_interpreter::{AcvusRuntime, PrepareCtx, prepare_module};
use acvus_interpreter_test::*;
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

mod sig {
    use acvus_extern::extern_signature;

    extern_signature! {
        ns: "nit",
        effect = E,
        fn next<I, T, E, Rt>(it: &mut I) -> Option<T>
        where
            I: Var<kind::Type>,
            T: Var<kind::Type>,
            E: Var<kind::Effect>,
            Rt: Runtime;
    }
}

pub struct NRangeBody {
    at: i64,
    end: i64,
}

#[derive(ExternType)]
#[extern_type(name = "NRange")]
#[repr(transparent)]
pub struct NRange(NRangeBody);

#[extern_fn(effect = pure)]
fn nrange(start: i64, end: i64) -> NRange {
    NRange(NRangeBody { at: start, end })
}

#[extern_fn(instance_of = sig::next, effect = pure)]
fn next_nrange(it: &mut NRange) -> Option<i64> {
    (it.0.at < it.0.end).then(|| {
        let at = it.0.at;
        it.0.at += 1;
        at
    })
}

#[derive(acvus_extern::Payload)]
pub struct NMapBody<'a, I, T, U, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    inner: I,
    next: Instance<'a, sig::next<I, T, E, Rt>, I, Rt>,
    f: Closure<'a, (T,), U, E, Rt>,
}

#[derive(ExternType)]
#[extern_type(name = "NMap")]
#[repr(transparent)]
pub struct NMap<'a, I, T, U, E, Rt>(NMapBody<'a, I, T, U, E, Rt>)
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

#[extern_fn(effect = pure)]
fn nmap<'a, I, T, U, E, Rt>(
    it: I,
    f: Closure<'a, (T,), U, E, Rt>,
    next: Instance<'a, sig::next<I, T, E, Rt>, I, Rt>,
) -> NMap<'a, I, T, U, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    NMap(NMapBody { inner: it, next, f })
}

#[extern_fn(instance_of = sig::next, effect = E)]
fn next_nmap<I, T, U, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut NMap<'_, I, T, U, E, Rt>) -> Option<U>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    U: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let x = it.0.next.call(ctx, &mut it.0.inner, ())?;
    Some(it.0.f.call_now(ctx, (x,)))
}

#[derive(acvus_extern::Payload)]
pub struct NFilterBody<'a, I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    inner: I,
    next: Instance<'a, sig::next<I, T, E, Rt>, I, Rt>,
    f: Closure<'a, (Ref<'a, T, Shared, Rt>,), bool, E, Rt>,
}

#[derive(ExternType)]
#[extern_type(name = "NFilter")]
#[repr(transparent)]
pub struct NFilter<'a, I, T, E, Rt>(NFilterBody<'a, I, T, E, Rt>)
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime;

#[extern_fn(effect = pure)]
fn nfilter<'a, I, T, E, Rt>(
    it: I,
    f: Closure<'a, (Ref<'a, T, Shared, Rt>,), bool, E, Rt>,
    next: Instance<'a, sig::next<I, T, E, Rt>, I, Rt>,
) -> NFilter<'a, I, T, E, Rt>
where
    I: Var<kind::Type>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    NFilter(NFilterBody { inner: it, next, f })
}

#[extern_fn(instance_of = sig::next, effect = E)]
fn next_nfilter<I, T, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut NFilter<'_, I, T, E, Rt>) -> Option<T>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    T: Var<kind::Type> + Stored<Rt> + Cross<Rt> + PassedByValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    loop {
        let x = it.0.next.call(ctx, &mut it.0.inner, ())?;
        let keep = it.0.f.call_now(ctx, (&x,));
        if keep {
            return Some(x);
        }
    }
}

#[extern_fn(effect = E)]
fn nsum<I, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: I,
    next: Instance<'_, sig::next<I, i64, E, Rt>, I, Rt>,
) -> i64
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut acc = 0i64;
    while let Some(x) = next.call(ctx, &mut it, ()) {
        acc = acc.wrapping_add(x);
    }
    acc
}

#[derive(acvus_extern::Payload)]
pub struct NSlowedBody<'a, I, Rt>
where
    I: Var<kind::Type>,
    Rt: Runtime,
{
    inner: I,
    next: Instance<'a, sig::next<I, i64, Pure, Rt>, I, Rt>,
}

#[derive(ExternType)]
#[extern_type(name = "NSlowed")]
#[repr(transparent)]
pub struct NSlowed<'a, I, Rt>(NSlowedBody<'a, I, Rt>)
where
    I: Var<kind::Type>,
    Rt: Runtime;

#[extern_fn(effect = pure)]
fn nslowed<'a, I, Rt>(it: I, next: Instance<'a, sig::next<I, i64, Pure, Rt>, I, Rt>) -> NSlowed<'a, I, Rt>
where
    I: Var<kind::Type>,
    Rt: Runtime,
{
    NSlowed(NSlowedBody { inner: it, next })
}

/// The file's async stage: it suspends once per element and reaches its
/// inner stage through `into_async` — the sync instance `nslowed` was
/// handed, called at the async task (RFC-0067 rule 5).
#[extern_fn(instance_of = sig::next, effect = pure)]
async fn next_nslowed<I, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut NSlowed<'_, I, Rt>) -> Option<i64>
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    tokio::task::yield_now().await;
    it.0.next
        .into_async()
        .call_await(ctx, &mut it.0.inner, ())
        .await
}

/// `nsum` at the async task: the requirement is written `Later`, so the
/// instance the site resolves is called with `call_await`.
#[extern_fn(effect = pure)]
async fn nsum_await<I, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: I,
    next: Instance<'_, sig::next<I, i64, Pure, Rt>, I, Rt, Later>,
) -> i64
where
    I: Var<kind::Type> + Deref<Target = Rt::Value>,
    Rt: Runtime,
{
    let mut it = it;
    let mut acc = 0i64;
    while let Some(x) = next.call_await(ctx, &mut it, ()).await {
        acc = acc.wrapping_add(x);
    }
    acc
}

pub fn next_registry<Rt>() -> Registry<Rt>
where
    Rt: Runtime,
{
    extern_registry! {
        ns: "nit",
        types: [NRange, NMap<_, _, _, _, Rt>, NFilter<_, _, _, Rt>, NSlowed<_, Rt>],
        signatures: [sig::next],
        fns: [nrange, next_nrange, nmap, next_nmap, nfilter, next_nfilter, nslowed,
              next_nslowed, nsum, nsum_await],
    }
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut rs = acvus_ext::std_registries::<AcvusRuntime>();
    rs.push(next_registry());
    rs
}

async fn run_i64(source: &str) -> i64 {
    let i = Interner::new();
    run_script_mode_with_externs(&i, source, Context::default(), registries(), Ty::I64)
        .await
        .value
        .as_int()
}

fn prepared_entry(source: &str, opt: Opt) -> Body {
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
    let module = cr.modules.get(&cr.entry_qref).expect("the entry module");
    let ctx = PrepareCtx {
        interner: &i,
        externs: &cr.extern_executables,
        context_names: &cr.context_names,
        instances: &cr.instances,
        access: acvus_mir::graph::Access::Sync,
    };
    let prepared = prepare_module(module, &ctx);
    Arc::try_unwrap(prepared.main).unwrap_or_else(|_| panic!("one reference to main"))
}

/// The same program at one optimization level: the async tests assert a
/// value at both, which `run_i64`'s harness fixes at `Opt::Full`.
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
    interp.execute().await.expect("the seeds hold every context the run fetches").as_int()
}

const SOURCE_ONLY: &str = "nsum(nrange(0, 5))";
const TWO_STAGE: &str = "nsum(nmap(nrange(0, 5), |x| -> x * 2))";
const THREE_STAGE: &str = "nsum(nfilter(nmap(nrange(0, 5), |x| -> x * 2), |x| -> *x > 3))";

#[tokio::test]
async fn a_pipeline_of_instances_yields_what_the_stages_say() {
    assert_eq!(run_i64(SOURCE_ONLY).await, 0 + 1 + 2 + 3 + 4);
    assert_eq!(run_i64(TWO_STAGE).await, 0 + 2 + 4 + 6 + 8);
    assert_eq!(run_i64(THREE_STAGE).await, 4 + 6 + 8);
}

#[test]
fn a_pure_pipeline_suspends_nowhere() {
    for opt in [Opt::None, Opt::Full] {
        assert!(
            !prepared_entry(THREE_STAGE, opt).may_suspend,
            "every call in the pipeline is pure at {opt:?}"
        );
    }
}

const AWAITED_SLOWED: &str = "nsum_await(nslowed(nrange(0, 5)))";
const AWAITED_SYNC: &str = "nsum_await(nrange(0, 5))";
const AWAITED_TWO_STAGE: &str = "nsum_await(nmap(nrange(0, 5), |x| -> x * 2))";

/// `Instance<_, _, _, Later>::call_await` drives a body that suspends —
/// `next_nslowed` — and one that returns, `nrange`'s glue, which
/// `Signature::call_later` tells apart by the task on the value's own
/// word. `next_nslowed` reaches its inner stage through `into_async`.
#[tokio::test]
async fn an_async_instance_is_driven_through_call_await() {
    for opt in [Opt::None, Opt::Full] {
        assert_eq!(run_i64_at(AWAITED_SLOWED, opt).await, 0 + 1 + 2 + 3 + 4);
        assert_eq!(run_i64_at(AWAITED_SYNC, opt).await, 0 + 1 + 2 + 3 + 4);
        assert_eq!(run_i64_at(AWAITED_TWO_STAGE, opt).await, 0 + 2 + 4 + 6 + 8);
    }
}

/// A pipeline holding the async stage suspends; the same consumer over
/// only sync stages does not, because `call_later`'s `Task::Sync` arm
/// returns a ready future rather than an awaiting glue.
#[test]
fn the_task_of_a_pipeline_is_its_stages() {
    for opt in [Opt::None, Opt::Full] {
        assert!(
            prepared_entry(AWAITED_SLOWED, opt).may_suspend,
            "`nslowed`'s instance is an `async fn` at {opt:?}"
        );
    }
}

/// The refusal is at check: a sync body's requirement is met with the
/// instances that run at `Task::Sync`, and `NSlowed`'s is not one.
#[test]
fn a_sync_consumer_given_an_async_instance_is_refused() {
    let i = Interner::new();
    let source = "nsum(nslowed(nrange(0, 5)))";
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse error"));
    let refused = check_source(
        &i,
        ast,
        &FxHashMap::default(),
        registries(),
        Ty::I64,
        Opt::Full,
        |_| {},
    )
    .err()
    .expect("a sync consumer has no instance of nit::next at NSlowed");
    let messages = refused.messages.join("; ");
    assert!(
        messages.contains("NSlowed") && messages.contains("NRange"),
        "the refusal names the type given and the instances a sync body may reach: {messages}"
    );
}

/// `nmap` is generic in its element types: what `x` is inside the closure
/// comes from the instance of `nit::next` at the source's type, decided
/// where `nmap`'s requirement is (RFC-0068 rule 1).
#[tokio::test]
async fn a_requirement_binds_the_signatures_other_variables() {
    assert_eq!(
        run_i64("nsum(nmap(nrange(0, 5), |x| -> x + 1))").await,
        1 + 2 + 3 + 4 + 5
    );
    assert_eq!(
        run_i64("nsum(nfilter(nrange(0, 6), |x| -> *x % 2 == 0))").await,
        0 + 2 + 4
    );
}

/// A consumer of `i64` over a stage yielding `f64`: the requirement's
/// instance is `next` at `NMap<.., f64, ..>`, whose result does not join
/// `Option<i64>`.
#[test]
fn a_consumer_whose_element_type_the_stage_does_not_yield_is_refused() {
    let i = Interner::new();
    let source = "nsum(nmap(nrange(0, 5), |x| -> 1.5))";
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse error"));
    let refused = check_source(
        &i,
        ast,
        &FxHashMap::default(),
        registries(),
        Ty::I64,
        Opt::Full,
        |_| {},
    )
    .err()
    .expect("nsum reads i64 and the map yields f64");
    let messages = refused.messages.join("; ");
    assert!(
        messages.contains("Float") && messages.contains("i64"),
        "the refusal names the two element types: {messages}"
    );
}

#[test]
fn a_consumer_given_a_non_iterator_is_refused() {
    let i = Interner::new();
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, "nsum(7)").expect("parse error"));
    let refused = check_source(
        &i,
        ast,
        &FxHashMap::default(),
        registries(),
        Ty::I64,
        Opt::Full,
        |_| {},
    )
    .err()
    .expect("an i64 has no instance of nit::next");
    let messages = refused.messages.join("; ");
    assert!(
        messages.contains("i64") && messages.contains("NRange") && messages.contains("NMap"),
        "the refusal names the ground type and every pattern an instance stands at: {messages}"
    );
}
