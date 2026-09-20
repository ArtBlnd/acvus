//! RFC-0067 step 3, second half: an entry is a tree.
//!
//! A source (`Counter`), an adaptor standing at a pattern (`Doubled<I>`)
//! and a consumer (`total`) requiring an instance of `probe::advance`. The
//! adaptor's value holds the inner iterator's value and no pointer; what
//! the adaptor's instance calls its inner `advance` through is the entry
//! the call site resolved, whose children are the instances its own bounds
//! required.
//!
//! The source of those children is the point: a pattern instance reached
//! from a script site reads them from its site table, and one reached
//! through an entry reads them from the run's trailing word. Both are the
//! same node, so the macro writes one body.

use std::marker::PhantomData;
use std::sync::Arc;

use acvus_extern::{
    Carrier, Closure, ClosureFn, ExternType, Held, InstanceOf, InstanceOfAsync, OneValue, Pure,
    Registry, Runtime, Var, extern_fn, extern_registry, kind,
};
use acvus_interpreter::code::Code;
use acvus_interpreter::{AcvusRuntime, PrepareCtx, prepare_module};
use acvus_interpreter_test::*;
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

mod sig {
    use acvus_extern::{Runtime, Var, extern_signature, kind};

    extern_signature! {
        ns: "probe",
        effect = E,
        fn advance<I, T, E, Rt>(it: &mut I) -> Option<T>
        where
            I: Var<kind::Type>,
            T: Var<kind::Type>,
            E: Var<kind::Effect>,
            Rt: Runtime;
    }
}

/// The source: it yields `n, n - 1, …, 1` and then nothing.
#[derive(ExternType)]
#[extern_type(name = "Counter")]
#[repr(transparent)]
pub struct Counter(i64);

#[extern_fn(effect = pure)]
fn counter(n: i64) -> Counter {
    Counter(n)
}

#[extern_fn(instance_of = sig::advance, effect = pure)]
fn advance_counter(it: &mut Counter) -> Option<i64> {
    (it.0 > 0).then(|| {
        let at = it.0;
        it.0 -= 1;
        at
    })
}

/// The adaptor's payload: the inner iterator's value with no pointer
/// beside it, and the closure each element passes through. It names no
/// type parameter, so the handler that wrote it and the instance that
/// reads it back name one Rust type.
pub struct DoubledBody<Rt>
where
    Rt: Runtime,
{
    inner: Held<Rt>,
    f: Closure<(i64,), i64, Pure, Rt>,
}

#[derive(ExternType)]
#[extern_type(name = "Doubled")]
#[repr(transparent)]
pub struct Doubled<I, Rt>(DoubledBody<Rt>, PhantomData<I>)
where
    I: Var<kind::Type>,
    Rt: Runtime;

/// The adaptor: the carrier it was handed becomes the value alone.
#[extern_fn(effect = pure)]
fn doubled<I, Rt>(it: I, f: Closure<(i64,), i64, Pure, Rt>) -> Doubled<I, Rt>
where
    I: Var<kind::Type> + Carrier<Rt> + InstanceOf<sig::advance<I, i64, Pure, Rt>, Rt>,
    Rt: Runtime,
{
    Doubled(
        DoubledBody {
            inner: Held::of(it),
            f,
        },
        PhantomData,
    )
}

/// The constructor that builds the adaptor's pattern over an inner type
/// that has no instance.
#[extern_fn(effect = pure)]
fn doubled_unchecked<I, Rt>(rt: &Rt, it: I, f: Closure<(i64,), i64, Pure, Rt>) -> Doubled<I, Rt>
where
    I: Var<kind::Type> + OneValue<Rt>,
    Rt: Runtime,
{
    Doubled(
        DoubledBody {
            inner: Held::of_value(it.erase(rt)),
            f,
        },
        PhantomData,
    )
}

/// The instance at the pattern `Doubled<I>`: its own bound is reached
/// through the entries the site resolved for it, which `#[extern_fn]`
/// appended as `i_bound` off the `where` clause below.
#[extern_fn(instance_of = sig::advance, effect = pure)]
fn advance_doubled<I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    it: &mut Doubled<I, Rt>,
) -> Option<i64>
where
    I: Var<kind::Type> + Carrier<Rt> + InstanceOf<sig::advance<I, i64, Pure, Rt>, Rt>,
    Rt: Runtime,
{
    let x = {
        let mut held = it.0.inner.at(&i_bound);
        I::call(&mut held, rt, frame, ())
    }?;
    Some(it.0.f.call_now(rt, frame, (x,)))
}

pub struct SlowedBody<Rt>
where
    Rt: Runtime,
{
    inner: Held<Rt>,
}

#[derive(ExternType)]
#[extern_type(name = "Slowed")]
#[repr(transparent)]
pub struct Slowed<I, Rt>(SlowedBody<Rt>, PhantomData<I>)
where
    I: Var<kind::Type>,
    Rt: Runtime;

#[extern_fn(effect = pure)]
fn slowed<I, Rt>(it: I) -> Slowed<I, Rt>
where
    I: Var<kind::Type> + Carrier<Rt> + InstanceOf<sig::advance<I, i64, Pure, Rt>, Rt>,
    Rt: Runtime,
{
    Slowed(
        SlowedBody {
            inner: Held::of(it),
        },
        PhantomData,
    )
}

/// The file's async stage.
#[extern_fn(instance_of = sig::advance, effect = pure)]
async fn advance_slowed<I, Rt>(
    rt: &Rt,
    frame: &mut Rt::Frame<'_>,
    it: &mut Slowed<I, Rt>,
) -> Option<i64>
where
    I: Var<kind::Type> + Carrier<Rt> + InstanceOfAsync<sig::advance<I, i64, Pure, Rt>, Rt>,
    Rt: Runtime,
{
    tokio::task::yield_now().await;

    let mut held = it.0.inner.at(&i_bound);
    <I as InstanceOfAsync<sig::advance<I, i64, Pure, Rt>, Rt>>::call(&mut held, rt, frame, ()).await
}

/// The consumer: it requires the signature and calls it until it ends.
#[extern_fn(effect = pure)]
fn total<I, Rt>(rt: &Rt, frame: &mut Rt::Frame<'_>, it: I) -> i64
where
    I: Var<kind::Type> + Carrier<Rt> + InstanceOf<sig::advance<I, i64, Pure, Rt>, Rt>,
    Rt: Runtime,
{
    let mut it = it;
    let mut acc = 0;
    while let Some(x) = I::call(&mut it, rt, frame, ()) {
        acc += x;
    }
    acc
}

/// The same consumer at the async task.
#[extern_fn(effect = pure)]
async fn total_await<I, Rt>(rt: &Rt, frame: &mut Rt::Frame<'_>, it: I) -> i64
where
    I: Var<kind::Type> + Carrier<Rt> + InstanceOfAsync<sig::advance<I, i64, Pure, Rt>, Rt>,
    Rt: Runtime,
{
    let mut it = it;
    let mut acc = 0;
    while let Some(x) =
        <I as InstanceOfAsync<sig::advance<I, i64, Pure, Rt>, Rt>>::call(&mut it, rt, frame, ())
            .await
    {
        acc += x;
    }
    acc
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "probe",
        types: [Counter, Doubled<_, AcvusRuntime>, Slowed<_, AcvusRuntime>],
        signatures: [sig::advance],
        fns: [counter, advance_counter, doubled, doubled_unchecked, advance_doubled, slowed,
              advance_slowed, total, total_await],
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

fn prepared_entry(source: &str, opt: Opt) -> Code {
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
    };
    let prepared = prepare_module(module, &ctx);
    Arc::try_unwrap(prepared.main).unwrap_or_else(|_| panic!("one reference to main"))
}

const ONE_DEEP: &str = "total(doubled(counter(5), |x| -> x * 2))";
const TWO_DEEP: &str = "total(doubled(doubled(counter(5), |x| -> x * 2), |x| -> x * 2))";

/// `counter(5)` yields 5 + 4 + 3 + 2 + 1 = 15; one `doubled` over it is
/// 30, and a second is 60. The inner instance is reached through the
/// entry's child at every level.
#[tokio::test]
async fn a_pattern_instance_reaches_its_own_bound_through_the_entrys_children() {
    assert_eq!(run_i64(ONE_DEEP).await, 30);
    assert_eq!(run_i64(TWO_DEEP).await, 60);
}

/// The second source of entries: the site of a script's own call to the
/// signature, where no entry stands above the instance and the site table
/// carries the same node.
#[tokio::test]
async fn a_script_site_calling_the_instance_supplies_the_same_entries() {
    let source = "let d = doubled(counter(3), |x| -> x * 2); \
                  let acc = 0; \
                  while let Some(x) = advance(&mut d) { acc = acc + x; } \
                  acc";
    assert_eq!(run_i64(source).await, 12, "(3 + 2 + 1) doubled");
}

/// The requirement is the declaration's bound, and a type with no instance
/// of the signature is refused before anything runs.
#[test]
fn a_consumer_given_a_type_with_no_instance_is_refused() {
    let i = Interner::new();
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, "total(7)").expect("parse error"));
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
    .expect("an i64 has no instance of probe::advance");
    let messages = refused.messages.join("; ");
    assert!(
        messages.contains("i64") && messages.contains("Counter") && messages.contains("Doubled"),
        "the refusal names the ground type and every pattern an instance stands at: {messages}"
    );
}

/// The effect of a pattern instance is its declaration's: every stage here
/// is pure, so the body that runs them suspends nowhere.
#[test]
fn a_pure_pipeline_suspends_nowhere() {
    for opt in [Opt::None, Opt::Full] {
        assert!(
            !prepared_entry(ONE_DEEP, opt).may_suspend(),
            "every call in the pipeline is pure at {opt:?}"
        );
    }
}

const SLOWED: &str = "total_await(slowed(doubled(counter(5), |x| -> x * 2)))";
const AWAITED_SYNC: &str = "total_await(doubled(counter(5), |x| -> x * 2))";

/// One consumer body reaching `Counter`'s and `Doubled`'s `Sync` nodes and
/// `Slowed`'s `Await` node. `slowed` passes each element through.
#[tokio::test]
async fn an_async_instance_is_reached_through_its_entry() {
    assert_eq!(run_i64(SLOWED).await, 30);
    assert_eq!(run_i64(AWAITED_SYNC).await, 30);
    assert_eq!(run_i64(ONE_DEEP).await, 30);
}

#[test]
fn the_task_of_a_pipeline_is_its_stages() {
    for opt in [Opt::None, Opt::Full] {
        assert!(
            prepared_entry(SLOWED, opt).may_suspend(),
            "`slowed`'s instance is an `async fn` at {opt:?}"
        );
    }
}

/// The refusal is at check: the program never reaches `prepare`, where the
/// entry it cannot call would be asked for.
#[test]
fn a_sync_consumer_given_an_async_instance_is_refused() {
    let i = Interner::new();
    let source = "total(slowed(counter(5)))";
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
    .expect("a sync consumer has no instance of probe::advance at Slowed");
    let messages = refused.messages.join("; ");
    assert!(
        messages.contains("Slowed") && messages.contains("Counter"),
        "the refusal names the type given and the instances a sync body may reach: {messages}"
    );
}

/// The instance at a pattern carries its own bound, so the ill-formed value
/// `doubled_unchecked` builds is refused at check.
#[test]
fn a_pattern_instance_whose_inner_has_no_instance_is_refused() {
    let i = Interner::new();
    let source = "total(doubled_unchecked(7, |x| -> x * 2))";
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
    .expect("an i64 inside Doubled has no instance of probe::advance");
    let messages = refused.messages.join("; ");
    assert!(
        messages.contains("no instance of probe::advance at Doubled<i64>"),
        "the refusal names the requirement the inner type fails: {messages}"
    );
}
