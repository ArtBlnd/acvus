//! RFC-0067 step 4, spike: an iterator is a type with a `next`.
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

use std::marker::PhantomData;
use std::ops::DerefMut;
use std::sync::Arc;

use acvus_extern::{
    Closure, ClosureFn, Ctx, ExternType, Instance, OneValue, Opaque, Owned, Ref, Registry, Runtime,
    Shared, Var, extern_fn, extern_registry, kind,
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

/// A payload may name no type or effect parameter, so it holds this at the
/// erased ones — as it already holds its `Closure` at `Opaque`.
type InnerNext<Rt> = Instance<sig::next<Owned<Rt>, i64, Opaque, Rt>, Owned<Rt>, Rt>;

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

pub struct NMapBody<Rt>
where
    Rt: Runtime,
{
    inner: Owned<Rt>,
    next: InnerNext<Rt>,
    f: Closure<(i64,), i64, Opaque, Rt>,
}

#[derive(ExternType)]
#[extern_type(name = "NMap")]
#[repr(transparent)]
pub struct NMap<I, E, Rt>(NMapBody<Rt>, PhantomData<(I, E)>)
where
    I: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime;

#[extern_fn(effect = pure)]
fn nmap<I, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: I,
    f: Closure<(i64,), i64, E, Rt>,
    next: Instance<sig::next<I, i64, E, Rt>, I, Rt>,
) -> NMap<I, E, Rt>
where
    I: Var<kind::Type> + Into<Owned<Rt>>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    NMap(
        NMapBody {
            inner: it.into(),
            // SAFETY: the same word, at the erased parameters the payload
            // names; `I` is the runtime's value and `E` is erased at run
            // time, so the glue this names is the one that was resolved.
            next: unsafe { Instance::at(next.into_value()) },
            f: Closure::new(ctx.rt, f.into_value()),
        },
        PhantomData,
    )
}

#[extern_fn(instance_of = sig::next, effect = E)]
fn next_nmap<I, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut NMap<I, E, Rt>) -> Option<i64>
where
    I: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    // SAFETY: `inner` and `next` were laid side by side by `nmap`, so the
    // instance stands at the type the value beside it holds.
    let x = unsafe { it.0.next.call(ctx, &mut it.0.inner, ()) }?;
    Some(it.0.f.call_now(ctx, (x,)))
}

pub struct NFilterBody<Rt>
where
    Rt: Runtime,
{
    inner: Owned<Rt>,
    next: InnerNext<Rt>,
    f: Closure<(Ref<i64, Shared, Rt>,), bool, Opaque, Rt>,
}

#[derive(ExternType)]
#[extern_type(name = "NFilter")]
#[repr(transparent)]
pub struct NFilter<I, E, Rt>(NFilterBody<Rt>, PhantomData<(I, E)>)
where
    I: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime;

#[extern_fn(effect = pure)]
fn nfilter<I, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: I,
    f: Closure<(Ref<i64, Shared, Rt>,), bool, E, Rt>,
    next: Instance<sig::next<I, i64, E, Rt>, I, Rt>,
) -> NFilter<I, E, Rt>
where
    I: Var<kind::Type> + Into<Owned<Rt>>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    NFilter(
        NFilterBody {
            inner: it.into(),
            // SAFETY: as `nmap`'s.
            next: unsafe { Instance::at(next.into_value()) },
            f: Closure::new(ctx.rt, f.into_value()),
        },
        PhantomData,
    )
}

#[extern_fn(instance_of = sig::next, effect = E)]
fn next_nfilter<I, E, Rt>(ctx: &mut Ctx<'_, Rt>, it: &mut NFilter<I, E, Rt>) -> Option<i64>
where
    I: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    loop {
        // SAFETY: as `next_nmap`'s.
        let x = unsafe { it.0.next.call(ctx, &mut it.0.inner, ()) }?;
        let lent = <i64 as OneValue<Rt>>::erase(x, ctx.rt);
        let keep = it.0.f.call_now(ctx, (Ref::lend(ctx.rt, &lent),));
        Owned::<Rt>::from_value(lent).release();
        if keep {
            return Some(x);
        }
    }
}

#[extern_fn(effect = E)]
fn nsum<I, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: I,
    next: Instance<sig::next<I, i64, E, Rt>, I, Rt>,
) -> i64
where
    I: Var<kind::Type> + DerefMut<Target = Rt::Value>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut acc = 0i64;
    // SAFETY: `Externs::combine` met this parameter's requirement with the
    // instance of `nit::next` at the ground type `I` was filled with, and
    // `it` is a value of that type.
    while let Some(x) = unsafe { next.call(ctx, &mut it, ()) } {
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
        types: [NRange, NMap<_, _, Rt>, NFilter<_, _, Rt>],
        signatures: [sig::next],
        fns: [nrange, next_nrange, nmap, next_nmap, nfilter, next_nfilter, nsum],
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
            !prepared_entry(THREE_STAGE, opt).may_suspend(),
            "every call in the pipeline is pure at {opt:?}"
        );
    }
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
