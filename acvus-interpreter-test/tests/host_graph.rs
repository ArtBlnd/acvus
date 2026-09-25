//! RFC-0095 at the host's contract: several hosts compiled as one graph,
//! a host calling another's entry as a function along a DAG.

use acvus_extern::{
    Closure, ClosureFn, Ctx, Declared, OneValue, Registry, Runtime, TyArg, Var, extern_fn,
    extern_registry, kind,
};
use acvus_interpreter::{
    AcvusRuntime, Cause, Host, HostError, HostGraph, MemoryStorage, Origin, Program, Refusal,
    SequentialExecutor, Source,
};
use acvus_mir::ty::{Effect, EffectArg, Flows, IntTy, ObjectTy, PolyTy, TyTerm, TypeArg};
use acvus_utils::{Interner, QualifiedRef};

#[derive(TyArg)]
pub struct N {
    n: i64,
}

fn graph() -> HostGraph {
    HostGraph::new(acvus_ext::std_registries::<AcvusRuntime>())
}

fn with_entry(name: &'static str, source: &'static str) -> impl FnOnce(Host) -> Result<Host, HostError> {
    move |host| Ok(host.entry::<(), i64>(name, Source::Script(source)))
}

fn with_n_entry(name: &'static str, source: &'static str) -> impl FnOnce(Host) -> Result<Host, HostError> {
    move |host| Ok(host.entry::<N, i64>(name, Source::Script(source)))
}

fn compiled(graph: Result<HostGraph, HostError>) -> Program {
    match graph.and_then(|graph| graph.compile(SequentialExecutor)) {
        Ok(program) => program,
        Err(error) => panic!("the graph is refused: {error}"),
    }
}

fn refusals(graph: Result<HostGraph, HostError>) -> Vec<Refusal> {
    match graph.and_then(|graph| graph.compile(SequentialExecutor)) {
        Ok(_) => panic!("the graph compiled"),
        Err(HostError::Refused(refusals)) => refusals,
        Err(other) => panic!("a graph is refused with its refusals, not {other:?}"),
    }
}

fn at_the_source<'s>(source: &'s str, refused: &[Refusal]) -> Vec<(String, &'s str)> {
    refused
        .iter()
        .map(|refusal| {
            assert_eq!(refusal.origin, Some(Origin::Entry("a/main".to_owned())), "{refusal:?}");
            let span = refusal.span.expect("an argument's refusal marks the call");
            (refusal.message.clone(), &source[span.start..span.end])
        })
        .collect()
}

fn messages(graph: Result<HostGraph, HostError>) -> Vec<String> {
    refusals(graph).into_iter().map(|refusal| refusal.message).collect()
}

async fn run_unit(program: &Program, storage: &mut MemoryStorage, entry: &str) -> i64 {
    program
        .scope(async |s| {
            let mut page = s.open(storage);
            let entry = s.entry::<(), i64>(entry).expect("the entry takes `()` and returns `i64`");
            let output = entry.run(&mut page, ()).await.expect("the run ends");
            output.with(|n: &i64| *n).expect("an `i64`")
        })
        .await
}

async fn context_i64(program: &Program, storage: &mut MemoryStorage, key: &str) -> i64 {
    program
        .scope(async |s| {
            let mut page = s.open(storage);
            page.with(key, |n: &i64| *n).await.expect("the context is an `i64`")
        })
        .await
}

#[tokio::test]
async fn two_hosts_keep_their_contexts_of_one_name_apart() {
    let program = compiled(
        graph()
            .host("a", with_entry("main", "@x = 1; 10"))
            .and_then(|g| g.host("b", with_entry("main", "@x = 2; 20")))
            .map(|g| g.entry("a", "main").entry("b", "main")),
    );
    let mut contexts: Vec<&str> = program.contexts().collect();
    contexts.sort_unstable();
    assert_eq!(contexts, ["a/x", "b/x"]);
    let mut storage = MemoryStorage::new();
    assert_eq!(run_unit(&program, &mut storage, "a/main").await, 10);
    assert_eq!(run_unit(&program, &mut storage, "b/main").await, 20);
    assert_eq!(context_i64(&program, &mut storage, "a/x").await, 1);
    assert_eq!(context_i64(&program, &mut storage, "b/x").await, 2);
}

fn caller(source: &'static str) -> Result<HostGraph, HostError> {
    graph()
        .host("a", with_entry("main", source))
        .and_then(|g| g.host("b", with_n_entry("twice", "$n * 2")))
        .map(|g| g.expose("a", "twice", "b", "twice").entry("a", "main"))
}

#[tokio::test]
async fn an_exposed_entry_is_called_with_one_object_of_its_inputs() {
    let program = compiled(caller("twice({ n: 21, })"));
    let mut storage = MemoryStorage::new();
    assert_eq!(run_unit(&program, &mut storage, "a/main").await, 42);
}

#[derive(TyArg)]
pub struct Sum {
    n: i64,
    k: i64,
}

#[test]
fn a_missing_field_is_refused_at_the_call() {
    let source = "plus({ n: 21, })";
    let refused = refusals(
        graph()
            .host("a", with_entry("main", source))
            .and_then(|g| g.host("b", |host| Ok(host.entry::<Sum, i64>("plus", Source::Script("$n + $k")))))
            .map(|g| g.expose("a", "plus", "b", "plus").entry("a", "main")),
    );
    assert_eq!(
        at_the_source(source, &refused),
        [(
            "this value has no `k` stored on every path that reaches here".to_owned(),
            "plus({ n: 21, })"
        )]
    );
}

#[test]
fn an_extra_field_is_refused_at_the_call() {
    let source = "twice({ n: 21, m: 1, })";
    let refused = refusals(caller(source));
    assert_eq!(
        at_the_source(source, &refused),
        [(
            "`m` cannot be added to a type laid out outside this body".to_owned(),
            "{ n: 21, m: 1, }"
        )]
    );
}

#[test]
fn a_mistyped_field_is_refused_at_the_call() {
    let source = "twice({ n: true, })";
    let refused = refusals(caller(source));
    assert_eq!(
        at_the_source(source, &refused),
        [("type mismatch: expected {n: i64}, got {n: Bool}".to_owned(), "{ n: true, }")]
    );
}

#[tokio::test]
async fn the_callee_writes_its_context_under_its_host() {
    let program = compiled(
        graph()
            .host("a", with_entry("main", "@count = 1; bump({ n: 5, })"))
            .and_then(|g| g.host("b", with_n_entry("bump", "@count = $n; $n + 1")))
            .map(|g| g.expose("a", "bump", "b", "bump").entry("a", "main")),
    );
    let mut storage = MemoryStorage::new();
    assert_eq!(run_unit(&program, &mut storage, "a/main").await, 6);
    assert_eq!(context_i64(&program, &mut storage, "b/count").await, 5);
    assert_eq!(context_i64(&program, &mut storage, "a/count").await, 1);
}

fn cycle_of(refusals: &[Refusal]) -> Vec<Vec<String>> {
    refusals
        .iter()
        .filter_map(|refusal| match &refusal.cause {
            Some(Cause::Cycle { hosts }) => Some(hosts.clone()),
            _ => None,
        })
        .collect()
}

fn hosts(names: &[&'static str]) -> Result<HostGraph, HostError> {
    names.iter().try_fold(graph(), |g, name| {
        g.host(name, with_n_entry("e", "$n"))
    })
}

#[test]
fn a_host_exposed_to_itself_is_a_cycle() {
    let refused = refusals(hosts(&["a"]).map(|g| g.expose("a", "f", "a", "e")));
    assert_eq!(cycle_of(&refused), [["a"]]);
}

#[test]
fn two_hosts_exposed_to_each_other_are_a_cycle() {
    let refused = refusals(hosts(&["a", "b"]).map(|g| g.expose("a", "f", "b", "e").expose("b", "f", "a", "e")));
    assert_eq!(cycle_of(&refused), [["a", "b"]]);
}

#[test]
fn three_hosts_in_a_ring_are_a_cycle() {
    let refused = refusals(
        hosts(&["a", "b", "c"])
            .map(|g| g.expose("a", "f", "b", "e").expose("b", "f", "c", "e").expose("c", "f", "a", "e")),
    );
    assert_eq!(cycle_of(&refused), [["a", "b", "c"]]);
    let message = &refused
        .iter()
        .find(|refusal| refusal.cause.is_some())
        .expect("the cycle is refused")
        .message;
    assert!(message.contains("a → b → c → a"), "{message}");
}

#[tokio::test]
async fn a_diamond_is_admitted() {
    let program = compiled(
        graph()
            .host("a", with_entry("main", "left({ n: 1, }) + right({ n: 2, })"))
            .and_then(|g| g.host("b", with_n_entry("e", "base({ n: $n * 10, })")))
            .and_then(|g| g.host("c", with_n_entry("e", "base({ n: $n * 100, })")))
            .and_then(|g| g.host("d", with_n_entry("e", "$n + 1")))
            .map(|g| {
                g.expose("a", "left", "b", "e")
                    .expose("a", "right", "c", "e")
                    .expose("b", "base", "d", "e")
                    .expose("c", "base", "d", "e")
                    .entry("a", "main")
            }),
    );
    let mut storage = MemoryStorage::new();
    assert_eq!(run_unit(&program, &mut storage, "a/main").await, 11 + 201);
}

#[test]
fn an_exposure_named_like_an_entry_of_its_host_is_refused() {
    let messages = messages(
        graph()
            .host("a", with_entry("main", "0"))
            .and_then(|g| g.host("b", with_n_entry("e", "$n")))
            .map(|g| g.expose("a", "main", "b", "e")),
    );
    assert!(
        messages.iter().any(|m| m.contains("`b/e` is exposed to `a` as `main`, which is already the name of an entry of `a`")),
        "{messages:#?}"
    );
}

// -- Nothing callable crosses ------------------------------------------------

fn int() -> PolyTy {
    TyTerm::Int(IntTy::I64)
}

fn thunk() -> PolyTy {
    TyTerm::Fn {
        params: vec![],
        ret: Box::new(int()),
        captures: vec![],
        effect: Effect::PURE.into(),
        flows: Flows::none().into(),
    }
}

fn object(interner: &Interner, name: &str, ty: PolyTy) -> PolyTy {
    TyTerm::Object(ObjectTy::written([(interner.intern(name), ty)].into_iter().collect()))
}

pub struct TakesAThunk;

impl Declared for TakesAThunk {
    fn declared(interner: &Interner) -> PolyTy {
        object(interner, "make", thunk())
    }
}

pub struct AThunk;

impl Declared for AThunk {
    fn declared(_: &Interner) -> PolyTy {
        thunk()
    }
}

/// `acvus_ext::Map`, as its registry declares it: three type parameters, an
/// effect and a region.
pub struct TakesAMap;

impl Declared for TakesAMap {
    fn declared(interner: &Interner) -> PolyTy {
        let map = TyTerm::UserDefined {
            id: QualifiedRef::root(interner.intern("Map")),
            type_args: vec![TypeArg::uniform(int()), TypeArg::uniform(int()), TypeArg::uniform(int())],
            effect_args: vec![EffectArg::uniform(Effect::PURE.into())],
            identity_args: vec![],
            region_params: 1,
        };
        object(interner, "it", map)
    }
}

#[derive(TyArg)]
pub struct Record {
    name: String,
    scores: Vec<i64>,
}

fn exposing<I, R>(source: &'static str) -> Result<HostGraph, HostError>
where
    I: Declared,
    R: Declared,
{
    graph()
        .host("a", with_entry("main", "0"))
        .and_then(|g| g.host("b", move |host| Ok(host.entry::<I, R>("e", Source::Script(source)))))
        .map(|g| g.expose("a", "f", "b", "e").entry("a", "main"))
}

fn crossing_messages(graph: Result<HostGraph, HostError>) -> Vec<String> {
    messages(graph)
        .into_iter()
        .filter(|message| message.contains("nothing callable crosses"))
        .collect()
}

#[test]
fn a_function_input_is_refused() {
    let messages = crossing_messages(exposing::<TakesAThunk, i64>("0"));
    assert_eq!(messages.len(), 1, "{messages:#?}");
    assert!(messages[0].contains("at `$make` of its input the function type"), "{messages:#?}");
}

#[test]
fn a_function_result_is_refused() {
    let messages = crossing_messages(exposing::<(), AThunk>("| | -> 1"));
    assert_eq!(messages.len(), 1, "{messages:#?}");
    assert!(messages[0].contains("at `result` of its result the function type"), "{messages:#?}");
}

#[test]
fn a_map_stage_is_refused() {
    let messages = crossing_messages(exposing::<TakesAMap, i64>("0"));
    assert_eq!(messages.len(), 1, "{messages:#?}");
    assert!(
        messages[0].contains("at `$it` of its input the extension type `Map"),
        "{messages:#?}"
    );
}

#[test]
fn a_record_and_a_vec_are_admitted() {
    compiled(exposing::<Record, Vec<i64>>("$scores"));
}

// -- An extern frames a call through a closure (RFC-0095 rule 6) ------------

fn frame_now<T, E, Rt>(ctx: &mut Ctx<'_, Rt>, make: Closure<'_, (), T, E, Rt>) -> T
where
    T: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    drop(make.call_now(ctx, ()));
    make.call_now(ctx, ())
}

#[extern_fn(effect = E, sync = frame_now)]
async fn frame<T, E, Rt>(ctx: &mut Ctx<'_, Rt>, make: Closure<'_, (), T, E, Rt>) -> T
where
    T: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    drop(make.call(ctx, ()).await);
    make.call(ctx, ()).await
}

fn frame_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "test",
        fns: [frame],
    }
}

fn framing(source: &'static str) -> Result<HostGraph, HostError> {
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(frame_registry());
    HostGraph::new(registries)
        .host("a", with_entry("main", source))
        .and_then(|g| {
            g.host("b", |host| {
                Ok(host
                    .init("count", Source::Expr("0"))
                    .entry::<N, i64>("tick", Source::Script("@count = @count + $n; @count")))
            })
        })
        .map(|g| g.expose("a", "tick", "b", "tick").entry("a", "main"))
}

#[tokio::test]
async fn an_extern_calls_an_exposed_entry_twice_through_a_closure() {
    let program = compiled(framing("let t = 5; frame(| | -> tick({ n: t, }))"));
    let mut storage = MemoryStorage::new();
    assert_eq!(run_unit(&program, &mut storage, "a/main").await, 10);
    assert_eq!(context_i64(&program, &mut storage, "b/count").await, 10);
}

#[test]
fn a_mistyped_field_inside_the_framing_closure_is_refused() {
    let source = "let t = true; frame(| | -> tick({ n: t, }))";
    let refused = refusals(framing(source));
    assert_eq!(
        at_the_source(source, &refused),
        [("type mismatch: expected {n: i64}, got {n: Bool}".to_owned(), "{ n: t, }")]
    );
}

// -- A host is a scope of the graph's names (RFC-0095 rule 1) ---------------

#[tokio::test]
async fn a_method_named_like_an_entry_of_its_host_is_still_the_method() {
    let program = compiled(
        graph()
            .host("a", |host| {
                Ok(host
                    .entry::<(), i64>("as_slice", Source::Script("0"))
                    .entry::<(), i64>("main", Source::Script("let v = [1, 2, 3]; v.as_slice().len() as i64")))
            })
            .map(|g| g.entry("a", "main")),
    );
    let mut storage = MemoryStorage::new();
    assert_eq!(run_unit(&program, &mut storage, "a/main").await, 3);
}

#[test]
fn an_entry_named_like_a_bare_callable_extern_is_refused_in_a_graph_too() {
    let refused = refusals(
        graph()
            .host("a", with_entry("len", "0"))
            .map(|g| g.entry("a", "len")),
    );
    let shadowing: Vec<&Refusal> = refused
        .iter()
        .filter(|refusal| matches!(refusal.cause, Some(Cause::Shadows { .. })))
        .collect();
    assert_eq!(shadowing.len(), 1, "{refused:#?}");
    assert_eq!(shadowing[0].origin, Some(Origin::Entry("a/len".to_owned())));
    assert!(
        shadowing[0].message.contains("which a script calls as `len`"),
        "{}",
        shadowing[0].message
    );
}

#[tokio::test]
async fn a_local_and_a_parameter_named_like_a_function_of_their_host_are_untouched() {
    let program = compiled(
        graph()
            .host("a", |host| {
                Ok(host
                    .entry::<(), i64>("f", Source::Script("100"))
                    .entry::<(), i64>("main", Source::Script("let f = 5; let g = |f| -> f + 1; g(f) + f() ")))
            })
            .map(|g| g.entry("a", "main")),
    );
    let mut storage = MemoryStorage::new();
    assert_eq!(run_unit(&program, &mut storage, "a/main").await, 106);
}

#[test]
fn a_diagnostic_shows_a_name_as_its_script_wrote_it() {
    let source = "f = 1; 0";
    let refused = refusals(
        graph()
            .host("a", |host| {
                Ok(host
                    .entry::<(), i64>("f", Source::Script("0"))
                    .entry::<(), i64>("main", Source::Script(source)))
            })
            .map(|g| g.entry("a", "main")),
    );
    assert_eq!(refused.len(), 1, "{refused:#?}");
    assert_eq!(refused[0].origin, Some(Origin::Entry("a/main".to_owned())));
    assert_eq!(
        refused[0].message,
        "cannot assign to `f`: no binding named `f` is in scope; `let f = ...;` binds it"
    );
}

#[test]
fn an_init_refusal_shows_its_context_as_written_and_names_the_host_in_its_origin() {
    let refused = refusals(
        graph()
            .host("a", |host| {
                Ok(host
                    .init("x", Source::Expr("@y"))
                    .entry::<(), i64>("main", Source::Script("@x")))
            })
            .map(|g| g.entry("a", "main")),
    );
    assert_eq!(refused.len(), 1, "{refused:#?}");
    assert_eq!(refused[0].origin, Some(Origin::Init("a/x".to_owned())));
    assert_eq!(refused[0].message, "the init of `@x` names `@y`, and an init names no context");
}

#[tokio::test]
async fn two_hosts_keep_their_bindings_of_one_name_apart() {
    let program = compiled(
        graph()
            .host("a", |host| host.bind("k", "1").map(|host| host.entry::<(), i64>("main", Source::Script("$k"))))
            .and_then(|g| {
                g.host("b", |host| host.bind("k", "2").map(|host| host.entry::<(), i64>("main", Source::Script("$k"))))
            })
            .map(|g| g.entry("a", "main").entry("b", "main")),
    );
    let mut storage = MemoryStorage::new();
    assert_eq!(run_unit(&program, &mut storage, "a/main").await, 1);
    assert_eq!(run_unit(&program, &mut storage, "b/main").await, 2);
}

#[test]
fn a_host_named_like_an_extern_namespace_is_not_reached_through_it() {
    let messages = messages(
        graph()
            .host("a", with_entry("main", "string::twice({ n: 1, })"))
            .and_then(|g| g.host("string", with_n_entry("twice", "$n * 2")))
            .map(|g| g.entry("a", "main").entry("string", "twice")),
    );
    assert_eq!(messages, ["type mismatch: expected i64, got string{twice({n: i64})}"]);
}

#[tokio::test]
async fn a_host_calls_its_own_function_by_its_bare_name_when_another_host_has_one_of_that_name() {
    let program = compiled(
        graph()
            .host("a", |host| {
                Ok(host
                    .entry::<(), i64>("f", Source::Script("1"))
                    .entry::<(), i64>("main", Source::Script("f()")))
            })
            .and_then(|g| {
                g.host("b", |host| {
                    Ok(host
                        .entry::<(), i64>("f", Source::Script("2"))
                        .entry::<(), i64>("main", Source::Script("f()")))
                })
            })
            .map(|g| g.entry("a", "main").entry("b", "main")),
    );
    let mut storage = MemoryStorage::new();
    assert_eq!(run_unit(&program, &mut storage, "a/main").await, 1);
    assert_eq!(run_unit(&program, &mut storage, "b/main").await, 2);
}

#[test]
fn a_function_of_another_host_is_not_reached_by_its_bare_name() {
    let messages = messages(
        graph()
            .host("a", with_entry("main", "g()"))
            .and_then(|g| g.host("b", with_entry("g", "2")))
            .map(|g| g.entry("a", "main").entry("b", "g")),
    );
    assert_eq!(messages.len(), 1, "{messages:#?}");
    assert!(messages[0].starts_with("undefined function `g`"), "{messages:#?}");
}

#[tokio::test]
async fn an_entry_of_no_inputs_is_called_with_an_empty_object() {
    let program = compiled(
        graph()
            .host("a", with_entry("main", "seven({})"))
            .and_then(|g| g.host("b", with_entry("seven", "7")))
            .map(|g| g.expose("a", "seven", "b", "seven").entry("a", "main")),
    );
    let mut storage = MemoryStorage::new();
    assert_eq!(run_unit(&program, &mut storage, "a/main").await, 7);
}
