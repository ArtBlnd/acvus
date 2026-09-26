//! RFC-0101 at the host's contract: a hook is a dynamic extern whose body is
//! a closure the host binds. The closure is lent the call's arguments, fills
//! the result through the call's `Output`, may wait and may capture the
//! host's state, and is handed no context.

use std::future::Future;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use acvus_extern::{extern_fn, extern_registry};
use acvus_interpreter::{
    AcvusRuntime, AsyncAccess, Cause, Host, HookEffect, HookName, HookPart, HostError, HostGraph, MemoryStorage, Named,
    Origin, Program, SequentialExecutor, Source,
};
use futures::channel::oneshot;

const ANSWER_DELAY: Duration = Duration::from_millis(2);

fn compiled(host: Host) -> Program {
    match host.compile(SequentialExecutor) {
        Ok(program) => program,
        Err(error) => panic!("the program is refused: {error}"),
    }
}

fn compiled_waited(host: Host<AsyncAccess>) -> Program<AsyncAccess> {
    match host.compile(SequentialExecutor) {
        Ok(program) => program,
        Err(error) => panic!("the program is refused: {error}"),
    }
}

fn host() -> Host {
    Host::new(acvus_ext::std_registries())
}

async fn run_i64(program: &Program) -> Result<i64, HostError> {
    program
        .scope(async |s| {
            let mut storage = MemoryStorage::new();
            let mut page = s.open(&mut storage);
            let entry = s.entry::<(), i64>("main")?;
            entry.run(&mut page, ()).await?.with(|n: &i64| *n)
        })
        .await
}

async fn run_string(program: &Program) -> Result<String, HostError> {
    program
        .scope(async |s| {
            let mut storage = MemoryStorage::new();
            let mut page = s.open(&mut storage);
            let entry = s.entry::<(), String>("main")?;
            entry.run(&mut page, ()).await?.with(|text: &String| text.clone())
        })
        .await
}

fn alone(hook: &str) -> HookName {
    HookName {
        host: None,
        hook: hook.to_owned(),
    }
}

fn of_host(host: &str, hook: &str) -> HookName {
    HookName {
        host: Some(host.to_owned()),
        hook: hook.to_owned(),
    }
}

fn refused_part(error: HostError) -> HookPart {
    let HostError::Refused(refusals) = error else {
        panic!("expected a refusal, got {error}");
    };
    let [refusal] = refusals.as_slice() else {
        panic!("a binding is refused once, and this one {} times", refusals.len());
    };
    match &refusal.cause {
        Some(Cause::Hook { part, .. }) => part.clone(),
        other => panic!("a binding's refusal names its hook, and this one has {other:?}"),
    }
}

// -- Lent arguments ------------------------------------------------------

#[tokio::test]
async fn the_closure_reads_its_lent_arguments_and_writes_through_a_mut_one() {
    let program = compiled(host().hook("bump", 3, HookEffect::Opaque).entry::<(), i64>(
        "main",
        Source::Script(
            r#"let n = 5;
let s = "ab".to_string();
let got = match bump(&mut n, &s, 7) { Some(v) => v, None => -1 };
n * 1000 + got * 10 + s.len() as i64"#,
        ),
    ));
    program
        .bind::<3, _>("bump", |mut args, mut out| {
            Box::pin(async move {
                let len = args.with(1, |s: &String| s.len() as i64).expect("argument 1 is a `&String`");
                let step = args.with(2, |step: &i64| *step).expect("argument 2 is an `i64`");
                args.with_mut(0, |n: &mut i64| *n += step).expect("argument 0 is a `&mut i64`");
                let n = args.with(0, |n: &i64| *n).expect("argument 0 is a `&mut i64`");
                out.write(n + len);
                out.finish()
            })
        })
        .expect("`bump` takes three arguments");
    let ran = run_i64(&program).await.expect("the run ends");
    assert_eq!(ran, 12_000 + 140 + 2, "`n` is 12 in the caller, the result 14, and `s` is lent untouched");
}

#[derive(Debug, PartialEq)]
struct Seen {
    as_string: Option<String>,
    past_the_end: Option<i64>,
    as_i64: Option<i64>,
    len: usize,
}

#[tokio::test]
async fn a_lent_argument_read_at_another_type_or_past_the_end_is_none() {
    let program = compiled(host().hook("probe", 1, HookEffect::Pure).entry::<(), i64>(
        "main",
        Source::Script("match probe(3) { Some(v) => v, None => -1 }"),
    ));
    let seen = Arc::new(Mutex::new(None));
    let seen_in_body = Arc::clone(&seen);
    program
        .bind::<1, _>("probe", move |args, mut out| {
            *seen_in_body.lock().expect("no test panicked holding the lock") = Some(Seen {
                as_string: args.with(0, |s: &String| s.clone()),
                past_the_end: args.with(1, |n: &i64| *n),
                as_i64: args.with(0, |n: &i64| *n),
                len: args.len(),
            });
            Box::pin(async move {
                out.write(0i64);
                out.finish()
            })
        })
        .expect("`probe` takes one argument");
    assert_eq!(run_i64(&program).await.expect("the run ends"), 0);
    let expected = Seen {
        as_string: None,
        past_the_end: None,
        as_i64: Some(3),
        len: 1,
    };
    assert_eq!(*seen.lock().expect("the run ended"), Some(expected));
}

// -- The result ----------------------------------------------------------

#[tokio::test]
async fn one_closure_fills_a_site_of_its_type_and_is_none_at_a_site_of_another() {
    let program = compiled(host().hook("answer", 1, HookEffect::Pure).entry::<(), String>(
        "main",
        Source::Script(
            r#"let n = match answer(4) { Some(v) => v, None => -1 };
let t = match answer(5) { Some(t) => t, None => "none".to_string() };
t + " " + n.to_string()"#,
        ),
    ));
    program
        .bind::<1, _>("answer", |args, mut out| {
            Box::pin(async move {
                let n = args.with(0, |n: &i64| *n).expect("argument 0 is an `i64` at both sites");
                out.write(n * 10);
                out.finish()
            })
        })
        .expect("`answer` takes one argument");
    let text = run_string(&program).await.expect("the run ends");
    assert_eq!(text, "none 40", "the `String` site sees `None`, the `i64` site the value");
}

#[tokio::test]
async fn an_object_result_is_filled_field_by_field() {
    let program = compiled(host().hook("pair", 1, HookEffect::Pure).entry::<(), i64>(
        "main",
        Source::Script(
            r#"let p = match pair(6) { Some(p) => p, None => { x: -1, y: -1, } };
p.x * 100 + p.y"#,
        ),
    ));
    program
        .bind::<1, _>("pair", |args, mut out| {
            Box::pin(async move {
                let n = args.with(0, |n: &i64| *n).expect("argument 0 is an `i64`");
                out.field("x", |x| x.write(n));
                out.field("y", |y| y.write(n + 1));
                out.finish()
            })
        })
        .expect("`pair` takes one argument");
    assert_eq!(run_i64(&program).await.expect("the run ends"), 607);
}

#[tokio::test]
async fn a_hook_of_no_arguments_is_handed_unit() {
    let program = compiled(host().hook("seed", 0, HookEffect::Idempotent).entry::<(), i64>(
        "main",
        Source::Script("match seed() { Some(v) => v, None => -1 }"),
    ));
    program
        .bind::<0, _>("seed", |(), mut out| {
            Box::pin(async move {
                out.write(42i64);
                out.finish()
            })
        })
        .expect("`seed` takes no argument");
    assert_eq!(run_i64(&program).await.expect("the run ends"), 42);
}

// -- Waiting and host state ----------------------------------------------

#[tokio::test]
async fn two_hooks_in_one_script_one_waiting_one_counting_into_host_state() {
    let program = compiled(
        host()
            .hook("ask", 1, HookEffect::Opaque)
            .hook("tally", 1, HookEffect::Opaque)
            .entry::<(), i64>(
                "main",
                Source::Script(
                    r#"let total = 0;
for i in 1..4 {
    let a = match ask(i) { Some(v) => v, None => -1 };
    let t = match tally(a) { Some(v) => v, None => -1 };
    total = total + t;
}
total"#,
                ),
            ),
    );
    program
        .bind::<1, _>("ask", |args, mut out| {
            Box::pin(async move {
                let n = args.with(0, |n: &i64| *n).expect("argument 0 is an `i64`");
                tokio::time::sleep(ANSWER_DELAY).await;
                out.write(n * 10);
                out.finish()
            })
        })
        .expect("`ask` takes one argument");
    let seen = Arc::new(AtomicI64::new(0));
    let calls = Arc::new(Mutex::new(Vec::new()));
    let seen_in_body = Arc::clone(&seen);
    let calls_in_body = Arc::clone(&calls);
    program
        .bind::<1, _>("tally", move |args, mut out| {
            let seen = Arc::clone(&seen_in_body);
            let calls = Arc::clone(&calls_in_body);
            Box::pin(async move {
                let n = args.with(0, |n: &i64| *n).expect("argument 0 is an `i64`");
                calls.lock().expect("no test panicked holding the lock").push(n);
                out.write(seen.fetch_add(n, Ordering::SeqCst) + n);
                out.finish()
            })
        })
        .expect("`tally` takes one argument");
    let ran = run_i64(&program).await.expect("the run ends");
    assert_eq!(ran, 10 + 30 + 60, "each `tally` returns the running total the host keeps");
    assert_eq!(seen.load(Ordering::SeqCst), 60, "the host reads its counter after the run");
    assert_eq!(*calls.lock().expect("the run ended"), [10, 20, 30]);
}

#[tokio::test]
async fn the_hook_call_suspends_the_caller_until_the_closure_ends() {
    let program = compiled(host().hook("ask", 0, HookEffect::Opaque).entry::<(), i64>(
        "main",
        Source::Script("match ask() { Some(n) => n + 1, None => -1 }"),
    ));
    let (send, receive) = oneshot::channel::<i64>();
    let gate = Arc::new(Mutex::new(Some(receive)));
    program
        .bind::<0, _>("ask", move |(), mut out| {
            let receiver = gate.lock().expect("the gate").take();
            Box::pin(async move {
                let word = receiver.expect("`ask` is called once").await.expect("the test sends the word");
                out.write(word);
                out.finish()
            })
        })
        .expect("`ask` takes no argument");

    let mut run = Box::pin(run_i64(&program));
    let waker = futures::task::noop_waker();
    let mut polling = Context::from_waker(&waker);
    for _ in 0..3 {
        assert!(
            matches!(run.as_mut().poll(&mut polling), Poll::Pending),
            "the caller waits while the closure does"
        );
    }
    send.send(41).expect("the closure is waiting on the word");
    assert_eq!(run.await.expect("the run ends"), 42);
}

// -- Binding -------------------------------------------------------------

#[tokio::test]
async fn a_program_with_an_unbound_hook_refuses_to_run_naming_it() {
    let program = compiled(
        host()
            .hook("bound", 0, HookEffect::Pure)
            .hook("unbound", 0, HookEffect::Pure)
            .entry::<(), i64>("main", Source::Script("match bound() { Some(v) => v, None => -1 }")),
    );
    program
        .bind::<0, _>("bound", |(), mut out| {
            Box::pin(async move {
                out.write(1i64);
                out.finish()
            })
        })
        .expect("`bound` takes no argument");
    match run_i64(&program).await {
        Err(HostError::Unbound { hook }) => assert_eq!(hook, alone("unbound")),
        other => panic!("a program with an unbound hook does not run, and this one gave {other:?}"),
    }
}

#[tokio::test]
async fn a_page_load_whose_init_calls_an_unbound_hook_ends_unbound() {
    let program = compiled_waited(
        host()
            .async_access()
            .hook("seed", 0, HookEffect::Pure)
            .init("n", Source::Script("match seed() { Some(v) => v, None => -1 }"))
            .entry::<(), i64>("main", Source::Script("@n")),
    );
    let loaded = program
        .scope(async |s| {
            let mut storage = MemoryStorage::new();
            let mut page = s.open(&mut storage);
            page.with("n", |n: &i64| *n).await
        })
        .await;
    match loaded {
        Err(HostError::Unbound { hook }) => assert_eq!(hook, alone("seed")),
        other => panic!("an init calling an unbound hook does not run, and this one gave {other:?}"),
    }
}

#[test]
fn a_binding_is_refused_for_another_arity_a_second_time_and_an_undeclared_hook() {
    let program = compiled(host().hook("pair", 2, HookEffect::Pure).entry::<(), i64>(
        "main",
        Source::Script("match pair(1, 2) { Some(v) => v, None => -1 }"),
    ));
    let one = program.bind::<1, _>("pair", |_, out| Box::pin(async move { out.finish() }));
    assert_eq!(refused_part(one.expect_err("`pair` takes two arguments")), HookPart::Arity);
    let two = || program.bind::<2, _>("pair", |_, out| Box::pin(async move { out.finish() }));
    two().expect("`pair` takes two arguments");
    assert_eq!(refused_part(two().expect_err("`pair` is bound already")), HookPart::Bound);
    match program.bind::<0, _>("absent", |(), out| Box::pin(async move { out.finish() })) {
        Err(HostError::NotInGraph {
            what: Named::Hook(hook),
        }) => assert_eq!(hook, alone("absent")),
        other => panic!("an undeclared hook is not in the graph, and binding it gave {other:?}"),
    }
}

// -- A hook of a graph's host (RFC-0101 rule 2) ---------------------------

fn graph() -> HostGraph {
    HostGraph::new(acvus_ext::std_registries::<AcvusRuntime>())
}

fn compiled_graph(graph: Result<HostGraph, HostError>) -> Program {
    match graph.and_then(|graph| graph.compile(SequentialExecutor)) {
        Ok(program) => program,
        Err(error) => panic!("the graph is refused: {error}"),
    }
}

async fn run_hosted(program: &Program, entry: &str) -> Result<i64, HostError> {
    program
        .scope(async |s| {
            let mut storage = MemoryStorage::new();
            let mut page = s.open(&mut storage);
            let entry = s.entry::<(), i64>(entry)?;
            entry.run(&mut page, ()).await?.with(|n: &i64| *n)
        })
        .await
}

fn doubling_an_i64_and_marking_a_string(
) -> impl for<'c> Fn(acvus_interpreter::HookArgs<'c, 1>, acvus_interpreter::HookOutput<'c>) -> futures::future::BoxFuture<'c, acvus_interpreter::HookFinished<'c>>
+ Send
+ Sync
+ 'static {
    |args, mut out| {
        Box::pin(async move {
            if let Some(n) = args.with(0, |n: &i64| *n) {
                out.write(n * 2);
            } else if let Some(text) = args.with(0, |text: &String| text.clone()) {
                out.write(text + "!");
            }
            out.finish()
        })
    }
}

#[tokio::test]
async fn a_hook_of_a_graphs_host_is_called_at_sites_of_two_types_and_bound_by_its_host() {
    let program = compiled_graph(
        graph()
            .host("a", |host| {
                Ok(host.hook("echo", 1, HookEffect::Pure).entry::<(), i64>(
                    "main",
                    Source::Script(
                        r#"let n = match echo(20) { Some(v) => v, None => -1 };
let t = match echo("ab".to_string()) { Some(t) => t, None => "".to_string() };
n + t.len() as i64"#,
                    ),
                ))
            })
            .map(|g| g.entry("a", "main")),
    );
    match program.bind::<1, _>("echo", doubling_an_i64_and_marking_a_string()) {
        Err(HostError::NotInGraph {
            what: Named::Hook(hook),
        }) => assert_eq!(hook, alone("echo")),
        other => panic!("a graph's hook is in its host's scope, and `bind` by its name alone gave {other:?}"),
    }
    program.bind_in::<1, _>("a", "echo", doubling_an_i64_and_marking_a_string()).expect("`a`'s `echo` takes one argument");
    assert_eq!(run_hosted(&program, "a/main").await.expect("the run ends"), 40 + 3);
}

fn answering(n: i64) -> impl for<'c> Fn((), acvus_interpreter::HookOutput<'c>) -> futures::future::BoxFuture<'c, acvus_interpreter::HookFinished<'c>>
+ Send
+ Sync
+ 'static {
    move |(), mut out| {
        Box::pin(async move {
            out.write(n);
            out.finish()
        })
    }
}

const CALLS_H: &str = "match h() { Some(v) => v, None => -1 }";

#[tokio::test]
async fn two_hosts_hooks_of_one_name_are_two_hooks_each_reaching_its_own_closure() {
    let declares_h = |host: Host| Ok(host.hook("h", 0, HookEffect::Pure).entry::<(), i64>("main", Source::Script(CALLS_H)));
    let program = compiled_graph(
        graph()
            .host("a", declares_h)
            .and_then(|g| g.host("b", declares_h))
            .map(|g| g.entry("a", "main").entry("b", "main")),
    );
    program.bind_in::<0, _>("a", "h", answering(1)).expect("`a`'s `h` takes no argument");
    program.bind_in::<0, _>("b", "h", answering(2)).expect("`b`'s `h` is another hook, unbound");
    assert_eq!(run_hosted(&program, "a/main").await.expect("the run ends"), 1);
    assert_eq!(run_hosted(&program, "b/main").await.expect("the run ends"), 2);
    let again = program.bind_in::<0, _>("a", "h", answering(3));
    assert_eq!(refused_part(again.expect_err("`a`'s `h` is bound already")), HookPart::Bound);
}

#[test]
fn a_script_of_another_host_does_not_reach_a_hosts_hook() {
    let graph = graph()
        .host("a", |host| Ok(host.hook("h", 0, HookEffect::Pure).entry::<(), i64>("main", Source::Script(CALLS_H))))
        .and_then(|g| g.host("b", |host| Ok(host.entry::<(), i64>("main", Source::Script(CALLS_H)))))
        .map(|g| g.entry("a", "main").entry("b", "main"));
    match graph.and_then(|graph| graph.compile(SequentialExecutor)) {
        Err(HostError::Refused(refusals)) => {
            let [refusal] = refusals.as_slice() else {
                panic!("`b`'s call is refused once, and here {} times: {refusals:#?}", refusals.len());
            };
            assert_eq!(refusal.origin, Some(Origin::Entry("b/main".to_owned())), "{refusal:?}");
            assert!(refusal.message.starts_with("undefined function `h`"), "{}", refusal.message);
        }
        Err(other) => panic!("a graph is refused with its refusals, not {other:?}"),
        Ok(_) => panic!("`b`'s script reached `a`'s hook"),
    }
}

fn declaring_h(host: Host) -> Result<Host, HostError> {
    Ok(host.hook("h", 0, HookEffect::Pure).entry::<(), i64>("main", Source::Script(CALLS_H)))
}

#[test]
fn an_entry_named_like_a_hook_of_its_own_host_is_refused_as_shadowing_it() {
    let graph = graph()
        .host("a", |host| declaring_h(host).map(|host| host.entry::<(), i64>("h", Source::Script("0"))))
        .map(|g| g.entry("a", "main"));
    match graph.and_then(|graph| graph.compile(SequentialExecutor)) {
        Err(HostError::Refused(refusals)) => {
            let [refusal] = refusals.as_slice() else {
                panic!("the entry is refused once, and here {} times: {refusals:#?}", refusals.len());
            };
            assert_eq!(refusal.origin, Some(Origin::Entry("a/h".to_owned())), "{refusal:?}");
            assert_eq!(refusal.cause, Some(Cause::Shadows { externs: vec!["h".to_owned()] }));
        }
        Err(other) => panic!("a graph is refused with its refusals, not {other:?}"),
        Ok(_) => panic!("an entry of `a` shadowed `a`'s hook"),
    }
}

#[tokio::test]
async fn an_entry_named_like_a_hook_of_another_host_is_admitted() {
    let program = compiled_graph(
        graph()
            .host("a", declaring_h)
            .and_then(|g| g.host("b", |host| Ok(host.entry::<(), i64>("h", Source::Script("7")))))
            .map(|g| g.entry("a", "main").entry("b", "h")),
    );
    program.bind_in::<0, _>("a", "h", answering(1)).expect("`a`'s `h` takes no argument");
    assert_eq!(run_hosted(&program, "a/main").await.expect("the run ends"), 1);
    assert_eq!(run_hosted(&program, "b/h").await.expect("the run ends"), 7);
}

#[tokio::test]
async fn an_unbound_hook_of_a_graphs_host_refuses_the_run_naming_host_and_hook() {
    let program = compiled_graph(
        graph()
            .host("a", |host| Ok(host.hook("h", 0, HookEffect::Pure).entry::<(), i64>("main", Source::Script(CALLS_H))))
            .and_then(|g| {
                g.host("b", |host| {
                    Ok(host
                        .hook("h", 0, HookEffect::Pure)
                        .entry::<(), i64>("main", Source::Script("0")))
                })
            })
            .map(|g| g.entry("a", "main").entry("b", "main")),
    );
    program.bind_in::<0, _>("a", "h", answering(1)).expect("`a`'s `h` takes no argument");
    match run_hosted(&program, "a/main").await {
        Err(error @ HostError::Unbound { .. }) => {
            let HostError::Unbound { hook } = &error else { unreachable!() };
            assert_eq!(*hook, of_host("b", "h"));
            assert!(error.to_string().starts_with("the hook `h` of the host `b` is unbound"), "{error}");
        }
        other => panic!("a graph with an unbound hook does not run, and this one gave {other:?}"),
    }
}

#[derive(acvus_extern::TyArg)]
pub struct N {
    n: i64,
}

fn counting(total: Arc<AtomicI64>) -> impl for<'c> Fn(acvus_interpreter::HookArgs<'c, 1>, acvus_interpreter::HookOutput<'c>) -> futures::future::BoxFuture<'c, acvus_interpreter::HookFinished<'c>>
+ Send
+ Sync
+ 'static {
    move |args, mut out| {
        let total = Arc::clone(&total);
        Box::pin(async move {
            let n = args.with(0, |n: &i64| *n).expect("argument 0 is an `i64`");
            out.write(total.fetch_add(n, Ordering::SeqCst) + n);
            out.finish()
        })
    }
}

fn exposing_a_hooked_entry(effect: HookEffect) -> Result<HostGraph, HostError> {
    graph()
        .host("a", |host| Ok(host.entry::<(), i64>("main", Source::Script("tick({ n: 1, }) * 10 + tick({ n: 1, })"))))
        .and_then(|g| {
            g.host("b", |host| {
                Ok(host
                    .hook("tally", 1, effect)
                    .entry::<N, i64>("tick", Source::Script("match tally($n) { Some(v) => v, None => -1 }")))
            })
        })
        .map(|g| g.expose("a", "tick", "b", "tick").entry("a", "main"))
}

#[tokio::test]
async fn a_hook_called_through_an_exposed_entry_runs_at_each_call_its_effect_orders() {
    let program = compiled_graph(exposing_a_hooked_entry(HookEffect::Opaque));
    let total = Arc::new(AtomicI64::new(0));
    program
        .bind_in::<1, _>("b", "tally", counting(Arc::clone(&total)))
        .expect("`b`'s `tally` takes one argument");
    assert_eq!(run_hosted(&program, "a/main").await.expect("the run ends"), 12);
    assert_eq!(total.load(Ordering::SeqCst), 2, "an opaque hook runs at each of the two calls");
}

fn orders_its_caller(effect: HookEffect) -> bool {
    let program = compiled_graph(exposing_a_hooked_entry(effect));
    match program.listing("a/main") {
        Ok(listing) => listing.mir.contains("; orders:"),
        Err(error) => panic!("`a/main` is compiled: {error}"),
    }
}

#[test]
fn a_hooks_declared_effect_reaches_the_caller_of_the_exposed_entry_that_calls_it() {
    assert!(orders_its_caller(HookEffect::Opaque), "an opaque hook orders `a/main` through `tick`");
    assert!(!orders_its_caller(HookEffect::Pure), "a pure hook orders nothing");
}

// -- mem2reg across a hook call (RFC-0101 rule 3) ------------------------

/// The hook `ask`'s place, taken by an ordinary extern of the same effect
/// and task.
#[extern_fn(effect = opaque)]
async fn ask(n: i64) -> Option<i64> {
    Some(n)
}

const AROUND_A_CALL: &str = r#"@n = @n + 1;
let got = match ask(@n) { Some(v) => v, None => 0 };
@n = @n + got;
@n"#;

fn listed(program: &Program) -> String {
    match program.listing("main") {
        Ok(listing) => listing.mir,
        Err(error) => panic!("`main` is compiled: {error}"),
    }
}

#[test]
fn a_context_is_promoted_across_a_hook_call_as_across_an_extern_call() {
    let hooked = compiled(
        host()
            .hook("ask", 1, HookEffect::Opaque)
            .init("n", Source::Expr("0"))
            .entry::<(), i64>("main", Source::Script(AROUND_A_CALL)),
    );
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(extern_registry! { ns: "stand_in", fns: [ask], });
    let externed = compiled(
        Host::new(registries)
            .init("n", Source::Expr("0"))
            .entry::<(), i64>("main", Source::Script(AROUND_A_CALL)),
    );
    let hooked = listed(&hooked);
    let externed = listed(&externed);
    assert_eq!(hooked, externed, "the hook's body is the extern's, instruction for instruction");
    let fetches = hooked.lines().filter(|line| line.contains("fetch")).count();
    assert_eq!(fetches, 1, "`@n` is fetched once:\n{hooked}");
    let commits = hooked.lines().filter(|line| line.contains("commit")).count();
    assert_eq!(commits, 1, "`@n` is committed once:\n{hooked}");
    let at = |op: &str| {
        hooked
            .lines()
            .position(|line| line.contains(op))
            .unwrap_or_else(|| panic!("the body holds a `{op}`:\n{hooked}"))
    };
    assert!(
        at("fetch @n") < at("spawn") && at("eval") < at("commit @n"),
        "`@n` is fetched before the call and committed after it, held in a register across it:\n{hooked}"
    );
}

#[test]
fn a_context_is_promoted_across_a_hook_call_of_a_graphs_host_as_across_an_extern_call() {
    let hooked = compiled_graph(
        graph()
            .host("a", |host| {
                Ok(host
                    .hook("ask", 1, HookEffect::Opaque)
                    .init("n", Source::Expr("0"))
                    .entry::<(), i64>("main", Source::Script(AROUND_A_CALL)))
            })
            .map(|g| g.entry("a", "main")),
    );
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(extern_registry! { ns: "stand_in", fns: [ask], });
    let externed = compiled_graph(
        HostGraph::new(registries)
            .host("a", |host| {
                Ok(host
                    .init("n", Source::Expr("0"))
                    .entry::<(), i64>("main", Source::Script(AROUND_A_CALL)))
            })
            .map(|g| g.entry("a", "main")),
    );
    let listed = |program: &Program| match program.listing("a/main") {
        Ok(listing) => listing.mir,
        Err(error) => panic!("`a/main` is compiled: {error}"),
    };
    assert_eq!(
        listed(&hooked),
        listed(&externed),
        "the graph's hook call is the extern's, instruction for instruction"
    );
}
