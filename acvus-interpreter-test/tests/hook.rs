//! RFC-0101 at the host's contract: a hook one program declares, bound to a
//! lent entry of another program compiled apart, runs that entry within the
//! call on the call's own arguments; an entry that waits suspends the caller
//! until it ends.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::task::{Context, Poll};
use std::time::Duration;

use acvus_extern::{Registry, TyArg, extern_fn, extern_registry};
use acvus_interpreter::{
    AcvusRuntime, Cause, Host, HookEffect, HookPart, HostError, InputShape, LentInputs, MemoryStorage, Names,
    Program, Refusal, SequentialExecutor, Source,
};
use futures::channel::oneshot;
use futures::lock::Mutex;

#[derive(TyArg)]
pub struct Pair {
    x: i64,
    y: i64,
}

#[derive(TyArg)]
pub enum Mode {
    Idle,
    Busy,
}

/// `Mode` with one variant, for an entry narrower than its caller.
mod narrow {
    use acvus_extern::TyArg;

    #[derive(TyArg)]
    pub enum Mode {
        Busy,
    }
}

#[derive(TyArg)]
pub enum Shape {
    Dot,
    Spot(Pair),
}

/// Stands for a model's answer: it waits before it gives `x` back.
#[extern_fn(effect = opaque)]
async fn slow(x: i64) -> i64 {
    tokio::time::sleep(Duration::from_millis(5)).await;
    x
}

/// Answers once the test sends the word: the entry waits on the test itself.
#[extern_fn(effect = opaque)]
async fn released(#[state] gate: &Arc<StdMutex<Option<oneshot::Receiver<i64>>>>) -> i64 {
    let receiver = gate.lock().expect("the gate").take().expect("`released` is called once");
    receiver.await.expect("the test sends the word")
}

fn waiting_registries() -> Vec<Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries();
    registries.push(extern_registry! { ns: "wait", fns: [slow], });
    registries
}

fn waiting_host(names: &Names) -> Host {
    Host::with_names(names, waiting_registries())
}

fn storage() -> Arc<Mutex<MemoryStorage>> {
    Arc::new(Mutex::new(MemoryStorage::new()))
}

fn compiled(host: Host) -> Program {
    match host.compile(SequentialExecutor) {
        Ok(program) => program,
        Err(error) => panic!("the program is refused: {error}"),
    }
}

fn host(names: &Names) -> Host {
    Host::with_names(names, acvus_ext::std_registries())
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
            entry.run(&mut page, ()).await?.with(|t: &String| t.clone())
        })
        .await
}

fn bind_over_fresh_storage(caller: &Program, hook: &str, callee: &Program, entry: &str) -> Result<(), HostError> {
    let lent = callee.lent(entry)?;
    caller.bind(hook, &lent, |call| call.run(&storage()))
}

fn refusals(error: HostError) -> Vec<Refusal> {
    match error {
        HostError::Refused(refusals) => refusals,
        other => panic!("expected refusals, got {other}"),
    }
}

fn parts(refusals: &[Refusal]) -> Vec<HookPart> {
    refusals
        .iter()
        .map(|refusal| match &refusal.cause {
            Some(Cause::Hook { part, .. }) => part.clone(),
            other => panic!("a binding refusal names its hook, and this one has {other:?}"),
        })
        .collect()
}

const DESCRIBE: &str = r#"let total = $n + $xs[0] + $xs[1] + $xs[2] + $p.x + $p.y;
string::concat(&$s, &total.to_string())"#;

fn describer(names: &Names) -> Program {
    compiled(host(names).lent_entry::<String>(
        "describe",
        LentInputs::new()
            .field::<i64>("n")
            .field::<String>("s")
            .field::<[i64; 3]>("xs")
            .field::<Pair>("p"),
        Source::Script(DESCRIBE),
    ))
}

const DESCRIBE_CALL: &str = r#"let s = "total=".to_string();
match describe(2, s, [1, 2, 3], { x: 10, y: 20, }) { Some(t) => t, None => "none".to_string() }"#;

#[tokio::test]
async fn a_round_trip_lends_a_scalar_a_string_an_array_and_an_object_and_fills_the_result() {
    let names = Names::new();
    let caller = compiled(
        host(&names)
            .hook("describe", 4, HookEffect::Pure)
            .entry::<(), String>("main", Source::Script(DESCRIBE_CALL)),
    );
    let callee = describer(&names);
    bind_over_fresh_storage(&caller, "describe", &callee, "describe").expect("every argument and the result match");
    for _ in 0..64 {
        let described = run_string(&caller).await.expect("the run ends");
        assert_eq!(described, "total=38", "2 + 1 + 2 + 3 + 10 + 20, after the lent string");
    }
}

#[tokio::test]
async fn a_write_through_a_mut_argument_lands_in_the_callers_storage() {
    let names = Names::new();
    let caller = compiled(host(&names).hook("bump", 2, HookEffect::Opaque).entry::<(), i64>(
        "main",
        Source::Script(
            r#"let n = 5;
let s = "a".to_string();
let got = match bump(&mut n, &mut s) { Some(v) => v, None => -1 };
n * 100 + got * 10 + s.len() as i64"#,
        ),
    ));
    let callee = compiled(host(&names).lent_entry::<i64>(
        "bump",
        LentInputs::new().field_mut::<i64>("n").field_mut::<String>("s"),
        Source::Script(
            r#"*$n = *$n + 1;
*$s = "changed".to_string();
*$n"#,
        ),
    ));
    bind_over_fresh_storage(&caller, "bump", &callee, "bump").expect("both references match");
    let ran = run_i64(&caller).await.expect("the run ends");
    assert_eq!(ran, 600 + 60 + 7, "`n` is 6 in the caller, the entry returned 6, and `s` is \"changed\"");
}

#[tokio::test]
async fn an_argument_of_another_type_refuses_the_binding_naming_the_site_and_position() {
    let names = Names::new();
    let caller = compiled(
        host(&names).hook("describe", 4, HookEffect::Pure).entry::<(), String>(
            "main",
            Source::Script(
                r#"match describe("2".to_string(), "s".to_string(), [1, 2, 3], { x: 10, y: 20, }) { Some(t) => t, None => "none".to_string() }"#,
            ),
        ),
    );
    let callee = describer(&names);
    let refused = refusals(bind_over_fresh_storage(&caller, "describe", &callee, "describe").expect_err("argument 0 differs"));
    assert_eq!(parts(&refused), [HookPart::Argument(0)]);
    let refusal = &refused[0];
    assert_eq!(refusal.origin, Some(acvus_interpreter::Origin::Entry("main".to_owned())));
    assert!(refusal.span.is_some(), "the refusal points at the call");
    assert!(
        refusal.message.contains("argument 0") && refusal.message.contains("$n: i64"),
        "{}",
        refusal.message
    );
}

#[tokio::test]
async fn a_result_of_another_type_refuses_the_binding() {
    let names = Names::new();
    let caller = compiled(
        host(&names)
            .hook("describe", 4, HookEffect::Pure)
            .entry::<(), i64>(
                "main",
                Source::Script(
                    r#"match describe(2, "s".to_string(), [1, 2, 3], { x: 10, y: 20, }) { Some(t) => t, None => 0 }"#,
                ),
            ),
    );
    let callee = describer(&names);
    let refused = refusals(bind_over_fresh_storage(&caller, "describe", &callee, "describe").expect_err("the result differs"));
    assert_eq!(parts(&refused), [HookPart::Result]);
}

#[tokio::test]
async fn an_extension_type_refuses_the_binding() {
    let names = Names::new();
    let caller = compiled(host(&names).hook("total", 1, HookEffect::Pure).entry::<(), i64>(
        "main",
        Source::Script("match total(vec([1, 2])) { Some(n) => n, None => 0 }"),
    ));
    let callee = compiled(host(&names).lent_entry::<i64>(
        "total",
        LentInputs::new().field::<Vec<i64>>("v"),
        Source::Script("$v.len() as i64"),
    ));
    let refused = refusals(bind_over_fresh_storage(&caller, "total", &callee, "total").expect_err("`Vec` is an extension type"));
    assert_eq!(parts(&refused), [HookPart::Argument(0)]);
    assert!(refused[0].message.contains("extension type"), "{}", refused[0].message);
}

const COUNTING: &str = "@count = @count + 1;\n@count";

fn counter(names: &Names) -> Program {
    compiled(
        host(names)
            .init("count", Source::Expr("0"))
            .lent_entry::<i64>("tally", LentInputs::new(), Source::Script(COUNTING)),
    )
}

fn count_caller(names: &Names, effect: HookEffect) -> Program {
    compiled(host(names).hook("count", 0, effect).entry::<(), i64>(
        "main",
        Source::Script("match count() { Some(n) => n, None => -1 }"),
    ))
}

#[tokio::test]
async fn an_entry_that_writes_its_context_is_refused_under_a_pure_hook_and_bound_under_an_opaque_one() {
    let names = Names::new();
    let callee = counter(&names);

    let pure = count_caller(&names, HookEffect::Pure);
    let refused = refusals(bind_over_fresh_storage(&pure, "count", &callee, "tally").expect_err("a write is opaque to the caller"));
    assert_eq!(parts(&refused), [HookPart::Effect]);

    let opaque = count_caller(&names, HookEffect::Opaque);
    let kept = storage();
    let lent = callee.lent("tally").expect("`tally` is lent");
    opaque
        .bind("count", &lent, move |call| call.run(&kept))
        .expect("an opaque hook admits a write");
    assert_eq!(run_i64(&opaque).await.expect("the run ends"), 1);
    assert_eq!(run_i64(&opaque).await.expect("the run ends"), 2, "the callee's storage kept `@count`");
}

#[test]
fn an_entry_that_moves_its_input_is_refused_naming_the_input() {
    let names = Names::new();
    let refused = host(&names)
        .lent_entry::<String>("echo", LentInputs::new().field::<String>("s"), Source::Script("$s"))
        .compile(SequentialExecutor)
        .map(|_| ())
        .expect_err("the body returns its input");
    let refused = refusals(refused);
    assert!(
        refused.iter().any(|refusal| refusal.message.contains("moves its input `$s`")),
        "{:?}",
        refused.iter().map(|refusal| &refusal.message).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn a_hook_bound_to_its_own_programs_entry_traps_at_the_call() {
    let names = Names::new();
    let program = compiled(
        host(&names)
            .hook("again", 1, HookEffect::Opaque)
            .entry::<(), i64>("main", Source::Script("match again(1) { Some(n) => n, None => -1 }"))
            .lent_entry::<i64>("once_more", LentInputs::new().field::<i64>("n"), Source::Script("$n + 1")),
    );
    bind_over_fresh_storage(&program, "again", &program, "once_more").expect("the types match; re-entry is a run's matter");
    let Err(HostError::Trapped { message }) = run_i64(&program).await else {
        panic!("the run re-enters its own program");
    };
    assert!(message.contains("already running"), "{message}");
}

#[tokio::test]
async fn a_cycle_between_two_programs_traps_where_it_closes() {
    let names = Names::new();
    let a = compiled(
        host(&names)
            .hook("to_b", 1, HookEffect::Opaque)
            .entry::<(), i64>("main", Source::Script("match to_b(1) { Some(n) => n, None => -1 }"))
            .lent_entry::<i64>("back", LentInputs::new().field::<i64>("n"), Source::Script("$n * 10")),
    );
    let b = compiled(
        host(&names)
            .hook("to_a", 1, HookEffect::Pure)
            .lent_entry::<i64>(
                "forward",
                LentInputs::new().field::<i64>("n"),
                Source::Script("match to_a($n + 1) { Some(n) => n, None => -1 }"),
            ),
    );
    bind_over_fresh_storage(&a, "to_b", &b, "forward").expect("the types match");
    bind_over_fresh_storage(&b, "to_a", &a, "back").expect("the types match");
    let Err(HostError::Trapped { message }) = run_i64(&a).await else {
        panic!("`back` is `a`'s, which is running");
    };
    assert!(message.contains("to_a") && message.contains("already running"), "{message}");
}

#[tokio::test]
async fn a_program_with_an_unbound_hook_is_refused_naming_it() {
    let names = Names::new();
    let caller = compiled(
        host(&names)
            .hook("describe", 4, HookEffect::Pure)
            .entry::<(), String>("main", Source::Script(DESCRIBE_CALL)),
    );
    match run_string(&caller).await {
        Err(HostError::Unbound { hook }) => assert_eq!(hook, "describe"),
        other => panic!("expected the unbound hook, got {other:?}"),
    }
}

#[tokio::test]
async fn a_result_two_sites_settle_differently_refuses_at_the_site_the_entry_does_not_fit() {
    let names = Names::new();
    let source = r#"let a = match halve(8) { Some(n) => n, None => 0 };
let b = match halve(6) { Some(t) => t, None => "none".to_string() };
a + b.len() as i64"#;
    let caller = compiled(host(&names).hook("halve", 1, HookEffect::Pure).entry::<(), i64>("main", Source::Script(source)));
    let callee = compiled(host(&names).lent_entry::<i64>(
        "halve",
        LentInputs::new().field::<i64>("n"),
        Source::Script("$n / 2"),
    ));
    let refused = refusals(bind_over_fresh_storage(&caller, "halve", &callee, "halve").expect_err("the second site settles `String`"));
    assert_eq!(parts(&refused), [HookPart::Result]);
    let span = refused[0].span.expect("the refusal points at the call");
    assert!(source[span.start..span.end].contains("halve(6)"), "{:?}", &source[span.start..span.end]);
}

#[tokio::test]
async fn programs_over_two_tables_of_names_are_refused() {
    let caller = compiled(
        host(&Names::new())
            .hook("describe", 4, HookEffect::Pure)
            .entry::<(), String>("main", Source::Script(DESCRIBE_CALL)),
    );
    let callee = describer(&Names::new());
    let refused = refusals(bind_over_fresh_storage(&caller, "describe", &callee, "describe").expect_err("two tables"));
    assert_eq!(parts(&refused), [HookPart::SeparateNames]);
}

#[tokio::test]
async fn a_lent_entry_is_no_entry_a_host_runs() {
    let callee = describer(&Names::new());
    let refused = callee
        .scope(async |s| s.entry_shaped::<String>("describe").map(|_| ()))
        .await
        .expect_err("a lent entry takes what a hook lends");
    assert!(matches!(refused, HostError::Mismatched { .. }), "{refused}");
}

#[test]
fn a_lent_entry_called_as_a_function_is_refused() {
    let names = Names::new();
    let refused = host(&names)
        .lent_entry::<i64>("inner", LentInputs::new().field::<i64>("n"), Source::Script("$n"))
        .entry_shaped::<i64>("main", InputShape::new().field::<i64>("n"), Source::Script("inner()"))
        .compile(SequentialExecutor)
        .map(|_| ())
        .expect_err("a lent entry runs only from a hook");
    let refused = refusals(refused);
    assert!(
        refused.iter().any(|refusal| refusal.message.contains("the lent entry `inner` is called here")),
        "{:?}",
        refused.iter().map(|refusal| &refusal.message).collect::<Vec<_>>()
    );
}

#[test]
fn two_live_copies_of_a_mut_input_are_refused() {
    let names = Names::new();
    let refused = host(&names)
        .lent_entry::<i64>(
            "twice",
            LentInputs::new().field_mut::<i64>("n"),
            Source::Script("let a = $n;\nlet b = $n;\n*a = 1;\n*b = 2;\n0"),
        )
        .compile(SequentialExecutor)
        .map(|_| ())
        .expect_err("`a` is live where `b` copies the same `&mut`");
    let refused = refusals(refused);
    assert!(
        refused.iter().any(|refusal| refusal.message.contains("while a reference to it is live")),
        "{:?}",
        refused.iter().map(|refusal| &refusal.message).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn an_enum_argument_and_a_result_result_cross_by_their_tags() {
    let names = Names::new();
    let caller = compiled(host(&names).hook("classify", 1, HookEffect::Pure).entry::<(), i64>(
        "main",
        Source::Script(
            r#"let m = if 1 > 2 { Mode::Idle } else { Mode::Busy };
match classify(m) {
    Some(r) => match r { Ok(n) => n, Err(e) => 0 - string::len(&e) as i64 },
    None => -100,
}"#,
        ),
    ));
    let callee = compiled(host(&names).lent_entry::<Result<i64, String>>(
        "classify",
        LentInputs::new().field::<Mode>("m"),
        Source::Script(r#"match &$m { Mode::Idle => Ok(7), Mode::Busy => Err("busy".to_string()) }"#),
    ));
    bind_over_fresh_storage(&caller, "classify", &callee, "classify").expect("`Mode{Busy, Idle}` both sides");
    assert_eq!(run_i64(&caller).await.expect("the run ends"), -4, "`Busy` is `Err(\"busy\")`");
}

#[tokio::test]
async fn an_entry_that_calls_another_hook_runs() {
    let names = Names::new();
    let a = compiled(
        host(&names)
            .hook("to_b", 1, HookEffect::Opaque)
            .entry::<(), i64>("main", Source::Script("match to_b(1) { Some(n) => n, None => -1 }")),
    );
    let b = compiled(host(&names).hook("to_c", 1, HookEffect::Opaque).lent_entry::<i64>(
        "forward",
        LentInputs::new().field::<i64>("n"),
        Source::Script("match to_c($n + 1) { Some(n) => n * 10, None => -1 }"),
    ));
    let c = compiled(waiting_host(&names).lent_entry::<i64>(
        "answer",
        LentInputs::new().field::<i64>("n"),
        Source::Script("slow($n + 100)"),
    ));
    bind_over_fresh_storage(&a, "to_b", &b, "forward").expect("an entry that calls a hook may wait");
    bind_over_fresh_storage(&b, "to_c", &c, "answer").expect("an entry that waits is bound");
    assert_eq!(run_i64(&a).await.expect("the run ends"), 1020, "(1 + 1 + 100) * 10");
}

// -- A waiting entry (RFC-0101 rule 2) ------------------------------------

#[tokio::test]
async fn an_entry_that_waits_runs_and_the_caller_waits_for_it() {
    let names = Names::new();
    let caller = compiled(host(&names).hook("ask", 1, HookEffect::Opaque).entry::<(), i64>(
        "main",
        Source::Script("match ask(20) { Some(n) => n + 1, None => -1 }"),
    ));
    let callee = compiled(waiting_host(&names).lent_entry::<i64>(
        "ask",
        LentInputs::new().field::<i64>("n"),
        Source::Script("slow($n * 2)"),
    ));
    bind_over_fresh_storage(&caller, "ask", &callee, "ask").expect("an entry that waits is bound");
    assert_eq!(run_i64(&caller).await.expect("the run ends"), 41);
}

/// The caller's run is polled while the entry waits on a word only the test
/// sends: it is pending, not polled once and given up, and it ends with the
/// entry's answer once the word arrives.
#[tokio::test]
async fn the_hook_call_suspends_the_caller_until_the_entry_ends() {
    let names = Names::new();
    let (send, receive) = oneshot::channel();
    let gate = Arc::new(StdMutex::new(Some(receive)));
    let mut registries = acvus_ext::std_registries();
    registries.push(extern_registry! { ns: "wait", fns: [released(Arc::clone(&gate))], });
    let caller = compiled(host(&names).hook("ask", 0, HookEffect::Opaque).entry::<(), i64>(
        "main",
        Source::Script("match ask() { Some(n) => n + 1, None => -1 }"),
    ));
    let callee = compiled(
        Host::with_names(&names, registries).lent_entry::<i64>("ask", LentInputs::new(), Source::Script("released()")),
    );
    bind_over_fresh_storage(&caller, "ask", &callee, "ask").expect("an entry that waits is bound");

    let mut run = Box::pin(run_i64(&caller));
    let waker = futures::task::noop_waker();
    let mut polling = Context::from_waker(&waker);
    for _ in 0..3 {
        assert!(
            matches!(run.as_mut().poll(&mut polling), Poll::Pending),
            "the caller waits while the entry does"
        );
    }
    send.send(41).expect("the entry is waiting on the word");
    assert_eq!(run.await.expect("the run ends"), 42);
}

#[tokio::test]
async fn two_hook_calls_in_one_caller_each_wait_for_their_entry() {
    let names = Names::new();
    let caller = compiled(host(&names).hook("ask", 1, HookEffect::Opaque).entry::<(), i64>(
        "main",
        Source::Script(
            r#"let a = match ask(1) { Some(n) => n, None => -1 };
let b = match ask(2) { Some(n) => n, None => -1 };
a * 100 + b"#,
        ),
    ));
    let callee = compiled(waiting_host(&names).lent_entry::<i64>(
        "ask",
        LentInputs::new().field::<i64>("n"),
        Source::Script("slow($n * 3)"),
    ));
    bind_over_fresh_storage(&caller, "ask", &callee, "ask").expect("an entry that waits is bound");
    assert_eq!(run_i64(&caller).await.expect("the run ends"), 306);
}

#[tokio::test]
async fn a_cycle_through_a_waiting_entry_traps_where_it_closes() {
    let names = Names::new();
    let a = compiled(
        host(&names)
            .hook("to_b", 1, HookEffect::Opaque)
            .entry::<(), i64>("main", Source::Script("match to_b(1) { Some(n) => n, None => -1 }"))
            .lent_entry::<i64>("back", LentInputs::new().field::<i64>("n"), Source::Script("$n * 10")),
    );
    let b = compiled(waiting_host(&names).hook("to_a", 1, HookEffect::Opaque).lent_entry::<i64>(
        "forward",
        LentInputs::new().field::<i64>("n"),
        Source::Script("let m = slow($n + 1);\nmatch to_a(m) { Some(n) => n, None => -1 }"),
    ));
    bind_over_fresh_storage(&a, "to_b", &b, "forward").expect("the types match");
    bind_over_fresh_storage(&b, "to_a", &a, "back").expect("the types match");
    let Err(HostError::Trapped { message }) = run_i64(&a).await else {
        panic!("`back` is `a`'s, which is running below the waiting `forward`");
    };
    assert!(message.contains("to_a") && message.contains("already running"), "{message}");
}

// -- Types within (RFC-0101 rule 3) ---------------------------------------

#[tokio::test]
async fn a_site_building_one_variant_passes_it_to_an_entry_taking_more() {
    let names = Names::new();
    let caller = compiled(host(&names).hook("classify", 1, HookEffect::Pure).entry::<(), i64>(
        "main",
        Source::Script("match classify(Mode::Busy) { Some(n) => n, None => -1 }"),
    ));
    let callee = compiled(host(&names).lent_entry::<i64>(
        "classify",
        LentInputs::new().field::<Mode>("m"),
        Source::Script("match &$m { Mode::Idle => 1, Mode::Busy => 2 }"),
    ));
    bind_over_fresh_storage(&caller, "classify", &callee, "classify").expect("`Mode{Busy}` is within `Mode{Busy, Idle}`");
    assert_eq!(run_i64(&caller).await.expect("the run ends"), 2);
}

#[tokio::test]
async fn a_site_building_more_variants_than_the_entry_takes_is_refused_naming_the_site_and_position() {
    let names = Names::new();
    let source = r#"let m = if 1 > 2 { Mode::Idle } else { Mode::Busy };
match classify(m) { Some(n) => n, None => -1 }"#;
    let caller = compiled(host(&names).hook("classify", 1, HookEffect::Pure).entry::<(), i64>("main", Source::Script(source)));
    let callee = compiled(host(&names).lent_entry::<i64>(
        "classify",
        LentInputs::new().field::<narrow::Mode>("m"),
        Source::Script("match &$m { Mode::Busy => 2 }"),
    ));
    let refused =
        refusals(bind_over_fresh_storage(&caller, "classify", &callee, "classify").expect_err("`Idle` is not the entry's"));
    assert_eq!(parts(&refused), [HookPart::Argument(0)]);
    let refusal = &refused[0];
    assert_eq!(refusal.origin, Some(acvus_interpreter::Origin::Entry("main".to_owned())));
    let span = refusal.span.expect("the refusal points at the call");
    assert!(source[span.start..span.end].contains("classify(m)"), "{:?}", &source[span.start..span.end]);
    assert!(refusal.message.contains("argument 0") && refusal.message.contains("not within"), "{}", refusal.message);
}

#[tokio::test]
async fn an_entry_result_with_fewer_variants_is_within_the_sites() {
    let names = Names::new();
    let caller = compiled(host(&names).hook("mode", 0, HookEffect::Pure).entry::<(), i64>(
        "main",
        Source::Script("match mode() { Some(m) => match m { Mode::Idle => 1, Mode::Busy => 2 }, None => -1 }"),
    ));
    let callee = compiled(host(&names).lent_entry::<narrow::Mode>("mode", LentInputs::new(), Source::Script("Mode::Busy")));
    bind_over_fresh_storage(&caller, "mode", &callee, "mode").expect("`Mode{Busy}` is within `Mode{Busy, Idle}`");
    assert_eq!(run_i64(&caller).await.expect("the run ends"), 2);

    let narrow_site = compiled(host(&names).hook("mode", 0, HookEffect::Pure).entry::<(), i64>(
        "main",
        Source::Script("match mode() { Some(m) => match m { Mode::Busy => 2 }, None => -1 }"),
    ));
    let wide = compiled(host(&names).lent_entry::<Mode>("mode", LentInputs::new(), Source::Script("Mode::Busy")));
    let refused = refusals(bind_over_fresh_storage(&narrow_site, "mode", &wide, "mode").expect_err("`Idle` has no arm"));
    assert_eq!(parts(&refused), [HookPart::Result]);
}

#[tokio::test]
async fn an_enum_laid_at_another_width_is_refused() {
    let names = Names::new();
    let caller = compiled(host(&names).hook("area", 1, HookEffect::Pure).entry::<(), i64>(
        "main",
        Source::Script("match area(Shape::Dot) { Some(n) => n, None => -1 }"),
    ));
    let callee = compiled(host(&names).lent_entry::<i64>(
        "area",
        LentInputs::new().field::<Shape>("s"),
        Source::Script("match &$s { Shape::Dot => 0, Shape::Spot(p) => p.x * p.y }"),
    ));
    let refused = refusals(bind_over_fresh_storage(&caller, "area", &callee, "area").expect_err("`Spot` widens the layout"));
    assert_eq!(parts(&refused), [HookPart::Argument(0)]);
    assert!(refused[0].message.contains("width"), "{}", refused[0].message);
}

#[tokio::test]
async fn behind_a_mut_reference_an_enum_is_the_same_on_both_sides() {
    let names = Names::new();
    let caller = compiled(host(&names).hook("flip", 1, HookEffect::Opaque).entry::<(), i64>(
        "main",
        Source::Script("let m = Mode::Busy;\nmatch flip(&mut m) { Some(n) => n, None => -1 }"),
    ));
    let callee = compiled(host(&names).lent_entry::<i64>(
        "flip",
        LentInputs::new().field_mut::<Mode>("m"),
        Source::Script("*$m = Mode::Idle;\n0"),
    ));
    let refused = refusals(
        bind_over_fresh_storage(&caller, "flip", &callee, "flip").expect_err("the entry could write `Idle` into `Mode{Busy}`"),
    );
    assert_eq!(parts(&refused), [HookPart::Argument(0)]);
}

// -- No strong cycle (RFC-0101 rule 4) ------------------------------------

/// Counts its drops: a binding's closure owns one, so it is released exactly
/// when the program holding the binding is.
struct Dropped(Arc<AtomicUsize>);

impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn two_programs_bound_to_each_other_are_both_released_with_their_hosts() {
    let names = Names::new();
    let dropped = Arc::new(AtomicUsize::new(0));
    let program = |hook: &str, entry: &str| {
        compiled(
            host(&names)
                .hook(hook, 1, HookEffect::Pure)
                .entry::<(), i64>("main", Source::Script(&format!("match {hook}(1) {{ Some(n) => n, None => -1 }}")))
                .lent_entry::<i64>(entry, LentInputs::new().field::<i64>("n"), Source::Script("$n + 1")),
        )
    };
    let a = program("to_b", "from_b");
    let b = program("to_a", "from_a");
    for (caller, hook, callee, entry) in [(&a, "to_b", &b, "from_a"), (&b, "to_a", &a, "from_b")] {
        let lent = callee.lent(entry).expect("the entry is lent");
        let kept = Dropped(Arc::clone(&dropped));
        let over = storage();
        caller
            .bind(hook, &lent, move |call| {
                let _owned = &kept;
                call.run(&over)
            })
            .expect("the types match");
    }
    assert_eq!(run_i64(&a).await.expect("the run ends"), 2);
    assert_eq!(dropped.load(Ordering::SeqCst), 0, "both programs are held");
    drop(a);
    assert_eq!(dropped.load(Ordering::SeqCst), 1, "`a`'s binding is released with `a`");
    let Err(HostError::Unbound { hook }) = run_i64(&b).await else {
        panic!("`b`'s hook is bound to `a`, which the host released");
    };
    assert_eq!(hook, "to_a");
    drop(b);
    assert_eq!(dropped.load(Ordering::SeqCst), 2, "`b`'s binding is released with `b`");
}
