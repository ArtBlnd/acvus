//! RFC-0101 at the host's contract: a hook one program declares, bound to a
//! lent entry of another program compiled apart, runs that entry within the
//! call on the call's own arguments.

use acvus_extern::TyArg;
use acvus_interpreter::{
    Cause, Host, HookEffect, HookPart, HostError, InputShape, LentInputs, MemoryStorage, Names, Program, Refusal,
    SequentialExecutor, Source,
};

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
    caller.bind(hook, &lent, |call| call.run(&mut MemoryStorage::new()))
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
    let storage = std::sync::Arc::new(std::sync::Mutex::new(MemoryStorage::new()));
    let kept = std::sync::Arc::clone(&storage);
    let lent = callee.lent("tally").expect("`tally` is lent");
    opaque
        .bind("count", &lent, move |call| {
            let mut storage = kept.lock().expect("no call panicked holding the storage");
            call.run(&mut *storage)
        })
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

#[test]
fn an_entry_that_calls_an_opaque_hook_can_wait_and_is_refused() {
    let names = Names::new();
    let a = compiled(
        host(&names)
            .hook("to_b", 1, HookEffect::Opaque)
            .entry::<(), i64>("main", Source::Script("match to_b(1) { Some(n) => n, None => -1 }")),
    );
    let b = compiled(host(&names).hook("to_c", 1, HookEffect::Opaque).lent_entry::<i64>(
        "forward",
        LentInputs::new().field::<i64>("n"),
        Source::Script("match to_c($n + 1) { Some(n) => n, None => -1 }"),
    ));
    let refused = refusals(bind_over_fresh_storage(&a, "to_b", &b, "forward").expect_err("the spawned call waits"));
    assert_eq!(parts(&refused), [HookPart::Effect]);
    assert!(refused[0].message.contains("can wait"), "{}", refused[0].message);
}
