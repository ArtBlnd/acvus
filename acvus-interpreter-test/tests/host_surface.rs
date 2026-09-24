//! RFC-0090 at the host's contract: entries declared by Rust type, contexts
//! whose type the graph solves, and pages read and written through Rust
//! types.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use acvus_extern::{
    Ctx, Declared, Erased, ExternType, Registry, TyArg, Var, extern_fn, extern_registry, kind,
};
use acvus_interpreter::{
    AcvusRuntime, EntryError, EntryPart, Host, InMemoryContext, Origin, OutputError, PageError,
    Plain, Program, Refusal, SequentialExecutor, Source, Space, SpacePage,
};
use acvus_mir::graph::{Bindings, BoundValue};

type Rt = AcvusRuntime;
use acvus_utils::Interner;

#[derive(TyArg)]
#[projection]
pub struct Profile {
    name: String,
    age: i64,
}

#[extern_fn(effect = pure)]
fn profile(name: String, age: i64) -> Profile {
    Profile { name, age }
}

#[derive(TyArg)]
pub enum Mode {
    Idle,
    Busy,
}

/// `#[projection]` on an enum none of whose variants has a payload does not
/// compile (E0392: the projection's lifetime is unused), so the read through
/// a projection is shown on an enum with one payload.
#[derive(TyArg)]
#[projection]
pub enum Job {
    Idle,
    Busy(i64),
}

static RELEASES: AtomicUsize = AtomicUsize::new(0);

static COUNTED_RUNS: Mutex<()> = Mutex::new(());

pub struct Counted;

impl Drop for Counted {
    fn drop(&mut self) {
        RELEASES.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(ExternType)]
#[repr(transparent)]
pub struct Tracked(Vec<Counted>);

#[extern_fn(effect = pure)]
fn tracked(n: i64) -> Tracked {
    Tracked((0..n).map(|_| Counted).collect())
}

static BUMPS: AtomicUsize = AtomicUsize::new(0);

static BUMPED_RUNS: Mutex<()> = Mutex::new(());

#[extern_fn(effect = opaque)]
fn bump() {
    BUMPS.fetch_add(1, Ordering::SeqCst);
}

#[extern_fn(effect = pure)]
fn coin() -> bool {
    true
}

static INIT_RUNS: AtomicUsize = AtomicUsize::new(0);

static INIT_COUNTED: Mutex<()> = Mutex::new(());

#[extern_fn(effect = opaque)]
fn init_ran() {
    INIT_RUNS.fetch_add(1, Ordering::SeqCst);
}

/// A value that is one source: `history()` begins a new one at every call.
#[derive(ExternType)]
#[repr(transparent)]
pub struct History<I>(Vec<i64>, PhantomData<I>)
where
    I: Var<kind::Identity>;

#[extern_fn(effect = pure)]
fn history<I>() -> History<I>
where
    I: Var<kind::Identity>,
{
    History(Vec::new(), PhantomData)
}

#[extern_fn(effect = pure)]
fn record<I>(h: &mut History<I>, n: i64)
where
    I: Var<kind::Identity>,
{
    h.0.push(n);
}

#[extern_fn(effect = pure)]
fn recorded<I>(h: &History<I>) -> i64
where
    I: Var<kind::Identity>,
{
    h.0.len() as i64
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(extern_registry! {
        ns: "host",
        types: [Tracked, History<_>],
        fns: [profile, tracked, bump, coin, init_ran, history, record, recorded],
    });
    registries
}

fn compiled(host: Host) -> Program {
    match host.compile(Arc::new(SequentialExecutor)) {
        Ok(program) => program,
        Err(refusals) => panic!("the program is refused: {refusals:?}"),
    }
}

fn host(i: &Interner) -> Host {
    Host::new(i, registries())
}

fn exclusive(page: &mut Arc<InMemoryContext>) -> &mut InMemoryContext {
    Arc::get_mut(page).expect("no run holds the page after it returns")
}

fn solved(program: &Program, i: &Interner, key: &str) -> String {
    program.contexts().solved()[key].display(i).to_string()
}

/// `@log`'s strings, each copied in Rust out of the element the page lends.
fn log_of(page: &mut InMemoryContext) -> Result<Vec<String>, PageError> {
    page.with("log", |ctx: &mut Ctx<'_, Rt>, xs: &[Erased<Rt, String>]| {
        xs.iter().map(|x| x.as_ref(ctx.rt).to_owned()).collect()
    })
}

/// An `i64` context, copied out of the page.
fn int_of(page: &mut InMemoryContext, key: &str) -> Result<i64, PageError> {
    page.with(key, |n: &i64| *n)
}

async fn run_unit(program: &Program, name: &str, page: &Arc<InMemoryContext>) {
    let entry = program.entry::<(), ()>(name).expect("the entry returns `()`");
    entry.run(page, ()).await.expect("the page holds the graph's contexts");
}

const INIT_LOG: &str = "@log = vec([]);";
const PUSH_LOG: &str = r#"@log.push("x".to_string());"#;

fn log_program(i: &Interner) -> Program {
    compiled(
        host(i)
            .entry::<(), ()>("init", Source::Script(INIT_LOG))
            .entry::<(), ()>("turn", Source::Script(PUSH_LOG)),
    )
}

#[tokio::test]
async fn a_script_stores_the_context_its_turn_scripts_push_to() {
    let i = Interner::new();
    let program = log_program(&i);
    assert_eq!(solved(&program, &i, "log"), "Vec<String>");
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));

    run_unit(&program, "init", &page).await;
    run_unit(&program, "turn", &page).await;
    run_unit(&program, "turn", &page).await;

    let log = log_of(exclusive(&mut page)).expect("`@log` holds a `Vec<String>`");
    assert_eq!(log, ["x", "x"]);
}

#[test]
fn two_stores_of_one_enum_context_solve_to_the_union_of_their_variants() {
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .entry::<(), ()>("init", Source::Script("@mode = Mode::Idle;"))
            .entry::<(), ()>("turn", Source::Script("@mode = Mode::Busy;")),
    );
    assert_eq!(solved(&program, &i, "mode"), "Mode{Busy, Idle}");
    let declared = <Mode as Declared>::declared(&i).display(&i).to_string();
    assert_eq!(declared, "Mode{Busy, Idle}");
}

#[tokio::test]
async fn an_enum_context_two_entries_store_is_read_through_its_projection() {
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .entry::<(), ()>("init", Source::Script("@job = Job::Idle;"))
            .entry::<(), ()>("turn", Source::Script("@job = Job::Busy(7);")),
    );
    assert_eq!(solved(&program, &i, "job"), "Job{Busy(i64), Idle}");
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));

    run_unit(&program, "init", &page).await;
    let idle = exclusive(&mut page)
        .with("job", |job: JobRef<'_>| matches!(job, JobRef::Idle))
        .expect("`@job` holds a `Job`");
    assert!(idle);
    run_unit(&program, "turn", &page).await;
    let busy = exclusive(&mut page)
        .with("job", |job: JobRef<'_>| match job {
            JobRef::Busy(n) => Some(*n),
            JobRef::Idle => None,
        })
        .expect("`@job` holds a `Job`");
    assert_eq!(busy, Some(7));
}

#[tokio::test]
async fn a_context_the_graph_leaves_open_closes_to_never_and_no_rust_type_reads_it() {
    let i = Interner::new();
    let program = compiled(host(&i).entry::<(), ()>("init", Source::Script("@xs = [];")));
    assert_eq!(solved(&program, &i, "xs"), "Array<!, 0>");
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    run_unit(&program, "init", &page).await;

    let read = exclusive(&mut page).with("xs", |_: &[Erased<Rt, String>]| ());
    let Err(PageError::Mismatched { key, held, asked }) = read else {
        panic!("a `&[String]` borrow of `Array<!, 0>` is refused, and it gave {read:?}")
    };
    assert_eq!(key, "xs");
    assert_eq!(held, "Array<!, 0>");
    assert_eq!(asked, "&[String]");
}

#[tokio::test]
async fn a_key_no_run_stored_yet_is_absent_and_the_page_is_unchanged() {
    let i = Interner::new();
    let program = log_program(&i);
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));

    assert!(matches!(log_of(exclusive(&mut page)), Err(PageError::Absent { .. })));
    assert!(matches!(
        exclusive(&mut page).with_mut("log", |_: &mut [Erased<Rt, String>]| ()),
        Err(PageError::Absent { .. })
    ));
    assert!(matches!(
        exclusive(&mut page).with("elsewhere", |_: &[Erased<Rt, String>]| ()),
        Err(PageError::NotInGraph { .. })
    ));
    assert!(matches!(log_of(exclusive(&mut page)), Err(PageError::Absent { .. })));
}

#[tokio::test]
async fn a_read_or_an_insert_at_a_wrong_type_is_refused_and_the_page_is_unchanged() {
    let i = Interner::new();
    let program = log_program(&i);
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    run_unit(&program, "init", &page).await;
    run_unit(&program, "turn", &page).await;

    assert!(matches!(
        exclusive(&mut page).with("log", |_: &[Erased<Rt, i64>]| ()),
        Err(PageError::Mismatched { .. })
    ));
    assert!(matches!(
        exclusive(&mut page).insert("log", 5_i64),
        Err(PageError::Mismatched { .. })
    ));
    assert!(matches!(
        exclusive(&mut page).insert("elsewhere", vec!["y".to_owned()]),
        Err(PageError::NotInGraph { .. })
    ));

    let log = log_of(exclusive(&mut page)).expect("`@log` still holds its `Vec<String>`");
    assert_eq!(log, ["x"]);
}

#[tokio::test]
async fn a_host_insert_is_what_the_next_run_reads() {
    let i = Interner::new();
    let program = log_program(&i);
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    exclusive(&mut page)
        .insert("log", vec!["from the host".to_owned()])
        .expect("`@log` is a `Vec<String>`");

    run_unit(&program, "turn", &page).await;

    let log = log_of(exclusive(&mut page)).expect("`@log` holds a `Vec<String>`");
    assert_eq!(log, ["from the host", "x"]);
}

#[tokio::test]
async fn a_space_that_stores_a_context_at_another_type_opens_no_page() {
    let i = Interner::new();
    let space = Arc::new(Space::new(Plain));
    let text = compiled(host(&i).entry::<(), ()>("init", Source::Script(r#"@n = "a".to_string();"#)));
    let page = Arc::new(
        SpacePage::of(Arc::clone(&space), text.contexts()).expect("the space holds nothing"),
    );
    let entry = text.entry::<(), ()>("init").expect("the entry returns `()`");
    entry.run(&page, ()).await.expect("the page holds `@n`");
    page.commit_opened().expect("the space takes `@n`");

    let number = compiled(host(&i).entry::<(), ()>("init", Source::Script("@n = 1;")));
    let opened = SpacePage::of(Arc::clone(&space), number.contexts()).map(|_| ());
    let Err(PageError::Mismatched { key, held, asked }) = opened else {
        panic!("a space storing `@n` as `String` opened for an `i64` `@n`: {opened:?}")
    };
    assert_eq!(key, "n");
    assert_eq!(held, "String");
    assert_eq!(asked, "i64");
}

#[tokio::test]
async fn a_page_that_holds_a_context_at_another_type_does_not_run() {
    let i = Interner::new();
    let program = compiled(host(&i).entry::<(), i64>("main", Source::Script("@n")).entry::<(), ()>(
        "init",
        Source::Script("@n = 1;"),
    ));
    let other = compiled(host(&i).entry::<(), ()>("init", Source::Script(r#"@n = "a".to_string();"#)));
    let page = Arc::new(InMemoryContext::of(other.contexts()));

    let entry = program.entry::<(), i64>("main").expect("the entry returns `i64`");
    assert!(matches!(
        entry.run(&page, ()).await,
        Err(PageError::Mismatched { .. })
    ));
}

#[tokio::test]
async fn a_string_result_is_read_and_edited_through_its_output() {
    let i = Interner::new();
    let program = compiled(host(&i).entry::<(), String>("main", Source::Script(r#""hello".to_string()"#)));
    let page = Arc::new(InMemoryContext::of(program.contexts()));

    let entry = program.entry::<(), String>("main").expect("the entry returns `String`");
    let mut output = entry.run(&page, ()).await.expect("the graph has no context");
    let kept: String = output.with(|s: &str| s.to_owned()).expect("the result is a `String`");
    assert_eq!(kept, "hello");
    output
        .with_mut(|s: &mut String| s.push_str(", host"))
        .expect("the result is a `String`");
    assert_eq!(output.with(|s: &String| s.clone()).expect("the result is a `String`"), "hello, host");
    assert_eq!(kept, "hello", "the copy is the host's own");
}

#[tokio::test]
async fn a_structural_result_is_read_and_edited_through_its_projection() {
    let i = Interner::new();
    let program = compiled(host(&i).entry::<(), Profile>(
        "main",
        Source::Script(r#"profile("ann".to_string(), 41)"#),
    ));
    let page = Arc::new(InMemoryContext::of(program.contexts()));

    let entry = program.entry::<(), Profile>("main").expect("the entry returns `Profile`");
    let mut output = entry.run(&page, ()).await.expect("the graph has no context");
    let read = |p: ProfileRef<'_>| (p.name.clone(), *p.age);
    assert_eq!(output.with(read).expect("the result is a `Profile`"), ("ann".to_owned(), 41));
    output
        .with_mut(|p: ProfileMut<'_>| {
            *p.age += 1;
            p.name.push('e');
        })
        .expect("the result is a `Profile`");
    assert_eq!(output.with(read).expect("the result is a `Profile`"), ("anne".to_owned(), 42));
}

#[test]
fn an_entry_that_returns_another_type_is_refused_at_compile_time() {
    let i = Interner::new();
    let refused = host(&i)
        .entry::<(), String>("main", Source::Script("1 + 1"))
        .compile(Arc::new(SequentialExecutor));

    let Err(refusals) = refused else {
        panic!("an `i64` tail against a declared `String` compiled")
    };
    assert!(
        refusals
            .iter()
            .all(|refusal| refusal.origin == Some(Origin::Entry("main".to_owned()))),
        "{refusals:?}"
    );
    assert!(!refusals.is_empty());
}

#[test]
fn an_entry_asked_for_at_another_type_than_it_declared_is_refused_before_it_runs() {
    let i = Interner::new();
    let program = compiled(host(&i).entry::<(), String>("main", Source::Script(r#""a".to_string()"#)));

    assert!(matches!(
        program.entry::<(), i64>("main").map(|_| ()),
        Err(EntryError::Mismatched { .. })
    ));
    assert!(matches!(
        program.entry::<(), String>("elsewhere").map(|_| ()),
        Err(EntryError::NotInGraph { .. })
    ));
}

struct Counting {
    at_start: usize,
    _runs: MutexGuard<'static, ()>,
}

impl Counting {
    fn start() -> Self {
        let runs = COUNTED_RUNS.lock().unwrap_or_else(PoisonError::into_inner);
        Self {
            at_start: RELEASES.load(Ordering::SeqCst),
            _runs: runs,
        }
    }

    fn released(&self) -> usize {
        RELEASES.load(Ordering::SeqCst) - self.at_start
    }
}

#[tokio::test]
async fn dropping_an_output_releases_its_value_once() {
    const ELEMENTS: usize = 3;
    let counting = Counting::start();
    let i = Interner::new();
    let program = compiled(host(&i).entry::<(), Tracked>(
        "main",
        Source::Script(&format!("tracked({ELEMENTS})")),
    ));
    let page = Arc::new(InMemoryContext::of(program.contexts()));

    let entry = program.entry::<(), Tracked>("main").expect("the entry returns `Tracked`");
    let output = entry.run(&page, ()).await.expect("the graph has no context");
    assert_eq!(output.with(|t: &Tracked| t.0.len()).expect("the result is a `Tracked`"), ELEMENTS);
    assert_eq!(counting.released(), 0, "the output holds every element");
    drop(output);
    assert_eq!(counting.released(), ELEMENTS);
}

#[test]
fn a_container_is_declared_from_its_parts() {
    let i = Interner::new();
    let shown = |ty: acvus_mir::ty::PolyTy| ty.display(&i).to_string();
    assert_eq!(shown(<Option<i64> as Declared>::declared(&i)), "Option<i64>");
    assert_eq!(
        shown(<Result<String, bool> as Declared>::declared(&i)),
        "Result<String, Bool>"
    );
    assert_eq!(shown(<(i64, String) as Declared>::declared(&i)), "(i64, String)");
    assert_eq!(shown(<[u8; 3] as Declared>::declared(&i)), "Array<u8, 3>");
    assert_eq!(
        shown(<Vec<Profile> as Declared>::declared(&i)),
        "Vec<Profile{age: i64, name: String}>"
    );
    assert_eq!(shown(<Mode as Declared>::declared(&i)), "Mode{Busy, Idle}");
}

// -- RFC-0025 rule 2: what a run fetches first ---------------------------

fn fresh_log_program(i: &Interner) -> Program {
    compiled(
        host(i)
            .entry::<(), ()>("init", Source::Script(INIT_LOG))
            .entry::<(), ()>("turn", Source::Script(&format!("bump(); {PUSH_LOG}"))),
    )
}

#[tokio::test]
async fn a_storing_script_runs_on_a_page_that_holds_nothing_and_its_turns_push_to_it() {
    let _bumps = BUMPED_RUNS.lock().unwrap_or_else(PoisonError::into_inner);
    let i = Interner::new();
    let program = fresh_log_program(&i);
    assert_eq!(solved(&program, &i, "log"), "Vec<String>");
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));

    run_unit(&program, "init", &page).await;
    run_unit(&program, "turn", &page).await;
    run_unit(&program, "turn", &page).await;

    let log = log_of(exclusive(&mut page)).expect("`@log` holds a `Vec<String>`");
    assert_eq!(log, ["x", "x"]);
}

#[tokio::test]
async fn a_turn_before_the_script_that_stores_its_context_is_refused_before_any_of_it_runs() {
    let _bumps = BUMPED_RUNS.lock().unwrap_or_else(PoisonError::into_inner);
    let i = Interner::new();
    let program = fresh_log_program(&i);
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    let bumps_before = BUMPS.load(Ordering::SeqCst);

    let turn = program.entry::<(), ()>("turn").expect("the entry returns `()`");
    let refused = turn.run(&page, ()).await.map(|_| ());

    let Err(PageError::Absent { key }) = refused else {
        panic!("a turn that pushes to `@log` ran on a page without `@log`: {refused:?}")
    };
    assert_eq!(key, "log");
    assert_eq!(
        BUMPS.load(Ordering::SeqCst),
        bumps_before,
        "the turn's first call did not run"
    );
    assert!(matches!(log_of(exclusive(&mut page)), Err(PageError::Absent { .. })));

    run_unit(&program, "init", &page).await;
    run_unit(&program, "turn", &page).await;
    assert_eq!(BUMPS.load(Ordering::SeqCst), bumps_before + 1);
}

/// The fetch is the callee's and comes after `bump()`: the refusal is the
/// page check before the run, not the fetch.
#[tokio::test]
async fn a_run_whose_callee_fetches_a_context_the_page_lacks_is_refused_before_its_first_call() {
    let _bumps = BUMPED_RUNS.lock().unwrap_or_else(PoisonError::into_inner);
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .entry::<(), ()>("init", Source::Script(INIT_LOG))
            .entry::<(), ()>("log_x", Source::Script(PUSH_LOG))
            .entry::<(), ()>("turn", Source::Script("bump(); log_x();")),
    );
    let page = Arc::new(InMemoryContext::of(program.contexts()));
    let bumps_before = BUMPS.load(Ordering::SeqCst);

    let turn = program.entry::<(), ()>("turn").expect("the entry returns `()`");
    let refused = turn.run(&page, ()).await.map(|_| ());
    let Err(PageError::Absent { key }) = refused else {
        panic!("a run whose callee fetches `@log` ran without `@log`: {refused:?}")
    };
    assert_eq!(key, "log");
    assert_eq!(BUMPS.load(Ordering::SeqCst), bumps_before);

    run_unit(&program, "init", &page).await;
    run_unit(&program, "turn", &page).await;
    assert_eq!(BUMPS.load(Ordering::SeqCst), bumps_before + 1);
}

#[tokio::test]
async fn a_context_assigned_on_one_branch_only_is_fetched_and_refused_on_an_empty_page() {
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .entry::<(), i64>("main", Source::Script("if coin() { @x = 1; } @x"))
            .entry::<(), ()>("init", Source::Script("@x = 5;")),
    );
    let page = Arc::new(InMemoryContext::of(program.contexts()));
    let main = program.entry::<(), i64>("main").expect("the entry returns `i64`");

    let refused = main.run(&page, ()).await.map(|_| ());
    let Err(PageError::Absent { key }) = refused else {
        panic!("a body that may read `@x` unassigned ran without `@x`: {refused:?}")
    };
    assert_eq!(key, "x");

    run_unit(&program, "init", &page).await;
    let output = main.run(&page, ()).await.expect("the page holds `@x`");
    assert_eq!(output.with(|n: &i64| *n).expect("the result is an `i64`"), 1);
}

/// A callee's store is in its summary as "may write"; it does not assign the
/// caller's variable, so the caller that reads `@x` after the call fetches
/// `@x` first.
#[tokio::test]
async fn a_whole_assignment_inside_a_callee_does_not_spare_the_callers_fetch() {
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .entry::<(), ()>("set_x", Source::Script("@x = 1;"))
            .entry::<(), i64>("main", Source::Script("set_x(); @x")),
    );
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    let main = program.entry::<(), i64>("main").expect("the entry returns `i64`");

    let refused = main.run(&page, ()).await.map(|_| ());
    let Err(PageError::Absent { key }) = refused else {
        panic!("the caller read `@x` without fetching it: {refused:?}")
    };
    assert_eq!(key, "x");

    exclusive(&mut page)
        .insert("x", 7_i64)
        .expect("`@x` is an `i64`");
    let output = main.run(&page, ()).await.expect("the page holds `@x`");
    assert_eq!(
        output.with(|n: &i64| *n).expect("the result is an `i64`"),
        1,
        "the caller reads what the callee stored"
    );
}

#[tokio::test]
async fn a_call_that_writes_a_context_before_the_body_assigns_it_makes_the_body_fetch_it() {
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .entry::<(), ()>("set_x", Source::Script("@x = 1;"))
            .entry::<(), i64>("main", Source::Script("set_x(); @x = 2; @x")),
    );
    let main = program.entry::<(), i64>("main").expect("the entry returns `i64`");

    let empty = Arc::new(InMemoryContext::of(program.contexts()));
    let refused = main.run(&empty, ()).await.map(|_| ());
    let Err(PageError::Absent { key }) = refused else {
        panic!("the body skipped the fetch the call's bracket needs: {refused:?}")
    };
    assert_eq!(key, "x");

    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    exclusive(&mut page)
        .insert("x", 7_i64)
        .expect("`@x` is an `i64`");
    let output = main.run(&page, ()).await.expect("the page holds `@x`");
    assert_eq!(
        output.with(|n: &i64| *n).expect("the result is an `i64`"),
        2,
        "the body's own store is the last"
    );
    drop(output);
    assert_eq!(int_of(exclusive(&mut page), "x").expect("`@x` is held"), 2);
}

#[tokio::test]
async fn a_callee_reads_what_its_caller_assigned_before_the_call() {
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .entry::<(), i64>("read_x", Source::Script("@x"))
            .entry::<(), i64>("main", Source::Script("@x = 3; read_x()")),
    );
    let page = Arc::new(InMemoryContext::of(program.contexts()));

    let main = program.entry::<(), i64>("main").expect("the entry returns `i64`");
    let output = main
        .run(&page, ())
        .await
        .expect("the caller commits `@x` before the callee fetches it");
    assert_eq!(output.with(|n: &i64| *n).expect("the result is an `i64`"), 3);

    let read_x = program.entry::<(), i64>("read_x").expect("the entry returns `i64`");
    let empty = Arc::new(InMemoryContext::of(program.contexts()));
    assert!(matches!(
        read_x.run(&empty, ()).await.map(|_| ()),
        Err(PageError::Absent { .. })
    ));
}

#[tokio::test]
async fn an_insert_of_a_key_no_script_names_is_not_in_the_graph() {
    let i = Interner::new();
    let program = fresh_log_program(&i);
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));

    let refused = exclusive(&mut page).insert("transcript", vec!["kept by the host".to_owned()]);
    let Err(PageError::NotInGraph { key }) = refused else {
        panic!("a key no script names was inserted: {refused:?}")
    };
    assert_eq!(key, "transcript");
}

/// Without a fetch, the value the page held is replaced at the exit's
/// commit, and released there once.
#[tokio::test]
async fn a_store_only_run_over_a_held_value_releases_the_value_it_replaces_once() {
    const ELEMENTS: usize = 3;
    let counting = Counting::start();
    let i = Interner::new();
    let program = compiled(host(&i).entry::<(), ()>(
        "init",
        Source::Script(&format!("@t = tracked({ELEMENTS});")),
    ));
    let page = Arc::new(InMemoryContext::of(program.contexts()));

    run_unit(&program, "init", &page).await;
    assert_eq!(counting.released(), 0, "the page holds every element");
    run_unit(&program, "init", &page).await;
    assert_eq!(counting.released(), ELEMENTS, "the replaced value is released once");
    drop(page);
    assert_eq!(counting.released(), 2 * ELEMENTS);
}

// -- RFC-0090 rules 3-5: a host is lent a value as a handler is ------------

#[tokio::test]
async fn an_element_edited_in_place_is_what_the_next_run_reads() {
    let i = Interner::new();
    let program = log_program(&i);
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    run_unit(&program, "init", &page).await;
    run_unit(&program, "turn", &page).await;

    exclusive(&mut page)
        .with_mut(
            "log",
            |ctx: &mut Ctx<'_, Rt>, xs: &mut [Erased<Rt, String>]| xs[0].as_mut(ctx.rt).push('!'),
        )
        .expect("`@log` holds a `Vec<String>`");
    run_unit(&program, "turn", &page).await;

    let log = log_of(exclusive(&mut page)).expect("`@log` holds a `Vec<String>`");
    assert_eq!(log, ["x!", "x"]);
}

#[tokio::test]
async fn a_derived_context_is_edited_through_its_projection_and_the_next_run_reads_the_edit() {
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .entry::<(), ()>("init", Source::Script(r#"@p = profile("ann".to_string(), 41);"#))
            .entry::<(), i64>("age", Source::Script("@p.age")),
    );
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    run_unit(&program, "init", &page).await;

    exclusive(&mut page)
        .with_mut("p", |p: ProfileMut<'_>| {
            *p.age += 1;
            p.name.push('e');
        })
        .expect("`@p` holds a `Profile`");
    let name = exclusive(&mut page)
        .with("p", |p: ProfileRef<'_>| p.name.clone())
        .expect("`@p` holds a `Profile`");
    assert_eq!(name, "anne");

    let age = program.entry::<(), i64>("age").expect("the entry returns `i64`");
    let output = age.run(&page, ()).await.expect("the page holds `@p`");
    assert_eq!(output.with(|n: &i64| *n).expect("the result is an `i64`"), 42);
}

#[tokio::test]
async fn a_closure_of_another_type_is_refused_before_it_runs_and_the_page_is_unchanged() {
    let i = Interner::new();
    let program = log_program(&i);
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    run_unit(&program, "init", &page).await;
    run_unit(&program, "turn", &page).await;

    let mut ran = false;
    let refused = exclusive(&mut page).with_mut("log", |_: &mut String| ran = true);
    let Err(PageError::Mismatched { key, held, asked }) = refused else {
        panic!("a `&mut String` borrow of `Vec<String>` ran: {refused:?}")
    };
    assert!(!ran, "the closure did not run");
    assert_eq!(key, "log");
    assert_eq!(held, "Vec<String>");
    assert_eq!(asked, "&mut String");
    assert_eq!(log_of(exclusive(&mut page)).expect("`@log` is held"), ["x"]);

    let entry = program.entry::<(), ()>("turn").expect("the entry returns `()`");
    let output = entry.run(&page, ()).await.expect("the page holds `@log`");
    let mut ran = false;
    let refused = output.with(|_: &i64| ran = true);
    assert!(matches!(refused, Err(OutputError::Mismatched { .. })), "{refused:?}");
    assert!(!ran, "the closure did not run");
}

#[tokio::test]
async fn a_string_context_is_kept_as_a_copy_the_host_makes() {
    let i = Interner::new();
    let program = compiled(host(&i).entry::<(), ()>("init", Source::Script(r#"@name = "ann".to_string();"#)));
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    run_unit(&program, "init", &page).await;

    let kept: String = exclusive(&mut page)
        .with("name", |s: &str| s.to_owned())
        .expect("`@name` is a `String`");
    exclusive(&mut page)
        .with_mut("name", |s: &mut String| s.push_str(" smith"))
        .expect("`@name` is a `String`");
    assert_eq!(kept, "ann");
    assert_eq!(
        exclusive(&mut page).with("name", |s: &str| s.to_owned()).expect("`@name` is held"),
        "ann smith"
    );
}

#[tokio::test]
async fn a_space_page_lends_what_it_loads_and_commits_what_it_was_lent() {
    let i = Interner::new();
    let space = Arc::new(Space::new(Plain));
    let program = compiled(
        host(&i)
            .entry::<(), ()>("init", Source::Script(r#"@name = "ann".to_string();"#))
            .entry::<(), String>("name", Source::Script("@name")),
    );
    let page = Arc::new(SpacePage::of(Arc::clone(&space), program.contexts()).expect("empty"));
    run_unit_on(&program, "init", &page).await;
    page.commit_opened().expect("the space takes `@name`");

    let mut reopened = SpacePage::of(Arc::clone(&space), program.contexts()).expect("same types");
    reopened
        .with_mut("name", |s: &mut String| s.push('?'))
        .expect("`@name` loads from the space");
    assert_eq!(
        reopened.with("name", |s: &str| s.to_owned()).expect("`@name` is held"),
        "ann?"
    );
    reopened.commit_opened().expect("the space takes the edit");

    let fresh = Arc::new(SpacePage::of(Arc::clone(&space), program.contexts()).expect("same types"));
    let entry = program.entry::<(), String>("name").expect("the entry returns `String`");
    let output = entry.run(&fresh, ()).await.expect("the space holds `@name`");
    assert_eq!(output.with(|s: &str| s.to_owned()).expect("a `String`"), "ann?");
}

async fn run_unit_on(program: &Program, name: &str, page: &Arc<SpacePage>) {
    let entry = program.entry::<(), ()>(name).expect("the entry returns `()`");
    entry.run(page, ()).await.expect("the page holds the graph's contexts");
}

// -- An entry's `$` inputs are declared by Rust type (RFC-0090 rule 2) --

#[derive(TyArg)]
pub struct N {
    n: i64,
}

#[derive(TyArg)]
pub struct S {
    s: String,
}

#[derive(TyArg)]
pub struct DeclaredAgainstNameOrder {
    b: i64,
    a: i64,
}

#[derive(TyArg)]
pub struct Spare {
    used: i64,
    unused: String,
}

#[derive(TyArg)]
pub struct HeldProfile {
    p: Profile,
}

#[derive(TyArg)]
pub struct Maybe {
    m: Option<i64>,
}

#[derive(TyArg)]
pub struct Wrong {
    n: String,
}

fn refusal_messages(host: Host) -> Vec<String> {
    match host.compile(Arc::new(SequentialExecutor)) {
        Ok(_) => panic!("the program compiled"),
        Err(refusals) => refusals.into_iter().map(|refusal| refusal.message).collect(),
    }
}

async fn int_result<I>(program: &Program, inputs: I) -> i64
where
    I: acvus_extern::Declared + acvus_extern::Cross<Rt, ReturnForm: acvus_extern::Returned<Verdict = ()>>,
{
    let page = Arc::new(InMemoryContext::of(program.contexts()));
    let entry = program.entry::<I, i64>("main").expect("the entry takes `I` and returns `i64`");
    let output = entry.run(&page, inputs).await.expect("the graph has no context");
    output.with(|n: &i64| *n).expect("the result is an `i64`")
}

#[test]
fn a_dollar_read_by_an_entry_declaring_no_inputs_is_refused_at_compile_time() {
    let i = Interner::new();
    let messages = refusal_messages(host(&i).entry::<(), i64>("main", Source::Script("$n + 1")));
    assert!(
        messages.iter().any(|message| message.contains("`$n`")),
        "{messages:?}"
    );
}

#[tokio::test]
async fn an_input_crosses_into_the_parameter_its_field_names() {
    let i = Interner::new();
    let program = compiled(host(&i).entry::<N, i64>("main", Source::Script("$n + 1")));
    assert_eq!(int_result(&program, N { n: 41 }).await, 42);
}

#[tokio::test]
async fn a_text_input_crosses_as_the_string_it_is() {
    let i = Interner::new();
    let program = compiled(host(&i).entry::<S, String>("main", Source::Script(r#"$s + "x""#)));
    let page = Arc::new(InMemoryContext::of(program.contexts()));
    let entry = program.entry::<S, String>("main").expect("the entry takes `S`");
    let output = entry
        .run(&page, S { s: "jun".to_owned() })
        .await
        .expect("the graph has no context");
    assert_eq!(output.with(|s: &str| s.to_owned()).expect("a `String`"), "junx");
}

#[tokio::test]
async fn two_inputs_each_reach_the_parameter_of_their_own_name() {
    let i = Interner::new();
    let program = compiled(host(&i).entry::<DeclaredAgainstNameOrder, i64>("main", Source::Script("$a * 10 + $b")));
    assert_eq!(int_result(&program, DeclaredAgainstNameOrder { b: 2, a: 1 }).await, 12);
}

#[tokio::test]
async fn an_input_the_entry_does_not_read_still_crosses() {
    let i = Interner::new();
    let program = compiled(host(&i).entry::<Spare, i64>("main", Source::Script("$used")));
    let spare = Spare {
        used: 7,
        unused: "left alone".to_owned(),
    };
    assert_eq!(int_result(&program, spare).await, 7);
}

#[tokio::test]
async fn a_derived_struct_input_is_read_by_its_fields() {
    let i = Interner::new();
    let program = compiled(host(&i).entry::<HeldProfile, i64>("main", Source::Script("$p.age + 1")));
    let held = HeldProfile {
        p: Profile {
            name: "ann".to_owned(),
            age: 41,
        },
    };
    assert_eq!(int_result(&program, held).await, 42);
}

#[tokio::test]
async fn an_option_input_crosses_present_or_absent() {
    let i = Interner::new();
    let program = compiled(host(&i).entry::<Maybe, i64>(
        "main",
        Source::Script("match $m { Some(x) => x, None => 0 }"),
    ));
    assert_eq!(int_result(&program, Maybe { m: Some(5) }).await, 5);
    assert_eq!(int_result(&program, Maybe { m: None }).await, 0);
}

#[test]
fn an_entry_asked_for_with_other_inputs_than_it_declared_is_refused_before_it_runs() {
    let i = Interner::new();
    let program = compiled(host(&i).entry::<N, i64>("main", Source::Script("$n")));

    for refused in [
        program.entry::<Wrong, i64>("main").map(|_| ()),
        program.entry::<(), i64>("main").map(|_| ()),
    ] {
        assert!(
            matches!(
                refused,
                Err(EntryError::Mismatched {
                    part: EntryPart::Inputs,
                    ..
                })
            ),
            "{refused:?}"
        );
    }
}

#[test]
fn an_input_a_binding_already_fixes_is_refused_at_compile_time() {
    let i = Interner::new();
    let mut bindings = Bindings::default();
    bindings
        .bind(i.intern("n"), BoundValue::Int(1))
        .expect("an integer types on its own");
    let messages = refusal_messages(
        host(&i)
            .bindings(bindings)
            .entry::<N, i64>("main", Source::Script("$n")),
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("`$n`") && message.contains("binding")),
        "{messages:?}"
    );
}

#[test]
fn inputs_declared_by_neither_unit_nor_a_struct_are_refused_at_compile_time() {
    let i = Interner::new();
    let messages = refusal_messages(host(&i).entry::<i64, i64>("main", Source::Script("1")));
    assert!(
        messages.iter().any(|message| message.contains("inputs of the entry `main`")),
        "{messages:?}"
    );
}

#[derive(TyArg)]
pub struct UnreadTracked {
    used: i64,
    tracked: Tracked,
}

#[tokio::test]
async fn an_input_the_entry_does_not_read_is_released_once_by_the_run() {
    const ELEMENTS: usize = 3;
    let counting = Counting::start();
    let i = Interner::new();
    let program = compiled(host(&i).entry::<UnreadTracked, i64>("main", Source::Script("$used")));
    let inputs = UnreadTracked {
        used: 7,
        tracked: Tracked((0..ELEMENTS).map(|_| Counted).collect()),
    };
    assert_eq!(int_result(&program, inputs).await, 7);
    assert_eq!(counting.released(), ELEMENTS);
}

// -- RFC-0090 rule 1: a context's first value is its init --------------

fn init_counted() -> MutexGuard<'static, ()> {
    INIT_COUNTED.lock().unwrap_or_else(PoisonError::into_inner)
}

fn refused(host: Host) -> Vec<Refusal> {
    match host.compile(Arc::new(SequentialExecutor)) {
        Ok(_) => panic!("the program compiled"),
        Err(refusals) => refusals,
    }
}

#[tokio::test]
async fn a_turn_on_an_empty_page_runs_the_expression_init_first() {
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .init("log", Source::Expr("vec([])"))
            .entry::<(), ()>("turn", Source::Script(PUSH_LOG)),
    );
    assert_eq!(solved(&program, &i, "log"), "Vec<String>");
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));

    run_unit(&program, "turn", &page).await;

    let log = log_of(exclusive(&mut page)).expect("`@log` holds a `Vec<String>`");
    assert_eq!(log, ["x"]);
}

#[tokio::test]
async fn a_script_init_edits_what_it_made_and_the_turn_sees_the_edit() {
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .init(
                "h",
                Source::Script(r#"let h = deque(); h.push_back("sys".to_string()); h"#),
            )
            .entry::<(), ()>("turn", Source::Script(r#"@h.push_back("user".to_string());"#))
            .entry::<(), u64>("size", Source::Script("@h.len()"))
            .entry::<(), String>("oldest", Source::Script("@h.pop_front().unwrap()")),
    );
    assert_eq!(solved(&program, &i, "h"), "Deque<String>");
    let page = Arc::new(InMemoryContext::of(program.contexts()));

    run_unit(&program, "turn", &page).await;

    let size = program.entry::<(), u64>("size").expect("`size` returns `u64`");
    let output = size.run(&page, ()).await.expect("the page holds `@h`");
    assert_eq!(output.with(|n: &u64| *n).expect("a `u64`"), 2);
    let oldest = program.entry::<(), String>("oldest").expect("`oldest` returns `String`");
    let output = oldest.run(&page, ()).await.expect("the page holds `@h`");
    assert_eq!(output.with(|s: &str| s.to_owned()).expect("a `String`"), "sys");
}

#[tokio::test]
async fn an_init_runs_once_and_a_second_run_reads_what_the_first_left() {
    let _counted = init_counted();
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .init("log", Source::Script("init_ran(); vec([])"))
            .entry::<(), ()>("turn", Source::Script(PUSH_LOG)),
    );
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    let before = INIT_RUNS.load(Ordering::SeqCst);

    run_unit(&program, "turn", &page).await;
    run_unit(&program, "turn", &page).await;

    assert_eq!(INIT_RUNS.load(Ordering::SeqCst), before + 1);
    let log = log_of(exclusive(&mut page)).expect("`@log` holds a `Vec<String>`");
    assert_eq!(log, ["x", "x"]);
}

#[tokio::test]
async fn a_page_that_holds_the_key_never_runs_its_init() {
    let _counted = init_counted();
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .init("log", Source::Script("init_ran(); vec([])"))
            .entry::<(), ()>("turn", Source::Script(PUSH_LOG)),
    );
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    exclusive(&mut page)
        .insert("log", vec!["held".to_owned()])
        .expect("`@log` is a `Vec<String>`");
    let before = INIT_RUNS.load(Ordering::SeqCst);

    run_unit(&program, "turn", &page).await;
    let filled = program.init_absent(&page).await.expect("the page is the program's");

    assert!(filled.is_empty(), "{filled:?}");
    assert_eq!(INIT_RUNS.load(Ordering::SeqCst), before);
    let log = log_of(exclusive(&mut page)).expect("`@log` holds a `Vec<String>`");
    assert_eq!(log, ["held", "x"]);
}

#[tokio::test]
async fn a_key_with_no_value_and_no_init_refuses_the_run_before_any_init_or_op_runs() {
    let _counted = init_counted();
    let _bumps = BUMPED_RUNS.lock().unwrap_or_else(PoisonError::into_inner);
    let i = Interner::new();
    let program = compiled(
        host(&i)
            .init("n", Source::Script("init_ran(); 1"))
            .entry::<(), ()>("turn", Source::Script(&format!("bump(); @n = @n + 1; {PUSH_LOG}"))),
    );
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    let inits_before = INIT_RUNS.load(Ordering::SeqCst);
    let bumps_before = BUMPS.load(Ordering::SeqCst);

    let turn = program.entry::<(), ()>("turn").expect("the entry returns `()`");
    let refused = turn.run(&page, ()).await.map(|_| ());

    let Err(PageError::Absent { key }) = refused else {
        panic!("a turn that pushes to `@log` ran without `@log` or its init: {refused:?}")
    };
    assert_eq!(key, "log");
    assert_eq!(INIT_RUNS.load(Ordering::SeqCst), inits_before);
    assert_eq!(BUMPS.load(Ordering::SeqCst), bumps_before);
    assert!(matches!(int_of(exclusive(&mut page), "n"), Err(PageError::Absent { .. })));
}

#[test]
fn an_init_that_names_a_context_is_refused_at_compile_naming_its_key() {
    let i = Interner::new();
    let refusals = refused(
        host(&i)
            .init("a", Source::Expr("@x + 1"))
            .entry::<(), i64>("main", Source::Script("@a + @x")),
    );
    let [refusal] = refusals.as_slice() else {
        panic!("one refusal, the init's: {refusals:?}")
    };
    assert_eq!(refusal.origin, Some(Origin::Init("a".to_owned())));
    assert_eq!(
        refusal.message,
        "the init of `@a` names `@x`, and an init names no context"
    );
}

#[test]
fn a_second_init_for_one_key_is_refused() {
    let i = Interner::new();
    let refusals = refused(
        host(&i)
            .init("a", Source::Expr("1"))
            .init("a", Source::Expr("2"))
            .entry::<(), i64>("main", Source::Script("@a")),
    );
    let [refusal] = refusals.as_slice() else {
        panic!("one refusal, the second init's: {refusals:?}")
    };
    assert_eq!(refusal.origin, Some(Origin::Init("a".to_owned())));
    assert_eq!(refusal.message, "`@a` is given two inits");
}

/// A declared type names no source (RFC-0012 rule 7): the source the init
/// makes becomes the context's, and a turn that stores another source is
/// refused, as it is against any declared context.
#[tokio::test]
async fn an_identity_carrying_context_takes_its_init_and_refuses_a_turns_new_source() {
    let i = Interner::new();
    let turn = "@h.record(7); @h.recorded()";
    let program = compiled(
        host(&i)
            .init("h", Source::Expr("history()"))
            .entry::<(), i64>("turn", Source::Script(turn)),
    );
    let page = Arc::new(InMemoryContext::of(program.contexts()));
    let entry = program.entry::<(), i64>("turn").expect("the entry returns `i64`");
    for expected in [1, 2] {
        let output = entry.run(&page, ()).await.expect("the init fills `@h`");
        assert_eq!(output.with(|n: &i64| *n).expect("an `i64`"), expected);
    }

    let refusals = refused(
        host(&i)
            .init("h", Source::Expr("history()"))
            .entry::<(), i64>("turn", Source::Script(turn))
            .entry::<(), ()>("reset", Source::Script("@h = history();")),
    );
    assert!(
        refusals
            .iter()
            .any(|refusal| refusal.origin == Some(Origin::Entry("reset".to_owned()))),
        "a turn's store of a new source into `@h` is refused: {refusals:?}"
    );
    assert!(
        refusals
            .iter()
            .all(|refusal| refusal.origin == Some(Origin::Entry("reset".to_owned()))),
        "only the turn that stores a new source is refused: {refusals:?}"
    );
}

#[tokio::test]
async fn an_empty_space_is_filled_committed_reopened_and_run_without_its_init() {
    let _counted = init_counted();
    let i = Interner::new();
    let space = Arc::new(Space::new(Plain));
    let program = compiled(
        host(&i)
            .init("name", Source::Script(r#"init_ran(); "ann".to_string()"#))
            .entry::<(), String>("name", Source::Script("@name")),
    );
    let before = INIT_RUNS.load(Ordering::SeqCst);

    let page = Arc::new(SpacePage::of(Arc::clone(&space), program.contexts()).expect("empty"));
    let filled = program.init_absent(&page).await.expect("the page is the program's");
    assert_eq!(filled, ["name"]);
    page.commit_opened().expect("the space takes the context it did not hold");

    let reopened =
        Arc::new(SpacePage::of(Arc::clone(&space), program.contexts()).expect("same types"));
    let entry = program.entry::<(), String>("name").expect("the entry returns `String`");
    let output = entry.run(&reopened, ()).await.expect("the space holds `@name`");
    assert_eq!(output.with(|s: &str| s.to_owned()).expect("a `String`"), "ann");
    assert_eq!(INIT_RUNS.load(Ordering::SeqCst), before + 1);
}
