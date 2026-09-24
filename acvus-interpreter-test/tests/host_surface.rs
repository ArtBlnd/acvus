//! RFC-0090 at the host's contract: entries declared by Rust type, contexts
//! whose type the graph solves, and pages read and written through Rust
//! types.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use acvus_extern::{
    Ctx, Declared, Erased, ExternType, NodeHash, Registry, SpaceError, SpaceResult, TyArg, Var,
    extern_fn, extern_registry, kind,
};
use acvus_interpreter::{
    AcvusRuntime, Codec, Head, Held, Host, HostError, MemoryStorage, Named, Origin, Page, Part,
    Plain, Program, Refusal, Scope, SequentialExecutor, Source, Space, SpaceStorage, Storage,
    StorageError, Store,
};

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
    match host.compile(SequentialExecutor) {
        Ok(program) => program,
        Err(error) => panic!("the program is refused: {error:?}"),
    }
}

fn host() -> Host {
    Host::new(registries())
}

fn solved(program: &Program, key: &str) -> String {
    program.context_type(key).expect("the graph has the context")
}

async fn open<'p>(scope: Scope<'p>) -> Page<'p, MemoryStorage> {
    scope.open(MemoryStorage::new()).await.expect("the storage opens")
}

/// `@log`'s strings, each copied in Rust out of the element the page lends.
fn log_of<S>(page: &Page<'_, S>) -> Result<Vec<String>, HostError>
where
    S: Storage,
{
    page.with("log", |ctx: &mut Ctx<'_, Rt>, xs: &[Erased<Rt, String>]| {
        xs.iter().map(|x| x.as_ref(ctx.rt).to_owned()).collect()
    })
}

fn int_of<S>(page: &Page<'_, S>, key: &str) -> Result<i64, HostError>
where
    S: Storage,
{
    page.with(key, |n: &i64| *n)
}

async fn run_unit<'p, S>(scope: Scope<'p>, page: &mut Page<'p, S>, name: &str)
where
    S: Storage,
{
    let entry = scope.entry::<(), ()>(name).expect("the entry returns `()`");
    entry.run(page, ()).await.expect("the page holds the graph's contexts");
}

async fn stored_by(program: &Program, space: &Space, name: &str) {
    program
        .scope(async |s| {
            let mut page = s.open(SpaceStorage::new(space)).await.expect("the space opens");
            run_unit(s, &mut page, name).await;
            page.commit().expect("the space takes what the run stored");
        })
        .await
}

const INIT_LOG: &str = "@log = vec([]);";
const PUSH_LOG: &str = r#"@log.push("x".to_string());"#;
const STORE_LOG: &str = r#"@log = vec([]); @log.push("x".to_string());"#;

fn log_program() -> Program {
    compiled(
        host()
            .init("log", Source::Expr("vec([])"))
            .entry::<(), ()>("init", Source::Script(INIT_LOG))
            .entry::<(), ()>("turn", Source::Script(PUSH_LOG)),
    )
}

fn store_log_program() -> Program {
    compiled(host().entry::<(), ()>("init", Source::Script(STORE_LOG)))
}

#[tokio::test]
async fn a_script_stores_the_context_its_turn_scripts_push_to() {
    let program = log_program();
    assert_eq!(solved(&program, "log"), "Vec<String>");
    program
        .scope(async |s| {
            let mut page = open(s).await;
            run_unit(s, &mut page, "init").await;
            run_unit(s, &mut page, "turn").await;
            run_unit(s, &mut page, "turn").await;
            let log = log_of(&page).expect("`@log` holds a `Vec<String>`");
            assert_eq!(log, ["x", "x"]);
        })
        .await
}

#[test]
fn two_stores_of_one_enum_context_solve_to_the_union_of_their_variants() {
    let program = compiled(
        host()
            .entry::<(), ()>("init", Source::Script("@mode = Mode::Idle;"))
            .entry::<(), ()>("turn", Source::Script("@mode = Mode::Busy;")),
    );
    assert_eq!(solved(&program, "mode"), "Mode{Busy, Idle}");
    let i = Interner::new();
    let declared = <Mode as Declared>::declared(&i).display(&i).to_string();
    assert_eq!(declared, "Mode{Busy, Idle}");
}

#[tokio::test]
async fn an_enum_context_two_entries_store_is_read_through_its_projection() {
    let program = compiled(
        host()
            .entry::<(), ()>("init", Source::Script("@job = Job::Idle;"))
            .entry::<(), ()>("turn", Source::Script("@job = Job::Busy(7);")),
    );
    assert_eq!(solved(&program, "job"), "Job{Busy(i64), Idle}");
    program
        .scope(async |s| {
            let mut page = open(s).await;
            run_unit(s, &mut page, "init").await;
            let idle = page
                .with("job", |job: JobRef<'_>| matches!(job, JobRef::Idle))
                .expect("`@job` holds a `Job`");
            assert!(idle);
            run_unit(s, &mut page, "turn").await;
            let busy = page
                .with("job", |job: JobRef<'_>| match job {
                    JobRef::Busy(n) => Some(*n),
                    JobRef::Idle => None,
                })
                .expect("`@job` holds a `Job`");
            assert_eq!(busy, Some(7));
        })
        .await
}

#[tokio::test]
async fn a_context_the_graph_leaves_open_closes_to_never_and_no_rust_type_reads_it() {
    let program = compiled(host().entry::<(), ()>("init", Source::Script("@xs = [];")));
    assert_eq!(solved(&program, "xs"), "Array<!, 0>");
    program
        .scope(async |s| {
            let mut page = open(s).await;
            run_unit(s, &mut page, "init").await;
            let read = page.with("xs", |_: &[Erased<Rt, String>]| ());
            let Err(HostError::Mismatched {
                what: Part::Context(key),
                held,
                asked,
            }) = read
            else {
                panic!("a `&[String]` borrow of `Array<!, 0>` is refused, and it gave {read:?}")
            };
            assert_eq!(key, "xs");
            assert_eq!(held, "Array<!, 0>");
            assert_eq!(asked, "&[String]");
        })
        .await
}

#[tokio::test]
async fn a_store_only_context_no_run_stored_yet_is_unstored_and_the_page_is_unchanged() {
    let program = store_log_program();
    program
        .scope(async |s| {
            let mut page = open(s).await;
            assert!(matches!(log_of(&page), Err(HostError::Unstored { .. })));
            assert!(matches!(
                page.with_mut("log", |_: &mut [Erased<Rt, String>]| ()),
                Err(HostError::Unstored { .. })
            ));
            assert!(matches!(
                page.with("elsewhere", |_: &[Erased<Rt, String>]| ()),
                Err(HostError::NotInGraph { what: Named::Context(_) })
            ));
            assert!(matches!(log_of(&page), Err(HostError::Unstored { .. })));
        })
        .await
}

#[tokio::test]
async fn a_read_or_an_insert_at_a_wrong_type_is_refused_and_the_page_is_unchanged() {
    let program = log_program();
    program
        .scope(async |s| {
            let mut page = open(s).await;
            run_unit(s, &mut page, "init").await;
            run_unit(s, &mut page, "turn").await;

            assert!(matches!(
                page.with("log", |_: &[Erased<Rt, i64>]| ()),
                Err(HostError::Mismatched { .. })
            ));
            assert!(matches!(page.insert("log", 5_i64), Err(HostError::Mismatched { .. })));
            assert!(matches!(
                page.insert("elsewhere", vec!["y".to_owned()]),
                Err(HostError::NotInGraph { .. })
            ));

            let log = log_of(&page).expect("`@log` still holds its `Vec<String>`");
            assert_eq!(log, ["x"]);
        })
        .await
}

#[tokio::test]
async fn a_host_insert_is_what_the_next_run_reads() {
    let program = log_program();
    program
        .scope(async |s| {
            let mut page = open(s).await;
            page.insert("log", vec!["from the host".to_owned()])
                .expect("`@log` is a `Vec<String>`");
            run_unit(s, &mut page, "turn").await;
            let log = log_of(&page).expect("`@log` holds a `Vec<String>`");
            assert_eq!(log, ["from the host", "x"]);
        })
        .await
}

#[tokio::test]
async fn a_space_that_stores_a_context_at_another_type_opens_no_page() {
    let space = Space::new(Plain);
    let text = compiled(host().entry::<(), ()>("init", Source::Script(r#"@n = "a".to_string();"#)));
    stored_by(&text, &space, "init").await;

    let number = compiled(host().entry::<(), ()>("init", Source::Script("@n = 1;")));
    number
        .scope(async |s| {
            let opened = s.open(SpaceStorage::new(&space)).await.map(|_| ());
            let Err(HostError::Mismatched {
                what: Part::Context(key),
                held,
                asked,
            }) = opened
            else {
                panic!("a space storing `@n` as `String` opened for an `i64` `@n`: {opened:?}")
            };
            assert_eq!(key, "n");
            assert_eq!(held, "String");
            assert_eq!(asked, "i64");
        })
        .await
}

/// A page belongs to one program, so a context held at another type is met
/// where the page opens, before any entry of the program can run.
#[tokio::test]
async fn a_storage_that_holds_a_context_at_another_type_runs_nothing() {
    let space = Space::new(Plain);
    let other = compiled(host().entry::<(), ()>("init", Source::Script(r#"@n = "a".to_string();"#)));
    stored_by(&other, &space, "init").await;

    let program = compiled(
        host()
            .entry::<(), i64>("main", Source::Script("@n"))
            .entry::<(), ()>("init", Source::Script("@n = 1;")),
    );
    program
        .scope(async |s| {
            let opened = s.open(SpaceStorage::new(&space)).await.map(|_| ());
            assert!(matches!(opened, Err(HostError::Mismatched { .. })), "{opened:?}");
        })
        .await
}

#[tokio::test]
async fn a_string_result_is_read_and_edited_through_its_output() {
    let program = compiled(host().entry::<(), String>("main", Source::Script(r#""hello".to_string()"#)));
    program
        .scope(async |s| {
            let mut page = open(s).await;
            let entry = s.entry::<(), String>("main").expect("the entry returns `String`");
            let mut output = entry.run(&mut page, ()).await.expect("the graph has no context");
            let kept: String = output.with(|s: &str| s.to_owned()).expect("the result is a `String`");
            assert_eq!(kept, "hello");
            output
                .with_mut(|s: &mut String| s.push_str(", host"))
                .expect("the result is a `String`");
            assert_eq!(output.with(|s: &String| s.clone()).expect("the result is a `String`"), "hello, host");
            assert_eq!(kept, "hello", "the copy is the host's own");
        })
        .await
}

#[tokio::test]
async fn a_structural_result_is_read_and_edited_through_its_projection() {
    let program = compiled(host().entry::<(), Profile>(
        "main",
        Source::Script(r#"profile("ann".to_string(), 41)"#),
    ));
    program
        .scope(async |s| {
            let mut page = open(s).await;
            let entry = s.entry::<(), Profile>("main").expect("the entry returns `Profile`");
            let mut output = entry.run(&mut page, ()).await.expect("the graph has no context");
            let read = |p: ProfileRef<'_>| (p.name.clone(), *p.age);
            assert_eq!(output.with(read).expect("the result is a `Profile`"), ("ann".to_owned(), 41));
            output
                .with_mut(|p: ProfileMut<'_>| {
                    *p.age += 1;
                    p.name.push('e');
                })
                .expect("the result is a `Profile`");
            assert_eq!(output.with(read).expect("the result is a `Profile`"), ("anne".to_owned(), 42));
        })
        .await
}

fn refused(host: Host) -> Vec<Refusal> {
    match host.compile(SequentialExecutor) {
        Ok(_) => panic!("the program compiled"),
        Err(HostError::Refused(refusals)) => refusals,
        Err(other) => panic!("a compilation is refused with its refusals, not {other:?}"),
    }
}

#[test]
fn an_entry_that_returns_another_type_is_refused_at_compile_time() {
    let refusals = refused(host().entry::<(), String>("main", Source::Script("1 + 1")));
    assert!(
        refusals
            .iter()
            .all(|refusal| refusal.origin == Some(Origin::Entry("main".to_owned()))),
        "{refusals:?}"
    );
    assert!(!refusals.is_empty());
}

#[tokio::test]
async fn an_entry_asked_for_at_another_type_than_it_declared_is_refused_before_it_runs() {
    let program = compiled(host().entry::<(), String>("main", Source::Script(r#""a".to_string()"#)));
    program
        .scope(async |s| {
            assert!(matches!(
                s.entry::<(), i64>("main").map(|_| ()),
                Err(HostError::Mismatched { what: Part::Result(_), .. })
            ));
            assert!(matches!(
                s.entry::<(), String>("elsewhere").map(|_| ()),
                Err(HostError::NotInGraph { what: Named::Entry(_) })
            ));
        })
        .await
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
    let program = compiled(host().entry::<(), Tracked>(
        "main",
        Source::Script(&format!("tracked({ELEMENTS})")),
    ));
    program
        .scope(async |s| {
            let mut page = open(s).await;
            let entry = s.entry::<(), Tracked>("main").expect("the entry returns `Tracked`");
            let output = entry.run(&mut page, ()).await.expect("the graph has no context");
            assert_eq!(output.with(|t: &Tracked| t.0.len()).expect("the result is a `Tracked`"), ELEMENTS);
            assert_eq!(counting.released(), 0, "the output holds every element");
            drop(output);
            assert_eq!(counting.released(), ELEMENTS);
        })
        .await
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

const PUSH_DEQUE: &str = r#"@log.push_back("x".to_string());"#;
const STORE_DEQUE: &str = r#"@log = deque(); @log.push_back("x".to_string());"#;

/// `@log` is a `Deque<String>` here, which a space holds, so one program
/// stores it and another opens over it.
fn fresh_log_program() -> Program {
    compiled(
        host()
            .entry::<(), ()>("init", Source::Script("@log = deque();"))
            .entry::<(), ()>("turn", Source::Script(&format!("bump(); {PUSH_DEQUE}")))
            .entry::<(), u64>("size", Source::Script("@log.len()")),
    )
}

fn store_deque_program() -> Program {
    compiled(host().entry::<(), ()>("init", Source::Script(STORE_DEQUE)))
}

async fn size_of<'p, S>(scope: Scope<'p>, page: &mut Page<'p, S>) -> u64
where
    S: Storage,
{
    let size = scope.entry::<(), u64>("size").expect("`size` returns `u64`");
    let output = size.run(page, ()).await.expect("the page holds `@log`");
    output.with(|n: &u64| *n).expect("a `u64`")
}

fn unfilled_key(opened: Result<(), HostError>) -> String {
    let Err(HostError::Unfilled { key }) = opened else {
        panic!("the page opened over a storage without a context an entry fetches first: {opened:?}")
    };
    key
}

async fn open_refusal(program: &Program) -> String {
    program
        .scope(async |s| unfilled_key(s.open(MemoryStorage::new()).await.map(|_| ())))
        .await
}

#[tokio::test]
async fn a_storing_script_runs_on_a_storage_that_holds_nothing_and_its_turns_push_to_it() {
    let _bumps = BUMPED_RUNS.lock().unwrap_or_else(PoisonError::into_inner);
    let space = Space::new(Plain);
    stored_by(&store_deque_program(), &space, "init").await;

    let program = fresh_log_program();
    assert_eq!(solved(&program, "log"), "Deque<String>");
    program
        .scope(async |s| {
            let mut page = s.open(SpaceStorage::new(&space)).await.expect("the space holds `@log`");
            run_unit(s, &mut page, "init").await;
            run_unit(s, &mut page, "turn").await;
            run_unit(s, &mut page, "turn").await;
            assert_eq!(size_of(s, &mut page).await, 2);
        })
        .await
}

#[tokio::test]
async fn a_turn_before_the_script_that_stores_its_context_is_refused_before_any_of_it_runs() {
    let _bumps = BUMPED_RUNS.lock().unwrap_or_else(PoisonError::into_inner);
    let space = Space::new(Plain);
    let program = fresh_log_program();
    let bumps_before = BUMPS.load(Ordering::SeqCst);

    assert_eq!(open_refusal(&program).await, "log");
    assert_eq!(BUMPS.load(Ordering::SeqCst), bumps_before, "the turn's first call did not run");

    stored_by(&store_deque_program(), &space, "init").await;
    stored_by(&program, &space, "turn").await;
    assert_eq!(BUMPS.load(Ordering::SeqCst), bumps_before + 1);
}

/// The fetch is the callee's and comes after `bump()`: the refusal is the
/// page's, before any run.
#[tokio::test]
async fn a_program_whose_callee_fetches_a_context_the_storage_lacks_opens_no_page() {
    let _bumps = BUMPED_RUNS.lock().unwrap_or_else(PoisonError::into_inner);
    let space = Space::new(Plain);
    let program = compiled(
        host()
            .entry::<(), ()>("init", Source::Script("@log = deque();"))
            .entry::<(), ()>("log_x", Source::Script(PUSH_DEQUE))
            .entry::<(), ()>("turn", Source::Script("bump(); log_x();")),
    );
    let bumps_before = BUMPS.load(Ordering::SeqCst);

    assert_eq!(open_refusal(&program).await, "log");
    assert_eq!(BUMPS.load(Ordering::SeqCst), bumps_before);

    stored_by(&store_deque_program(), &space, "init").await;
    stored_by(&program, &space, "turn").await;
    assert_eq!(BUMPS.load(Ordering::SeqCst), bumps_before + 1);
}

#[tokio::test]
async fn a_context_assigned_on_one_branch_only_is_fetched_and_refused_on_an_empty_storage() {
    let bare = compiled(host().entry::<(), i64>("main", Source::Script("if coin() { @x = 1; } @x")));
    assert_eq!(open_refusal(&bare).await, "x");

    let program = compiled(
        host()
            .init("x", Source::Expr("5"))
            .entry::<(), i64>("main", Source::Script("if coin() { @x = 1; } @x")),
    );
    program
        .scope(async |s| {
            let mut page = open(s).await;
            let main = s.entry::<(), i64>("main").expect("the entry returns `i64`");
            let output = main.run(&mut page, ()).await.expect("the page holds `@x`");
            assert_eq!(output.with(|n: &i64| *n).expect("the result is an `i64`"), 1);
        })
        .await
}

/// A callee's store is in its summary as "may write"; it does not assign the
/// caller's variable, so the caller that reads `@x` after the call fetches
/// `@x` first.
#[tokio::test]
async fn a_whole_assignment_inside_a_callee_does_not_spare_the_callers_fetch() {
    let bare = compiled(
        host()
            .entry::<(), ()>("set_x", Source::Script("@x = 1;"))
            .entry::<(), i64>("main", Source::Script("set_x(); @x")),
    );
    assert_eq!(open_refusal(&bare).await, "x");

    let program = compiled(
        host()
            .init("x", Source::Expr("7"))
            .entry::<(), ()>("set_x", Source::Script("@x = 1;"))
            .entry::<(), i64>("main", Source::Script("set_x(); @x")),
    );
    program
        .scope(async |s| {
            let mut page = open(s).await;
            let main = s.entry::<(), i64>("main").expect("the entry returns `i64`");
            let output = main.run(&mut page, ()).await.expect("the page holds `@x`");
            assert_eq!(
                output.with(|n: &i64| *n).expect("the result is an `i64`"),
                1,
                "the caller reads what the callee stored"
            );
        })
        .await
}

#[tokio::test]
async fn a_call_that_writes_a_context_before_the_body_assigns_it_makes_the_body_fetch_it() {
    let bare = compiled(
        host()
            .entry::<(), ()>("set_x", Source::Script("@x = 1;"))
            .entry::<(), i64>("main", Source::Script("set_x(); @x = 2; @x")),
    );
    assert_eq!(open_refusal(&bare).await, "x", "the body skipped the fetch the call's bracket needs");

    let program = compiled(
        host()
            .init("x", Source::Expr("7"))
            .entry::<(), ()>("set_x", Source::Script("@x = 1;"))
            .entry::<(), i64>("main", Source::Script("set_x(); @x = 2; @x")),
    );
    program
        .scope(async |s| {
            let mut page = open(s).await;
            let main = s.entry::<(), i64>("main").expect("the entry returns `i64`");
            let output = main.run(&mut page, ()).await.expect("the page holds `@x`");
            assert_eq!(
                output.with(|n: &i64| *n).expect("the result is an `i64`"),
                2,
                "the body's own store is the last"
            );
            drop(output);
            assert_eq!(int_of(&page, "x").expect("`@x` is held"), 2);
        })
        .await
}

/// The init writes 0, so a callee that read `@x` before its caller committed
/// would return 0.
#[tokio::test]
async fn a_callee_reads_what_its_caller_assigned_before_the_call() {
    let program = compiled(
        host()
            .init("x", Source::Expr("0"))
            .entry::<(), i64>("read_x", Source::Script("@x"))
            .entry::<(), i64>("main", Source::Script("@x = 3; read_x()")),
    );
    program
        .scope(async |s| {
            let mut page = open(s).await;
            let main = s.entry::<(), i64>("main").expect("the entry returns `i64`");
            let output = main
                .run(&mut page, ())
                .await
                .expect("the caller commits `@x` before the callee fetches it");
            assert_eq!(output.with(|n: &i64| *n).expect("the result is an `i64`"), 3);
        })
        .await;

    let bare = compiled(
        host()
            .entry::<(), i64>("read_x", Source::Script("@x"))
            .entry::<(), i64>("main", Source::Script("@x = 3; read_x()")),
    );
    assert_eq!(open_refusal(&bare).await, "x");
}

#[tokio::test]
async fn an_insert_of_a_key_no_script_names_is_not_in_the_graph() {
    let program = store_log_program();
    program
        .scope(async |s| {
            let mut page = open(s).await;
            let refused = page.insert("transcript", vec!["kept by the host".to_owned()]);
            let Err(HostError::NotInGraph {
                what: Named::Context(key),
            }) = refused
            else {
                panic!("a key no script names was inserted: {refused:?}")
            };
            assert_eq!(key, "transcript");
        })
        .await
}

/// Without a fetch, the value the page held is replaced at the exit's
/// commit, and released there once.
#[tokio::test]
async fn a_store_only_run_over_a_held_value_releases_the_value_it_replaces_once() {
    const ELEMENTS: usize = 3;
    let counting = Counting::start();
    let program = compiled(host().entry::<(), ()>(
        "init",
        Source::Script(&format!("@t = tracked({ELEMENTS});")),
    ));
    program
        .scope(async |s| {
            let mut page = open(s).await;
            run_unit(s, &mut page, "init").await;
            assert_eq!(counting.released(), 0, "the page holds every element");
            run_unit(s, &mut page, "init").await;
            assert_eq!(counting.released(), ELEMENTS, "the replaced value is released once");
            drop(page);
            assert_eq!(counting.released(), 2 * ELEMENTS);
        })
        .await
}

// -- RFC-0090 rules 3-5: a host is lent a value as a handler is ------------

#[tokio::test]
async fn an_element_edited_in_place_is_what_the_next_run_reads() {
    let program = log_program();
    program
        .scope(async |s| {
            let mut page = open(s).await;
            run_unit(s, &mut page, "init").await;
            run_unit(s, &mut page, "turn").await;
            page.with_mut(
                "log",
                |ctx: &mut Ctx<'_, Rt>, xs: &mut [Erased<Rt, String>]| xs[0].as_mut(ctx.rt).push('!'),
            )
            .expect("`@log` holds a `Vec<String>`");
            run_unit(s, &mut page, "turn").await;
            let log = log_of(&page).expect("`@log` holds a `Vec<String>`");
            assert_eq!(log, ["x!", "x"]);
        })
        .await
}

#[tokio::test]
async fn a_derived_context_is_edited_through_its_projection_and_the_next_run_reads_the_edit() {
    let program = compiled(
        host()
            .init("p", Source::Expr(r#"profile("bob".to_string(), 1)"#))
            .entry::<(), ()>("init", Source::Script(r#"@p = profile("ann".to_string(), 41);"#))
            .entry::<(), i64>("age", Source::Script("@p.age")),
    );
    program
        .scope(async |s| {
            let mut page = open(s).await;
            run_unit(s, &mut page, "init").await;
            page.with_mut("p", |p: ProfileMut<'_>| {
                *p.age += 1;
                p.name.push('e');
            })
            .expect("`@p` holds a `Profile`");
            let name = page
                .with("p", |p: ProfileRef<'_>| p.name.clone())
                .expect("`@p` holds a `Profile`");
            assert_eq!(name, "anne");

            let age = s.entry::<(), i64>("age").expect("the entry returns `i64`");
            let output = age.run(&mut page, ()).await.expect("the page holds `@p`");
            assert_eq!(output.with(|n: &i64| *n).expect("the result is an `i64`"), 42);
        })
        .await
}

#[tokio::test]
async fn a_closure_of_another_type_is_refused_before_it_runs_and_the_page_is_unchanged() {
    let program = log_program();
    program
        .scope(async |s| {
            let mut page = open(s).await;
            run_unit(s, &mut page, "init").await;
            run_unit(s, &mut page, "turn").await;

            let mut ran = false;
            let refused = page.with_mut("log", |_: &mut String| ran = true);
            let Err(HostError::Mismatched {
                what: Part::Context(key),
                held,
                asked,
            }) = refused
            else {
                panic!("a `&mut String` borrow of `Vec<String>` ran: {refused:?}")
            };
            assert!(!ran, "the closure did not run");
            assert_eq!(key, "log");
            assert_eq!(held, "Vec<String>");
            assert_eq!(asked, "&mut String");
            assert_eq!(log_of(&page).expect("`@log` is held"), ["x"]);

            let entry = s.entry::<(), ()>("turn").expect("the entry returns `()`");
            let output = entry.run(&mut page, ()).await.expect("the page holds `@log`");
            let mut ran = false;
            let refused = output.with(|_: &i64| ran = true);
            assert!(
                matches!(refused, Err(HostError::Mismatched { what: Part::Result(_), .. })),
                "{refused:?}"
            );
            assert!(!ran, "the closure did not run");
        })
        .await
}

#[tokio::test]
async fn a_string_context_is_kept_as_a_copy_the_host_makes() {
    let program = compiled(host().entry::<(), ()>("init", Source::Script(r#"@name = "ann".to_string();"#)));
    program
        .scope(async |s| {
            let mut page = open(s).await;
            run_unit(s, &mut page, "init").await;
            let kept: String = page.with("name", |s: &str| s.to_owned()).expect("`@name` is a `String`");
            page.with_mut("name", |s: &mut String| s.push_str(" smith"))
                .expect("`@name` is a `String`");
            assert_eq!(kept, "ann");
            assert_eq!(
                page.with("name", |s: &str| s.to_owned()).expect("`@name` is held"),
                "ann smith"
            );
        })
        .await
}

#[tokio::test]
async fn a_space_page_lends_what_it_loads_and_commits_what_it_was_lent() {
    let space = Space::new(Plain);
    let storing = compiled(host().entry::<(), ()>("init", Source::Script(r#"@name = "ann".to_string();"#)));
    stored_by(&storing, &space, "init").await;

    let program = compiled(
        host()
            .entry::<(), ()>("init", Source::Script(r#"@name = "ann".to_string();"#))
            .entry::<(), String>("name", Source::Script("@name")),
    );
    program
        .scope(async |s| {
            let mut reopened = s.open(SpaceStorage::new(&space)).await.expect("same types");
            reopened
                .with_mut("name", |s: &mut String| s.push('?'))
                .expect("`@name` loads from the space");
            assert_eq!(reopened.with("name", |s: &str| s.to_owned()).expect("`@name` is held"), "ann?");
            reopened.commit().expect("the space takes the edit");
        })
        .await;
    program
        .scope(async |s| {
            let mut fresh = s.open(SpaceStorage::new(&space)).await.expect("same types");
            let entry = s.entry::<(), String>("name").expect("the entry returns `String`");
            let output = entry.run(&mut fresh, ()).await.expect("the space holds `@name`");
            assert_eq!(output.with(|s: &str| s.to_owned()).expect("a `String`"), "ann?");
        })
        .await
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
    refused(host).into_iter().map(|refusal| refusal.message).collect()
}

async fn int_result<I>(program: &Program, inputs: I) -> i64
where
    I: acvus_extern::Declared + acvus_extern::Cross<Rt, ReturnForm: acvus_extern::Returned<Verdict = ()>>,
{
    program
        .scope(async |s| {
            let mut page = open(s).await;
            let entry = s.entry::<I, i64>("main").expect("the entry takes `I` and returns `i64`");
            let output = entry.run(&mut page, inputs).await.expect("the graph has no context");
            output.with(|n: &i64| *n).expect("the result is an `i64`")
        })
        .await
}

#[test]
fn a_dollar_read_by_an_entry_declaring_no_inputs_is_refused_at_compile_time() {
    let messages = refusal_messages(host().entry::<(), i64>("main", Source::Script("$n + 1")));
    assert!(
        messages.iter().any(|message| message.contains("`$n`")),
        "{messages:?}"
    );
}

#[tokio::test]
async fn an_input_crosses_into_the_parameter_its_field_names() {
    let program = compiled(host().entry::<N, i64>("main", Source::Script("$n + 1")));
    assert_eq!(int_result(&program, N { n: 41 }).await, 42);
}

#[tokio::test]
async fn a_text_input_crosses_as_the_string_it_is() {
    let program = compiled(host().entry::<S, String>("main", Source::Script(r#"$s + "x""#)));
    program
        .scope(async |s| {
            let mut page = open(s).await;
            let entry = s.entry::<S, String>("main").expect("the entry takes `S`");
            let output = entry
                .run(&mut page, S { s: "jun".to_owned() })
                .await
                .expect("the graph has no context");
            assert_eq!(output.with(|s: &str| s.to_owned()).expect("a `String`"), "junx");
        })
        .await
}

#[tokio::test]
async fn two_inputs_each_reach_the_parameter_of_their_own_name() {
    let program = compiled(host().entry::<DeclaredAgainstNameOrder, i64>("main", Source::Script("$a * 10 + $b")));
    assert_eq!(int_result(&program, DeclaredAgainstNameOrder { b: 2, a: 1 }).await, 12);
}

#[tokio::test]
async fn an_input_the_entry_does_not_read_still_crosses() {
    let program = compiled(host().entry::<Spare, i64>("main", Source::Script("$used")));
    let spare = Spare {
        used: 7,
        unused: "left alone".to_owned(),
    };
    assert_eq!(int_result(&program, spare).await, 7);
}

#[tokio::test]
async fn a_derived_struct_input_is_read_by_its_fields() {
    let program = compiled(host().entry::<HeldProfile, i64>("main", Source::Script("$p.age + 1")));
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
    let program = compiled(host().entry::<Maybe, i64>(
        "main",
        Source::Script("match $m { Some(x) => x, None => 0 }"),
    ));
    assert_eq!(int_result(&program, Maybe { m: Some(5) }).await, 5);
    assert_eq!(int_result(&program, Maybe { m: None }).await, 0);
}

#[tokio::test]
async fn an_entry_asked_for_with_other_inputs_than_it_declared_is_refused_before_it_runs() {
    let program = compiled(host().entry::<N, i64>("main", Source::Script("$n")));
    program
        .scope(async |s| {
            for refused in [
                s.entry::<Wrong, i64>("main").map(|_| ()),
                s.entry::<(), i64>("main").map(|_| ()),
            ] {
                assert!(
                    matches!(refused, Err(HostError::Mismatched { what: Part::Inputs(_), .. })),
                    "{refused:?}"
                );
            }
        })
        .await
}

#[test]
fn an_input_a_binding_already_fixes_is_refused_at_compile_time() {
    let bound = host().bind("n", "1").expect("an integer is a literal");
    let messages = refusal_messages(bound.entry::<N, i64>("main", Source::Script("$n")));
    assert!(
        messages
            .iter()
            .any(|message| message.contains("`$n`") && message.contains("binding")),
        "{messages:?}"
    );
}

#[test]
fn inputs_declared_by_neither_unit_nor_a_struct_are_refused_at_compile_time() {
    let messages = refusal_messages(host().entry::<i64, i64>("main", Source::Script("1")));
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
    let program = compiled(host().entry::<UnreadTracked, i64>("main", Source::Script("$used")));
    let inputs = UnreadTracked {
        used: 7,
        tracked: Tracked((0..ELEMENTS).map(|_| Counted).collect()),
    };
    assert_eq!(int_result(&program, inputs).await, 7);
    assert_eq!(counting.released(), ELEMENTS);
}

// -- A binding folds as the CLI's `name=<literal>` does (RFC-0087) -------

const BY_VARIANT: &str = "\
% match $mode
% Mode::Review =>
Review.
% Mode::Explain =>
Explain.
% end
";

#[tokio::test]
async fn a_bound_variant_folds_to_the_arm_it_chose() {
    for (literal, expected) in [("Mode::Review", "Review.\n"), ("Mode::Explain", "Explain.\n")] {
        let program = compiled(
            host()
                .bind("mode", literal)
                .expect("a variant is a literal")
                .entry::<(), String>("main", Source::Template(BY_VARIANT)),
        );
        let text = program
            .scope(async |s| {
                let mut page = open(s).await;
                let entry = s.entry::<(), String>("main").expect("the entry returns `String`");
                let output = entry.run(&mut page, ()).await.expect("the graph has no context");
                output.with(|s: &str| s.to_owned()).expect("a `String`")
            })
            .await;
        assert_eq!(text, expected);
    }
}

#[test]
fn a_binding_outside_the_literal_grammar_is_refused_naming_it() {
    let Err(HostError::Refused(refusals)) = host().bind("x", "f()").map(|_| ()) else {
        panic!("a call was bound as a literal")
    };
    let [refusal] = refusals.as_slice() else {
        panic!("one refusal, the binding's: {refusals:?}")
    };
    assert_eq!(refusal.origin, Some(Origin::Binding("x".to_owned())));
    assert_eq!(refusal.message, "$x: `f()` is not a value a literal writes");
}

// -- RFC-0090 rule 1: a context's first value is its init --------------

fn init_counted() -> MutexGuard<'static, ()> {
    INIT_COUNTED.lock().unwrap_or_else(PoisonError::into_inner)
}

#[tokio::test]
async fn opening_an_empty_storage_runs_the_expression_init_and_a_turn_reads_it() {
    let program = compiled(
        host()
            .init("log", Source::Expr("vec([])"))
            .entry::<(), ()>("turn", Source::Script(PUSH_LOG)),
    );
    assert_eq!(solved(&program, "log"), "Vec<String>");
    program
        .scope(async |s| {
            let mut page = open(s).await;
            assert_eq!(page.filled(), ["log"]);
            assert_eq!(log_of(&page).expect("the init filled `@log`"), Vec::<String>::new());
            run_unit(s, &mut page, "turn").await;
            let log = log_of(&page).expect("`@log` holds a `Vec<String>`");
            assert_eq!(log, ["x"]);
        })
        .await
}

#[tokio::test]
async fn a_script_init_edits_what_it_made_and_the_turn_sees_the_edit() {
    let program = compiled(
        host()
            .init(
                "h",
                Source::Script(r#"let h = deque(); h.push_back("sys".to_string()); h"#),
            )
            .entry::<(), ()>("turn", Source::Script(r#"@h.push_back("user".to_string());"#))
            .entry::<(), u64>("size", Source::Script("@h.len()"))
            .entry::<(), String>("oldest", Source::Script("@h.pop_front().unwrap()")),
    );
    assert_eq!(solved(&program, "h"), "Deque<String>");
    program
        .scope(async |s| {
            let mut page = open(s).await;
            run_unit(s, &mut page, "turn").await;
            let size = s.entry::<(), u64>("size").expect("`size` returns `u64`");
            let output = size.run(&mut page, ()).await.expect("the page holds `@h`");
            assert_eq!(output.with(|n: &u64| *n).expect("a `u64`"), 2);
            let oldest = s.entry::<(), String>("oldest").expect("`oldest` returns `String`");
            let output = oldest.run(&mut page, ()).await.expect("the page holds `@h`");
            assert_eq!(output.with(|s: &str| s.to_owned()).expect("a `String`"), "sys");
        })
        .await
}

#[tokio::test]
async fn an_init_runs_once_and_a_second_run_reads_what_the_first_left() {
    let _counted = init_counted();
    let program = compiled(
        host()
            .init("log", Source::Script("init_ran(); vec([])"))
            .entry::<(), ()>("turn", Source::Script(PUSH_LOG)),
    );
    let before = INIT_RUNS.load(Ordering::SeqCst);
    program
        .scope(async |s| {
            let mut page = open(s).await;
            run_unit(s, &mut page, "turn").await;
            run_unit(s, &mut page, "turn").await;
            let log = log_of(&page).expect("`@log` holds a `Vec<String>`");
            assert_eq!(log, ["x", "x"]);
        })
        .await;
    assert_eq!(INIT_RUNS.load(Ordering::SeqCst), before + 1);
}

#[tokio::test]
async fn a_storage_that_holds_the_key_never_runs_its_init() {
    let _counted = init_counted();
    let space = Space::new(Plain);
    let seeding = compiled(
        host()
            .init("log", Source::Script("init_ran(); deque()"))
            .entry::<(), ()>("seed", Source::Script(STORE_DEQUE)),
    );
    let program = compiled(
        host()
            .init("log", Source::Script("init_ran(); deque()"))
            .entry::<(), ()>("turn", Source::Script(PUSH_DEQUE))
            .entry::<(), u64>("size", Source::Script("@log.len()")),
    );
    let before = INIT_RUNS.load(Ordering::SeqCst);

    stored_by(&seeding, &space, "seed").await;
    program
        .scope(async |s| {
            let mut page = s.open(SpaceStorage::new(&space)).await.expect("the space holds `@log`");
            assert!(page.filled().is_empty(), "{:?}", page.filled());
            run_unit(s, &mut page, "turn").await;
            assert_eq!(size_of(s, &mut page).await, 2, "the held element and the turn's");
        })
        .await;
    assert_eq!(INIT_RUNS.load(Ordering::SeqCst), before);
}

#[tokio::test]
async fn a_key_with_no_value_and_no_init_refuses_the_open_before_any_init_or_op_runs() {
    let _counted = init_counted();
    let _bumps = BUMPED_RUNS.lock().unwrap_or_else(PoisonError::into_inner);
    let program = compiled(
        host()
            .init("n", Source::Script("init_ran(); 1"))
            .entry::<(), ()>("turn", Source::Script(&format!("bump(); @n = @n + 1; {PUSH_LOG}"))),
    );
    let inits_before = INIT_RUNS.load(Ordering::SeqCst);
    let bumps_before = BUMPS.load(Ordering::SeqCst);

    let key = program
        .scope(async |s| unfilled_key(s.open(MemoryStorage::new()).await.map(|_| ())))
        .await;

    assert_eq!(key, "log");
    assert_eq!(INIT_RUNS.load(Ordering::SeqCst), inits_before);
    assert_eq!(BUMPS.load(Ordering::SeqCst), bumps_before);
}

#[test]
fn an_init_that_names_a_context_is_refused_at_compile_naming_its_key() {
    let refusals = refused(
        host()
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
    let refusals = refused(
        host()
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
    let turn = "@h.record(7); @h.recorded()";
    let program = compiled(
        host()
            .init("h", Source::Expr("history()"))
            .entry::<(), i64>("turn", Source::Script(turn)),
    );
    program
        .scope(async |s| {
            let mut page = open(s).await;
            let entry = s.entry::<(), i64>("turn").expect("the entry returns `i64`");
            for expected in [1, 2] {
                let output = entry.run(&mut page, ()).await.expect("the init fills `@h`");
                assert_eq!(output.with(|n: &i64| *n).expect("an `i64`"), expected);
            }
        })
        .await;

    let refusals = refused(
        host()
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
    let space = Space::new(Plain);
    let program = compiled(
        host()
            .init("name", Source::Script(r#"init_ran(); "ann".to_string()"#))
            .entry::<(), String>("name", Source::Script("@name")),
    );
    let before = INIT_RUNS.load(Ordering::SeqCst);

    program
        .scope(async |s| {
            let mut page = s.open(SpaceStorage::new(&space)).await.expect("the space opens");
            assert_eq!(page.filled(), ["name"]);
            page.commit().expect("the space takes the context it did not hold");
        })
        .await;
    program
        .scope(async |s| {
            let mut reopened = s.open(SpaceStorage::new(&space)).await.expect("same types");
            assert!(reopened.filled().is_empty());
            let entry = s.entry::<(), String>("name").expect("the entry returns `String`");
            let output = entry.run(&mut reopened, ()).await.expect("the space holds `@name`");
            assert_eq!(output.with(|s: &str| s.to_owned()).expect("a `String`"), "ann");
        })
        .await;
    assert_eq!(INIT_RUNS.load(Ordering::SeqCst), before + 1);
}

// -- A storage's errors reach the host where the page opens (RFC-0090 rule 6) --

struct Unreadable;

impl Storage for Unreadable {
    fn load(&mut self, key: &str, _: &Codec<'_>) -> Result<Option<Held>, StorageError> {
        Err(StorageError::new(format!("`@{key}` is on a disk that is gone")))
    }

    fn store(&mut self, _: &str, _: Held) {}

    fn commit(&mut self, _: &Codec<'_>) -> Result<(), StorageError> {
        Ok(())
    }
}

struct HeadlessStore;

impl Store for HeadlessStore {
    fn put(&self, _: NodeHash, _: &[u8]) -> SpaceResult<()> {
        Ok(())
    }

    fn get(&self, _: NodeHash) -> SpaceResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn head(&self, id: &str) -> SpaceResult<Option<Head>> {
        Err(SpaceError::new(format!("the head of `@{id}` does not read")))
    }

    fn identities(&self) -> SpaceResult<Vec<String>> {
        Ok(Vec::new())
    }

    fn node_count(&self) -> SpaceResult<usize> {
        Ok(0)
    }

    fn cmpxchg(&self, _: &str, _: Option<NodeHash>, _: Head) -> SpaceResult<Result<(), Option<NodeHash>>> {
        Ok(Ok(()))
    }
}

#[tokio::test]
async fn a_storage_that_fails_to_load_refuses_the_open() {
    let program = compiled(
        host()
            .init("n", Source::Expr("1"))
            .entry::<(), i64>("main", Source::Script("@n")),
    );
    let opened = program
        .scope(async |s| s.open(Unreadable).await.map(|_| ()))
        .await;
    assert!(matches!(opened, Err(HostError::Storage(_))), "{opened:?}");

    let space = Space::over(Plain, Box::new(HeadlessStore));
    let opened = program
        .scope(async |s| s.open(SpaceStorage::new(&space)).await.map(|_| ()))
        .await;
    let Err(HostError::Storage(error)) = opened else {
        panic!("a space whose head does not read opened a page: {opened:?}")
    };
    assert_eq!(error.to_string(), "the head of `@n` does not read");
}

/// Keeps every holder it is handed and never gives one back to the page
/// that handed it, so a later page is offered it.
#[derive(Default)]
struct Hoard {
    held: std::collections::HashMap<String, Held>,
    giving: bool,
}

impl Storage for &mut Hoard {
    fn load(&mut self, key: &str, _: &Codec<'_>) -> Result<Option<Held>, StorageError> {
        match self.giving {
            true => Ok(self.held.remove(key)),
            false => Ok(None),
        }
    }

    fn store(&mut self, key: &str, held: Held) {
        self.held.insert(key.to_owned(), held);
    }

    fn commit(&mut self, _: &Codec<'_>) -> Result<(), StorageError> {
        Ok(())
    }
}

#[tokio::test]
async fn a_holder_another_compilation_made_is_refused_where_the_page_opens() {
    let source = r#"@name = "ann".to_string();"#;
    let first = compiled(host().entry::<(), ()>("init", Source::Script(source)));
    let second = compiled(host().entry::<(), ()>("init", Source::Script(source)));
    let mut hoard = Hoard::default();

    let committed = first
        .scope(async |s| {
            let mut page = s.open(&mut hoard).await.expect("the hoard gives nothing");
            run_unit(s, &mut page, "init").await;
            page.commit().map(|_| ())
        })
        .await;
    assert!(matches!(committed, Err(HostError::Storage(_))), "{committed:?}");
    assert!(hoard.held.contains_key("name"), "the hoard kept the holder");

    hoard.giving = true;
    let opened = second.scope(async |s| s.open(&mut hoard).await.map(|_| ())).await;
    let Err(HostError::Storage(error)) = opened else {
        panic!("a page took a holder another compilation made: {opened:?}")
    };
    assert_eq!(error.to_string(), "the storage gave `@name` a holder another compilation made");
}
