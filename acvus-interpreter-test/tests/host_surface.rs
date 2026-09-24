use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use acvus_extern::{ExternType, Registry, TyArg, Var, extern_fn, extern_registry, kind};
use acvus_interpreter::{
    AcvusRuntime, Host, InMemoryContext, PageError, Program, SequentialExecutor, Source,
};
use acvus_utils::Interner;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct LogState {
    entries: Vec<String>,
}

#[derive(ExternType)]
#[extern_type(name = "Log")]
#[repr(transparent)]
pub struct Log<I>(LogState, PhantomData<I>)
where
    I: Var<kind::Identity>;

impl Log<()> {
    fn empty() -> Self {
        Log(LogState::default(), PhantomData)
    }
}

#[extern_fn(effect = pure)]
fn log_push<I>(log: &mut Log<I>, entry: &str)
where
    I: Var<kind::Identity>,
{
    log.0.entries.push(entry.to_owned());
}

#[extern_fn(effect = pure)]
fn log_len<I>(log: &Log<I>) -> i64
where
    I: Var<kind::Identity>,
{
    i64::try_from(log.0.entries.len()).expect("a test log fits in i64")
}

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

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(extern_registry! {
        ns: "host",
        types: [Log<_>, Tracked],
        fns: [log_push, log_len, profile, tracked],
    });
    registries
}

fn compiled<R>(host: Host, source: &str) -> Program<R>
where
    R: acvus_extern::Declared,
{
    match host.compile::<R>(Source::Script(source), Arc::new(SequentialExecutor)) {
        Ok(program) => program,
        Err(refusals) => panic!("the program is refused: {refusals:?}"),
    }
}

fn exclusive(page: &mut Arc<InMemoryContext>) -> &mut InMemoryContext {
    Arc::get_mut(page).expect("no run holds the page after it returns")
}

#[tokio::test]
async fn a_string_result_is_read_and_edited_through_its_output() {
    let i = Interner::new();
    let program: Program<String> = compiled(Host::new(&i, registries()), r#""hello".to_string()"#);
    let page = Arc::new(InMemoryContext::of(program.contexts()));

    let mut output = program.run(&page).await.expect("the page declares nothing");
    assert_eq!(output.get(), "hello");
    output.get_mut().push_str(", host");
    assert_eq!(output.get(), "hello, host");
}

#[tokio::test]
async fn a_structural_result_is_read_and_edited_through_its_projection() {
    let i = Interner::new();
    let program: Program<Profile> =
        compiled(Host::new(&i, registries()), r#"profile("ann".to_string(), 41)"#);
    let page = Arc::new(InMemoryContext::of(program.contexts()));

    let mut output = program.run(&page).await.expect("the page declares nothing");
    let read = output.get();
    assert_eq!((read.name.as_str(), *read.age), ("ann", 41));
    *output.get_mut().age += 1;
    output.get_mut().name.push('e');
    let read = output.get();
    assert_eq!((read.name.as_str(), *read.age), ("anne", 42));
}

#[tokio::test]
async fn an_opaque_context_is_inserted_updated_by_a_run_and_by_the_host_and_read() {
    let i = Interner::new();
    let program: Program<i64> = compiled(
        Host::new(&i, registries()).context::<Log<()>>("log"),
        r#"log_push(&mut @log, "from the script"); log_len(&@log)"#,
    );
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    exclusive(&mut page)
        .insert("log", Log::<()>::empty())
        .expect("`@log` is declared a `Log`");

    let first = program.run(&page).await.expect("the page declares `@log`");
    assert_eq!(*first.get(), 1);
    let held: &LogState = exclusive(&mut page)
        .read::<Log<()>>("log")
        .expect("`@log` holds a `Log`");
    assert_eq!(held.entries, ["from the script"]);

    exclusive(&mut page)
        .update::<Log<()>, _, _>("log", |log| log.entries.push("from the host".to_owned()))
        .expect("`@log` holds a `Log`");
    let second = program.run(&page).await.expect("the page declares `@log`");
    assert_eq!(*second.get(), 3);
    let held = exclusive(&mut page)
        .read::<Log<()>>("log")
        .expect("`@log` holds a `Log`");
    assert_eq!(
        held.entries,
        ["from the script", "from the host", "from the script"]
    );
}

#[derive(TyArg)]
#[projection]
pub struct Settings {
    title: String,
    limit: i64,
}

#[tokio::test]
async fn a_structural_context_and_a_run_s_write_are_read_through_their_types() {
    let i = Interner::new();
    let program: Program<i64> = compiled(
        Host::new(&i, registries())
            .context::<Settings>("settings")
            .context::<i64>("count"),
        "@count = @settings.limit * 2; @count",
    );
    let mut page = Arc::new(InMemoryContext::of(program.contexts()));
    exclusive(&mut page)
        .insert(
            "settings",
            Settings {
                title: "daily".to_owned(),
                limit: 21,
            },
        )
        .expect("`@settings` is declared a `Settings`");
    exclusive(&mut page)
        .insert("count", 0_i64)
        .expect("`@count` is declared an `i64`");

    let output = program.run(&page).await.expect("the page declares both");
    assert_eq!(*output.get(), 42);
    let settings = exclusive(&mut page)
        .read::<Settings>("settings")
        .expect("`@settings` holds a `Settings`");
    assert_eq!((settings.title.as_str(), *settings.limit), ("daily", 21));
    let count = exclusive(&mut page)
        .read::<i64>("count")
        .expect("`@count` holds an `i64`");
    assert_eq!(*count, 42);
}

#[tokio::test]
async fn a_wrong_type_or_an_absent_key_is_refused_and_leaves_the_page_as_it_was() {
    let i = Interner::new();
    let program: Program<i64> = compiled(
        Host::new(&i, registries())
            .context::<Log<()>>("log")
            .context::<i64>("unset"),
        "log_len(&@log) + @unset",
    );
    let mut page = InMemoryContext::of(program.contexts());
    page.insert("log", Log::<()>::empty())
        .expect("`@log` is declared a `Log`");
    page.update::<Log<()>, _, _>("log", |log| log.entries.push("kept".to_owned()))
        .expect("`@log` holds a `Log`");

    assert!(matches!(
        page.read::<String>("log"),
        Err(PageError::Mismatched { .. })
    ));
    assert!(matches!(
        page.read::<Log<()>>("elsewhere"),
        Err(PageError::Undeclared { .. })
    ));
    assert!(matches!(
        page.read::<i64>("unset"),
        Err(PageError::Absent { .. })
    ));
    assert!(matches!(
        page.update::<i64, _, _>("log", |n| *n += 1),
        Err(PageError::Mismatched { .. })
    ));
    assert!(matches!(
        page.insert("log", "a string".to_owned()),
        Err(PageError::Mismatched { .. })
    ));
    assert!(matches!(
        page.insert("elsewhere", 1_i64),
        Err(PageError::Undeclared { .. })
    ));

    let held = page
        .read::<Log<()>>("log")
        .expect("`@log` still holds its `Log`");
    assert_eq!(held.entries, ["kept"]);
}

#[tokio::test]
async fn a_page_that_holds_a_declared_key_at_another_type_does_not_run() {
    let i = Interner::new();
    let program: Program<i64> = compiled(
        Host::new(&i, registries()).context::<i64>("n"),
        "@n",
    );
    let other: Program<i64> = compiled(
        Host::new(&i, registries()).context::<String>("n"),
        "1",
    );
    let page = Arc::new(InMemoryContext::of(other.contexts()));

    assert!(matches!(
        program.run(&page).await,
        Err(PageError::Mismatched { .. })
    ));
}

#[test]
fn an_entry_that_returns_another_type_is_refused_at_compile_time() {
    let i = Interner::new();
    let refused = Host::new(&i, registries())
        .compile::<String>(Source::Script("1 + 1"), Arc::new(SequentialExecutor));

    let Err(refusals) = refused else {
        panic!("an `i64` tail against a declared `String` compiled")
    };
    assert!(!refusals.is_empty());
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
    let program: Program<Tracked> = compiled(
        Host::new(&i, registries()),
        &format!("tracked({ELEMENTS})"),
    );
    let page = Arc::new(InMemoryContext::of(program.contexts()));

    let output = program.run(&page).await.expect("the page declares nothing");
    assert_eq!(output.get().len(), ELEMENTS);
    assert_eq!(counting.released(), 0, "the output holds every element");
    drop(output);
    assert_eq!(counting.released(), ELEMENTS);
}
