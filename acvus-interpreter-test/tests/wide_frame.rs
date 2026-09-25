//! A body wider than one mark word runs: every register of its frame has a
//! mark bit, an operation that takes registers above word 0 drops their
//! claims, and the sweep releases what the frame still owns exactly once
//! (RFC-0048 rules 3 to 6).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use acvus_extern::{ExternType, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, HostError, Value};
use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_interpreter_test::listing::{
    main_body, ops_of_anywhere, prepared_script_with_externs, script_listing_with_externs,
};
use acvus_interpreter_test::{Context, check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

const WIDTHS: [usize; 4] = [64, 65, 128, 200];

fn value(source: &str) -> String {
    let full = corpus::attempt(source, Opt::Full, Stage::Run);
    let none = corpus::attempt(source, Opt::None, Stage::Run);
    assert_eq!(full, none, "the two levels disagree");
    match full {
        Outcome::Value(v) => v,
        other => panic!("the source produced no value: {other:?}"),
    }
}

fn int_literal(n: usize) -> String {
    let elements: Vec<String> = (1..=n).map(|i| i.to_string()).collect();
    format!("let v = vec([{}]); v.len()", elements.join(", "))
}

fn string_literal(n: usize) -> String {
    let elements: Vec<String> = (0..n).map(|i| format!("\"s{i}\".to_string()")).collect();
    format!("let v = [{}]; v.len()", elements.join(", "))
}

#[test]
fn a_vec_literal_as_wide_as_the_frame_runs() {
    for n in WIDTHS {
        assert_eq!(value(&int_literal(n)), n.to_string(), "{n} elements");
    }
}

#[test]
fn owned_strings_across_mark_words_are_taken_once() {
    for n in WIDTHS {
        assert_eq!(value(&string_literal(n)), n.to_string(), "{n} strings");
    }
}

/// The app corpus's 04, 06 and 08 were split into local functions to keep each
/// body inside one mark word. These are them with the functions written back
/// into the entry. 04 and 06 reach 74 and 71 registers at `Opt::Full`; 08
/// reaches 61, so it is a regression case rather than a wide one.
#[test]
fn the_app_corpus_runs_unsplit() {
    for (name, source) in [
        (
            "04_kmeans_step",
            include_str!("wide_frame/04_kmeans_step.acvus"),
        ),
        (
            "06_record_transform",
            include_str!("wide_frame/06_record_transform.acvus"),
        ),
        (
            "08_prefix_histogram",
            include_str!("wide_frame/08_prefix_histogram.acvus"),
        ),
    ] {
        let expected = source
            .lines()
            .find_map(|line| line.strip_prefix("// expect: "))
            .unwrap_or_else(|| panic!("{name} states what it prints"));
        assert_eq!(value(source), format!("{expected:?}"), "{name}");
    }
}

// -- Releases, counted -----------------------------------------------------

static RELEASES: AtomicUsize = AtomicUsize::new(0);

/// One counter, and the harness runs tests on parallel threads.
static TRACKED_SCRIPTS: Mutex<()> = Mutex::new(());

struct Measured {
    at_start: usize,
    _scripts: MutexGuard<'static, ()>,
}

impl Measured {
    fn start() -> Self {
        let scripts = TRACKED_SCRIPTS
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Self {
            at_start: RELEASES.load(Ordering::SeqCst),
            _scripts: scripts,
        }
    }

    fn count(&self) -> usize {
        RELEASES.load(Ordering::SeqCst) - self.at_start
    }
}

static MADE: AtomicUsize = AtomicUsize::new(0);

struct Counted;

impl Counted {
    fn made() -> Self {
        MADE.fetch_add(1, Ordering::SeqCst);
        Counted
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        RELEASES.fetch_add(1, Ordering::SeqCst);
    }
}

/// No identity variable, so the values of every `tracked` call are one
/// source and an array of them typechecks.
#[derive(ExternType)]
#[repr(transparent)]
struct Tracked(Vec<Counted>);

#[extern_fn(effect = pure)]
fn tracked(n: i64) -> Tracked {
    Tracked((0..n).map(|_| Counted::made()).collect())
}

#[extern_fn(effect = pure)]
fn rank(t: &Tracked) -> i64 {
    t.0.len() as i64
}

/// `tracked` at an effect, so its calls keep their order around a trap.
#[extern_fn(effect = opaque)]
fn tracked_in_order(n: i64) -> Tracked {
    tracked(n)
}

/// Its result holds what its argument does (RFC-0079 rule 6).
#[extern_fn(effect = pure)]
fn same<T>(x: T) -> T
where
    T: acvus_extern::Var<acvus_extern::kind::Type>,
{
    x
}

fn regs() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        types: [Tracked],
        fns: [tracked, tracked_in_order, rank, same],
    });
    regs
}

fn run_tracked(source: &str, opt: Opt) -> Value {
    let i = Interner::new();
    let parsed = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("the source parses"));
    let cr = check_graph(
        &i,
        parsed,
        &[],
        &FxHashMap::default(),
        regs(),
        Ty::I64,
        opt,
        |_| {},
    )
    .unwrap_or_else(|r| panic!("{opt:?} refused:\n  {}", r.messages.join("\n  ")));
    let (_, mut interp) = execute_compiled(
        &i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(acvus_interpreter::SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    runtime
        .block_on(interp.execute())
        .expect("the run fetches no context")
}

/// Each value here holds two registers for the whole body, so `n` values
/// make a frame of about `2 * n`.
fn ranked(n: usize) -> String {
    let lets: Vec<String> = (0..n).map(|i| format!("let t{i} = tracked(1);")).collect();
    let ranks: Vec<String> = (0..n)
        .map(|i| format!("total = total + rank(&t{i});"))
        .collect();
    format!(
        "{} let total = 0; {} total",
        lets.join(" "),
        ranks.join(" ")
    )
}

fn gathered(n: usize) -> String {
    let elements: Vec<&str> = (0..n).map(|_| "tracked(1)").collect();
    format!("let all = [{}]; all.len() as i64", elements.join(", "))
}

#[test]
fn each_owned_register_is_released_once() {
    for n in WIDTHS {
        for opt in [Opt::None, Opt::Full] {
            let measured = Measured::start();
            assert_eq!(run_tracked(&ranked(n / 2), opt).as_int(), (n / 2) as i64);
            assert_eq!(measured.count(), n / 2, "{} ranked at {opt:?}", n / 2);
            drop(measured);

            let measured = Measured::start();
            assert_eq!(run_tracked(&gathered(n / 2), opt).as_int(), (n / 2) as i64);
            assert_eq!(measured.count(), n / 2, "{} gathered at {opt:?}", n / 2);
        }
    }
}

#[test]
fn a_take_above_word_zero_is_disowned_before_its_operation() {
    let at = |n| {
        let i = Interner::new();
        let blocks =
            script_listing_with_externs(&i, &gathered(n), Context::default(), regs(), Ty::I64);
        ops_of_anywhere(&blocks)
    };
    let wide = at(100);
    let disown = wide
        .iter()
        .position(|op| op == "Disown")
        .unwrap_or_else(|| panic!("a take above word 0 is disowned: {wide:?}"));
    assert_eq!(wide[disown + 1], "MakeArray", "{wide:?}");
    let narrow = at(8);
    assert!(
        !narrow.iter().any(|op| op == "Disown"),
        "a frame of one mark word disowns nothing: {narrow:?}"
    );
}

// -- Storage slots by live range -------------------------------------------

fn frame_len(source: &str) -> u16 {
    let i = Interner::new();
    let prepared = prepared_script_with_externs(&i, source, Context::default(), regs(), Ty::I64);
    main_body(&prepared).frame_len
}

/// Before storages were coloured by live range, this body took 404 registers.
#[test]
fn two_hundred_lets_read_by_reference_run() {
    for opt in [Opt::None, Opt::Full] {
        let measured = Measured::start();
        assert_eq!(run_tracked(&ranked(200), opt).as_int(), 200);
        assert_eq!(measured.count(), 200, "200 ranked at {opt:?}");
    }
    let len = frame_len(&ranked(200));
    assert!((200..=216).contains(&len), "a frame of {len}");
}

fn interleaved(n: usize) -> String {
    let steps: Vec<String> = (0..n)
        .map(|i| format!("let t{i} = tracked(1); total = total + rank(&t{i});"))
        .collect();
    format!("let total = 0; {} total", steps.join(" "))
}

#[test]
fn a_storage_past_its_last_read_shares_its_register() {
    for opt in [Opt::None, Opt::Full] {
        let measured = Measured::start();
        assert_eq!(run_tracked(&interleaved(400), opt).as_int(), 400);
        assert_eq!(measured.count(), 400, "400 interleaved at {opt:?}");
    }
    let len = frame_len(&interleaved(400));
    assert!(len <= 8, "a frame of {len}");
}

#[test]
fn a_storage_a_live_reference_points_into_keeps_its_register() {
    let held_in_a_let = "let a = tracked(2); let r = &a; \
        let b = tracked(3); let x = rank(&b); \
        let c = tracked(5); let y = rank(&c); \
        rank(r) * 100 + x + y";
    let held_by_a_call = "let a = tracked(2); let r = same(&a); \
        let b = tracked(3); let x = rank(&b); \
        let c = tracked(5); let y = rank(&c); \
        rank(r) * 100 + x + y";
    for source in [held_in_a_let, held_by_a_call] {
        for opt in [Opt::None, Opt::Full] {
            let measured = Measured::start();
            assert_eq!(run_tracked(source, opt).as_int(), 208, "{source} at {opt:?}");
            assert_eq!(measured.count(), 10, "{source} at {opt:?}");
        }
    }
}

// -- The frame's bounds, refused -------------------------------------------

/// `opaque` is effectful, so its calls keep their order and every value is
/// made before the first is read.
fn all_live(n: usize) -> String {
    let lets: Vec<String> = (0..n).map(|i| format!("let a{i} = opaque({i});")).collect();
    let sums: Vec<String> = (0..n).map(|i| format!("total = total + a{i};")).collect();
    format!("{} let total = 0; {} total", lets.join(" "), sums.join(" "))
}

#[test]
fn a_body_past_the_frame_is_refused() {
    for opt in [Opt::None, Opt::Full] {
        match corpus::attempt(&all_live(400), opt, Stage::Run) {
            Outcome::Refused(message) => assert!(
                message.contains("registers, past the 320 one frame holds"),
                "{opt:?}: {message}"
            ),
            other => panic!("{opt:?}: {other:?}"),
        }
    }
    assert_eq!(value(&all_live(100)), (0..100).sum::<i64>().to_string());
}

#[test]
fn a_call_past_the_argument_cell_is_refused() {
    let params: Vec<String> = (0..17).map(|i| format!("a{i}")).collect();
    let args: Vec<String> = (0..17).map(|i| i.to_string()).collect();
    let source = format!(
        "let f = |{}| -> {}; f({})",
        params.join(", "),
        params.join(" + "),
        args.join(", ")
    );
    for opt in [Opt::None, Opt::Full] {
        match corpus::attempt(&source, opt, Stage::Run) {
            Outcome::Refused(message) => assert!(
                message.contains("this call lays 17 argument registers, past the 16"),
                "{opt:?}: {message}"
            ),
            other => panic!("{opt:?}: {other:?}"),
        }
    }
}

// -- A literal that traps part way -----------------------------------------

fn try_run_tracked(source: &str, opt: Opt) -> Result<Value, HostError> {
    let i = Interner::new();
    let parsed = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("the source parses"));
    let cr = check_graph(
        &i,
        parsed,
        &[],
        &FxHashMap::default(),
        regs(),
        Ty::I64,
        opt,
        |_| {},
    )
    .unwrap_or_else(|r| panic!("{opt:?} refused:\n  {}", r.messages.join("\n  ")));
    let (_, mut interp) = execute_compiled(
        &i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(acvus_interpreter::SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    runtime.block_on(interp.execute())
}

/// A trap releases nothing and is not ordered with effects (RFC-0048 rule 8),
/// so what this pins is that no element built before the trap is released
/// twice. When this was written, `Opt::None` built all 70 and released one
/// value, and `Opt::Full` trapped before building any.
#[test]
fn an_element_that_traps_releases_no_built_element_twice() {
    let elements: Vec<&str> = (0..100)
        .map(|at| match at {
            70 => "tracked_in_order(1 / zero)",
            _ => "tracked_in_order(1)",
        })
        .collect();
    let source = format!(
        "let one = tracked(1); let zero = rank(&one) - 1; let all = [{}]; all.len() as i64",
        elements.join(", ")
    );
    for opt in [Opt::None, Opt::Full] {
        let measured = Measured::start();
        let made_before = MADE.load(Ordering::SeqCst);
        match try_run_tracked(&source, opt) {
            Err(HostError::Trapped { message }) => {
                assert!(message.contains("divide by zero"), "{opt:?}: {message}")
            }
            other => panic!("{opt:?}: {other:?}"),
        }
        let made = MADE.load(Ordering::SeqCst) - made_before;
        assert!((1..=71).contains(&made), "{made} made at {opt:?}");
        let released = measured.count();
        assert!(released <= made, "{released} released of {made} made at {opt:?}");
    }
}


