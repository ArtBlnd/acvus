//! A body wider than one mark word runs: every register of its frame has a
//! mark bit, an operation that takes registers above word 0 drops their
//! claims, and the sweep releases what the frame still owns exactly once
//! (RFC-0048 rules 3 to 6).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use acvus_extern::{ExternType, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_interpreter_test::listing::{ops_of_anywhere, script_listing_with_externs};
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

struct Counted;

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
    Tracked((0..n).map(|_| Counted).collect())
}

#[extern_fn(effect = pure)]
fn rank(t: &Tracked) -> i64 {
    t.0.len() as i64
}

fn regs() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        types: [Tracked],
        fns: [tracked, rank],
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
