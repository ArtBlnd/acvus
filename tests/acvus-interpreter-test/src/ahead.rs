//! The externs and the run RFC-0103's tests and corpus rows share: a heavy
//! parser and kernel that do real work, io calls that wait different times so
//! they complete out of order, and a line log in place of stdout.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::{
    AcvusRuntime, Declined, Declines, Executable, Executor, HostError, Interpreter,
    InterpreterContext, Lower, Lowering, PrepareCtx, SequentialExecutor, TokioExecutor,
    prepare_module,
};
use acvus_mir::graph::{ParsedAst, QualifiedRef};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

use crate::corpus::render;
use crate::{compile_source_with_externs, split_context};

/// The rounds of `crunch`'s and `weigh`'s mixing: enough that a call is
/// work a pool thread is worth, and few enough that a test of sixty-four
/// calls stays well under a second. Chosen for this test, not measured.
const ROUNDS: u64 = 200_000;

/// The unit of an io call's wait: a call at index `n` waits `5 - n % 5`
/// units, so within each five the later indices complete first.
const FAST_STEP: Duration = Duration::from_millis(4);
pub const SLOW_STEP: Duration = Duration::from_millis(40);

/// The calls of the overlap test.
pub const SLOW_CALLS: u32 = 8;

/// What the io externs saw of one run.
#[derive(Default)]
pub struct Wire {
    lines: Mutex<Vec<String>>,
    completed: Mutex<Vec<String>>,
    in_flight: AtomicUsize,
    most_in_flight: AtomicUsize,
}

impl Wire {
    async fn answer(&self, url: &str, step: Duration) -> String {
        let n = index_of(url);
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.most_in_flight.fetch_max(now, Ordering::SeqCst);
        let steps = u32::try_from(5 - n % 5).expect("a step count below six");
        tokio::time::sleep(step * steps).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        self.completed.lock().expect("the completion log").push(url.to_owned());
        format!(" {} ", n * 3)
    }

    pub fn lines(&self) -> Vec<String> {
        self.lines.lock().expect("the line log").clone()
    }

    pub fn most_in_flight(&self) -> usize {
        self.most_in_flight.load(Ordering::SeqCst)
    }

    pub fn completed(&self) -> Vec<String> {
        self.completed.lock().expect("the completion log").clone()
    }
}

/// The number after a url's last `/`, up to its first non-digit; a url
/// with no `/` is all tail.
fn index_of(url: &str) -> u64 {
    let tail = url.rsplit_once('/').map_or(url, |(_, tail)| tail);
    let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().unwrap_or_else(|_| panic!("the url {url:?} ends in no index"))
}

fn mixed(seed: u64) -> u64 {
    let mut h = seed ^ 0x9E37_79B9_7F4A_7C15;
    for round in 0..ROUNDS {
        h = h.rotate_left(7) ^ h.wrapping_mul(0xBF58_476D_1CE4_E5B9).wrapping_add(round);
    }
    h
}

/// `n` scaled, with a remainder below a thousand that only the mixing
/// gives.
fn crunched(n: i64) -> i64 {
    let remainder = i64::try_from(mixed(n.cast_unsigned()) % 1000)
        .expect("a remainder below a thousand is an i64");
    n * 1000 + remainder
}

/// `None` where the text is no number.
#[extern_fn(heavy, effect = pure, returns)]
fn crunch(s: String) -> Option<i64> {
    assert_ne!(s, "boom", "crunch: the job traps on its input");
    s.trim().parse().ok().map(crunched)
}

#[extern_fn(heavy, effect = pure, returns)]
fn crunch_ref(s: &String) -> Option<i64> {
    s.trim().parse().ok().map(crunched)
}

#[extern_fn(heavy, effect = pure, returns)]
fn weigh(x: i64) -> i64 {
    i64::try_from(mixed(x.cast_unsigned()) % 10_000).expect("a remainder below ten thousand is an i64")
}

#[extern_fn(effect = pure, returns)]
async fn fetch(#[state] wire: &Arc<Wire>, url: String) -> String {
    wire.answer(&url, FAST_STEP).await
}

#[extern_fn(effect = idempotent)]
async fn post(#[state] wire: &Arc<Wire>, url: String) -> String {
    wire.answer(&url, FAST_STEP).await
}

#[extern_fn(effect = pure, returns)]
async fn fetch_slow(#[state] wire: &Arc<Wire>, url: String) -> String {
    wire.answer(&url, SLOW_STEP).await
}

#[extern_fn(effect = opaque)]
fn emit(#[state] wire: &Arc<Wire>, line: &str) {
    wire.lines.lock().expect("the line log").push(line.to_owned());
}

pub fn registries(wire: &Arc<Wire>) -> Vec<Registry<AcvusRuntime>> {
    let wire = Arc::clone(wire);
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "ah",
        fns: [
            crunch,
            crunch_ref,
            weigh,
            fetch(Arc::clone(&wire)),
            post(Arc::clone(&wire)),
            fetch_slow(Arc::clone(&wire)),
            emit(Arc::clone(&wire)),
        ],
    });
    regs
}

#[derive(Clone, Copy, Debug)]
pub enum On {
    Sequential,
    Tokio,
}

impl On {
    pub fn executor(self) -> Arc<dyn Executor> {
        match self {
            On::Sequential => Arc::new(SequentialExecutor),
            On::Tokio => Arc::new(TokioExecutor),
        }
    }
}

/// What a run gave, compared across the lowering and the executors.
#[derive(Debug, PartialEq)]
pub struct Outcome {
    pub value: Result<serde_json::Value, String>,
    pub lines: Vec<String>,
}

pub struct Ran {
    pub outcome: Outcome,
    pub wire: Arc<Wire>,
    /// Whether the entry body holds `ForAhead`, and the loops `prepare`
    /// declined.
    pub lowered: bool,
    pub declined: Vec<Declined>,
    pub took: Duration,
}

pub async fn run(source: &str, ret: Ty, lower: Lower, on: On) -> Ran {
    let interner = Interner::new();
    let wire = Arc::new(Wire::default());
    let (types, snapshot) = split_context(&interner, Default::default());
    let ast = ParsedAst::Script(acvus_ast::parse_script(&interner, source).expect("parse error"));
    let cr = compile_source_with_externs(&interner, ast, &types, registries(&wire), ret);
    let mut functions = cr.extern_executables;
    let declines = Declines::default();
    let ctx = PrepareCtx {
        interner: &interner,
        externs: &functions,
        context_names: &cr.context_names,
        instances: &cr.instances,
        access: acvus_mir::graph::Access::Sync,
        lowering: match lower {
            Lower::Ahead => Lowering::Ahead {
                laws: &cr.laws,
                declined: &declines,
            },
            Lower::InPlace => Lowering::InPlace,
        },
    };
    let prepared: Vec<(QualifiedRef, Executable)> = cr
        .modules
        .iter()
        .map(|(qref, module)| {
            let prepared = prepare_module(module, &ctx)
                .unwrap_or_else(|refused| panic!("the body is refused: {refused}"));
            (*qref, Executable::Module(Arc::new(prepared)))
        })
        .collect();
    let Some(Executable::Module(entry)) = prepared
        .iter()
        .find(|(qref, _)| *qref == cr.entry_qref)
        .map(|(_, held)| held)
    else {
        panic!("the entry is prepared")
    };
    let listed = serde_json::to_string(&acvus_interpreter::listing::body_listing(entry.main()))
        .expect("a listing serializes");
    let lowered = listed.contains("\"ForAhead<");
    assert_eq!(
        lowered,
        listed.contains("\"ForAheadStart<"),
        "a loop lowered ahead is entered by `ForAheadStart`"
    );
    functions.extend(prepared);
    let shared = InterpreterContext::new(&interner, functions, on.executor())
        .with_fn_types(cr.fn_types)
        .with_context_names(cr.context_names);
    let mut interp = Interpreter::new(shared, cr.entry_qref, snapshot);
    let started = Instant::now();
    let value = match interp.execute().await {
        Ok(value) => Ok(render(&interner, &value)),
        Err(HostError::Trapped { message }) => Err(message),
        Err(other) => panic!("the run ended with {other}"),
    };
    let took = started.elapsed();
    Ran {
        outcome: Outcome {
            value,
            lines: wire.lines(),
        },
        wire,
        lowered,
        declined: declines.take(),
        took,
    }
}

