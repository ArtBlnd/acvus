//! An object a body never lets out is never built, and each of its
//! move-only fields is released exactly once (RFC-0050).
//!
//! `Tracked` is an extension type over a `Vec<Counted>`, so the value the
//! script moves around is a `Large` whose release the counter observes; an
//! extension type over an `i64` would cross as a word and count nothing.
//! The object shell is an `FxHashMap` the interpreter allocates and no
//! counter can see, so its absence is read off the compiled MIR instead.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use acvus_extern::{ExternType, Registry, Var, extern_fn, extern_registry, kind};
use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::{
    Context, compile_source_with_externs, run_script_mode_with_externs, split_context,
};
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

static RELEASES: AtomicUsize = AtomicUsize::new(0);

/// `RELEASES` is one counter and the harness runs these tests on parallel
/// threads, so every script that builds a `Tracked` runs under this lock.
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

#[derive(ExternType)]
#[repr(transparent)]
struct Tracked<I>(Vec<Counted>, PhantomData<I>)
where
    I: Var<kind::Identity>;

#[extern_fn(effect = pure)]
fn tracked<I>(n: i64) -> Tracked<I>
where
    I: Var<kind::Identity>,
{
    Tracked((0..n).map(|_| Counted).collect(), PhantomData)
}

#[extern_fn(effect = pure)]
fn rank<I>(t: &Tracked<I>) -> i64
where
    I: Var<kind::Identity>,
{
    t.0.len() as i64
}

fn regs() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        types: [Tracked<_>],
        fns: [tracked, rank],
    });
    regs
}

const ITERATIONS: i64 = 3;
const FIELDS_PER_ITERATION: usize = 2;

/// Two `Large` fields built and read once each, in a loop, reaching
/// nothing outside the body.
fn two_large_fields() -> String {
    format!(
        "let n = 0; let i = 0; while i < {ITERATIONS} {{ \
         let q = {{ a: tracked(1), b: tracked(1), }}; \
         let x = q.a; let y = q.b; \
         n = n + rank(&x) + rank(&y); \
         i = i + 1; }} n"
    )
}

async fn run(i: &Interner, source: &str) -> Value {
    run_script_mode_with_externs(i, source, Context::default(), regs(), Ty::I64)
        .await
        .value
}

fn optimized(i: &Interner, source: &str) -> String {
    let context_types = split_context(i, Context::default()).0;
    let ast = ParsedAst::Script(acvus_ast::parse_script(i, source).expect("parse error"));
    let compiled = compile_source_with_externs(i, ast, &context_types, regs(), Ty::I64);
    compiled
        .modules
        .values()
        .map(|module| acvus_mir::printer::dump(i, module))
        .collect()
}

#[tokio::test]
async fn an_object_that_stays_in_the_body_releases_each_field_once() {
    let measured = Measured::start();
    let i = Interner::new();
    let value = run(&i, &two_large_fields()).await;
    assert_eq!(value.as_int(), ITERATIONS * FIELDS_PER_ITERATION as i64);
    assert_eq!(
        measured.count(),
        ITERATIONS as usize * FIELDS_PER_ITERATION,
        "each field is released once per iteration, and the shell adds none"
    );
}

#[tokio::test]
async fn the_object_the_loop_reads_is_never_built() {
    let i = Interner::new();
    let listing = optimized(&i, &two_large_fields());
    assert!(
        !listing.contains("object "),
        "an object nothing outside the body reaches is never built: {listing}"
    );
    assert!(
        !listing.contains("take q"),
        "a field of an object that does not exist is read from no storage: {listing}"
    );
}
