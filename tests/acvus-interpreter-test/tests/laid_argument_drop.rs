//! A `Large` laid in a callee's first register is released exactly once
//! (RFC-0052 rule 7).

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use acvus_extern::{ExternType, Registry, Var, extern_fn, extern_registry, kind};
use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::listing::{ops_of_anywhere, script_listing_with_externs};
use acvus_interpreter_test::{Context, run_script_mode_with_externs};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

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

/// An extension type over a `Vec`, so the value crosses as a `Large` and its
/// release is what the counter sees.
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

const CALLS: usize = 1000;

/// The body holds two statements, which is what keeps it a `Body` rather than
/// the one chain a frameless `Code::Expr` is: a frameless call binds no frame,
/// so it has no mark word to claim a `Large` parameter with.
///
/// The `if` puts the body in two blocks, so the inliner leaves the closure
/// alone and there is a call here to read at all (RFC-0060). Flatten it and
/// this file still compiles while both tests lose their subject.
fn source() -> String {
    format!(
        "let take = |t| -> {{ let n = rank(&t); let m = if n > 0 {{ n }} else {{ 0 }}; m }}; \
         let acc = 0; let i = 0; \
         while i < {CALLS} {{ acc = acc + take(tracked(1)); i = i + 1; }} acc"
    )
}

async fn run(i: &Interner, source: &str) -> Value {
    run_script_mode_with_externs(i, source, Context::default(), regs(), Ty::I64)
        .await
        .value
}

#[tokio::test]
async fn a_large_argument_is_released_once_per_call() {
    let measured = Measured::start();
    let i = Interner::new();
    let v = run(&i, &source()).await;
    assert_eq!(v.as_int(), CALLS as i64);
    assert_eq!(measured.count(), CALLS);
}

#[tokio::test]
async fn the_large_reaches_the_callee_through_the_laid_run() {
    let i = Interner::new();
    let blocks = script_listing_with_externs(&i, &source(), Context::default(), regs(), Ty::I64);
    let ops = ops_of_anywhere(&blocks);
    assert!(
        ops.iter().any(|op| op == "LayArg"),
        "the call lays its argument in the callee's first register: {ops:?}"
    );
    assert!(
        ops.iter().any(|op| op.starts_with("CallIndirect<")),
        "the closure call is an operation and not a terminator: {ops:?}"
    );
}
