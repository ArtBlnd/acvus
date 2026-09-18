//! A frame the stage owns is reused across elements, and the sweep still
//! runs per call (RFC-0052 §6).
//!
//! `map`'s stage makes one `Store` when it is built and lends it to every
//! call. Nothing about the frame's lifetime changes what the frame owes: a
//! `Large` the closure's body produces is claimed by the mark word of that
//! one frame and released when the body returns.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use acvus_extern::{ExternType, IdentityVar, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Value};
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
    I: IdentityVar;

#[extern_fn(effect = pure)]
fn tracked<I>(n: i64) -> Tracked<I>
where
    I: IdentityVar,
{
    Tracked((0..n).map(|_| Counted).collect(), PhantomData)
}

#[extern_fn(effect = pure)]
fn rank<I>(t: &Tracked<I>) -> i64
where
    I: IdentityVar,
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

async fn run(i: &Interner, source: &str) -> Value {
    run_script_mode_with_externs(i, source, Context::default(), regs(), Ty::I64)
        .await
        .value
}

const ELEMENTS: usize = 1000;

#[tokio::test]
async fn a_large_in_a_closure_body_is_released_once_per_element() {
    let measured = Measured::start();
    let i = Interner::new();
    let source =
        format!("range(0, {ELEMENTS}) | map(|x| -> {{ let t = tracked(1); rank(&t) }}) | sum");
    let v = run(&i, &source).await;
    assert_eq!(v.as_int(), ELEMENTS as i64);
    assert_eq!(measured.count(), ELEMENTS);
}

/// The same body with a capture, which `enter` writes outside the binding.
#[tokio::test]
async fn a_captured_closure_releases_once_per_element() {
    let measured = Measured::start();
    let i = Interner::new();
    let source = format!(
        "let k = 1; range(0, {ELEMENTS}) | map(|x| -> {{ let t = tracked(k); rank(&t) }}) | sum"
    );
    let v = run(&i, &source).await;
    assert_eq!(v.as_int(), ELEMENTS as i64);
    assert_eq!(measured.count(), ELEMENTS);
}
