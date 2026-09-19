//! A small pure closure called where it was made is its body (RFC-0060), and
//! a `Large` it captured is released exactly once either way.
//!
//! The two sources below differ in one thing: the second's `if` puts its body
//! in two blocks, which is what keeps the inliner off it. Everything else —
//! the capture, the two calls, the value — is the same, so the release count
//! is read one variable apart.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use acvus_extern::{ExternType, IdentityVar, Registry, extern_fn, extern_registry};
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

const ONE_BLOCK: &str = "let t = tracked(1); \
     let f = |x| -> rank(&t) + x; \
     let a = f(1); let b = f(2); a + b";

const TWO_BLOCKS: &str = "let t = tracked(1); \
     let f = |x| -> { let n = rank(&t); let m = if n > 0 { n } else { 0 }; m + x }; \
     let a = f(1); let b = f(2); a + b";

async fn value(source: &str) -> Value {
    let i = Interner::new();
    run_script_mode_with_externs(&i, source, Context::default(), regs(), Ty::I64)
        .await
        .value
}

fn has_indirect_call(source: &str) -> bool {
    let i = Interner::new();
    let blocks = script_listing_with_externs(&i, source, Context::default(), regs(), Ty::I64);
    ops_of_anywhere(&blocks)
        .iter()
        .any(|op| op.starts_with("CallIndirect"))
}

#[tokio::test]
async fn the_one_block_body_is_spliced_and_the_two_block_body_is_called() {
    assert!(!has_indirect_call(ONE_BLOCK));
    assert!(has_indirect_call(TWO_BLOCKS));
}

#[tokio::test]
async fn an_inlined_large_capture_is_released_once_for_two_calls() {
    let measured = Measured::start();
    assert_eq!(value(ONE_BLOCK).await.as_int(), 5);
    assert_eq!(measured.count(), 1);
}

#[tokio::test]
async fn the_called_form_releases_the_same_capture_the_same_number_of_times() {
    let measured = Measured::start();
    assert_eq!(value(TWO_BLOCKS).await.as_int(), 5);
    assert_eq!(measured.count(), 1);
}
