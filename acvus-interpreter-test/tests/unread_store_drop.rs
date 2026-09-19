//! A store nothing reads is dead, and the `Large` it would have stored is
//! released exactly once.
//!
//! The two sources below differ in one thing: the second reads the binding.
//! The value, the call that makes it and the release are the same, so the
//! release count is read one variable apart.

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
struct Held<I>(Vec<Counted>, PhantomData<I>)
where
    I: IdentityVar;

/// Opaque, so the call itself is a root: what the store leaves without a
/// reader is a value that exists and owes a release.
#[extern_fn(effect = opaque)]
fn held<I>(n: i64) -> Held<I>
where
    I: IdentityVar,
{
    Held((0..n).map(|_| Counted).collect(), PhantomData)
}

#[extern_fn(effect = pure)]
fn rank<I>(t: &Held<I>) -> i64
where
    I: IdentityVar,
{
    t.0.len() as i64
}

fn regs() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "h",
        types: [Held<_>],
        fns: [held, rank],
    });
    regs
}

const NEVER_READ: &str = "let t = held(3); 7";

const READ: &str = "let t = held(3); rank(&t) + 4";

async fn value(source: &str) -> Value {
    let i = Interner::new();
    run_script_mode_with_externs(&i, source, Context::default(), regs(), Ty::I64)
        .await
        .value
}

#[tokio::test]
async fn a_large_a_store_left_without_a_reader_is_released_once() {
    let measured = Measured::start();
    assert_eq!(value(NEVER_READ).await.as_int(), 7);
    assert_eq!(measured.count(), 3);
}

#[tokio::test]
async fn the_read_form_releases_the_same_value_the_same_number_of_times() {
    let measured = Measured::start();
    assert_eq!(value(READ).await.as_int(), 7);
    assert_eq!(measured.count(), 3);
}
