//! A flat option's payload is released once, by the binding or by the
//! scrutinee and never by both (RFC-0039: `Some(v)` is `v`, so the
//! scrutinee's register and the binding's hold one value).
//!
//! `Tracked` is an extension type over a `Vec<Counted>`, so the value the
//! script moves around is a `Large` whose release the counter observes; an
//! extension type over an `i64` would cross as a word and count nothing.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use acvus_extern::{ExternType, IdentityVar, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::{Context, run_script_mode_with_externs};
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
    I: IdentityVar;

#[extern_fn(effect = pure)]
fn tracked<I>(n: i64) -> Tracked<I>
where
    I: IdentityVar,
{
    Tracked((0..n).map(|_| Counted).collect(), PhantomData)
}

#[extern_fn(effect = pure)]
fn maybe_tracked<I>(n: i64) -> Option<Tracked<I>>
where
    I: IdentityVar,
{
    (n > 0).then(|| Tracked((0..n).map(|_| Counted).collect(), PhantomData))
}

#[extern_fn(effect = opaque)]
fn opaque_tracked<I>(n: i64) -> Option<Tracked<I>>
where
    I: IdentityVar,
{
    (n > 0).then(|| Tracked((0..n).map(|_| Counted).collect(), PhantomData))
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
        fns: [tracked, maybe_tracked, opaque_tracked, rank],
    });
    regs
}

async fn run(i: &Interner, source: &str) -> Value {
    run_script_mode_with_externs(i, source, Context::default(), regs())
        .await
        .value
}

#[tokio::test]
async fn a_matched_option_releases_its_payload_once() {
    let measured = Measured::start();
    let i = Interner::new();
    let v = run(
        &i,
        "let o = Some(tracked(1)); let n = 0; if let Some(b) = o { n = rank(&b); }; n",
    )
    .await;
    assert_eq!(v.as_int(), 1);
    assert_eq!(measured.count(), 1);
}

#[tokio::test]
async fn a_matched_temporary_releases_its_payload_once() {
    let measured = Measured::start();
    let i = Interner::new();
    let v = run(
        &i,
        "let n = 0; if let Some(b) = maybe_tracked(1) { n = rank(&b); }; n",
    )
    .await;
    assert_eq!(v.as_int(), 1);
    assert_eq!(measured.count(), 1);
}

#[tokio::test]
async fn an_option_nobody_matches_releases_its_payload_once() {
    let measured = Measured::start();
    let i = Interner::new();
    let v = run(&i, "let o = opaque_tracked(1); let n = 3; n").await;
    assert_eq!(v.as_int(), 3);
    assert_eq!(measured.count(), 1);
}

#[tokio::test]
async fn an_empty_option_releases_nothing() {
    let measured = Measured::start();
    let i = Interner::new();
    let v = run(
        &i,
        "let o = maybe_tracked(0); let n = 0; if let Some(b) = o { n = rank(&b); }; n",
    )
    .await;
    assert_eq!(v.as_int(), 0);
    assert_eq!(measured.count(), 0);
}
