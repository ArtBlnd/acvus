//! `Runtime::sleep` on this interpreter's runtime (RFC-0075 rules 1 and 4):
//! it waits, on the executor's timer, and outlives the runtime it came from.

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_extern::Runtime;
use acvus_interpreter::{
    AcvusRuntime, Executor, InterpreterContext, SequentialExecutor, TokioExecutor,
};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn runtime_on<E>(executor: E) -> AcvusRuntime
where
    E: Executor + 'static,
{
    InterpreterContext::new(&Interner::new(), FxHashMap::default(), Arc::new(executor))
        .runtime_over_an_empty_page()
}

/// A generic caller sees only the trait's bounds, and those make the future
/// `Send + 'static`: this compiling is the check.
fn _static_send<R>(rt: &R) -> impl Future<Output = ()> + Send + 'static
where
    R: Runtime,
{
    rt.sleep(Duration::ZERO)
}

#[test]
fn sequential_sleep_waits_its_duration() {
    let rt = runtime_on(SequentialExecutor);
    let d = Duration::from_millis(40);
    let start = Instant::now();
    futures::executor::block_on(rt.sleep(d));
    let elapsed = start.elapsed();
    assert!(elapsed >= d, "slept {elapsed:?}, asked for {d:?}");
}

#[tokio::test]
async fn tokio_sleep_waits_its_duration() {
    let rt = runtime_on(TokioExecutor);
    let d = Duration::from_millis(40);
    let start = Instant::now();
    rt.sleep(d).await;
    let elapsed = start.elapsed();
    assert!(elapsed >= d, "slept {elapsed:?}, asked for {d:?}");
}

/// Two joined 60ms sleeps on one current-thread runtime take 60ms on a timer
/// and 120ms as thread sleeps; 110ms leaves 50ms of scheduling slack over the
/// first and stays 10ms under the second.
const TWO_JOINED_60MS_BOUND: Duration = Duration::from_millis(110);

#[tokio::test]
async fn tokio_sleeps_joined_overlap() {
    let rt = runtime_on(TokioExecutor);
    let d = Duration::from_millis(60);
    let start = Instant::now();
    futures::future::join(rt.sleep(d), rt.sleep(d)).await;
    let elapsed = start.elapsed();
    assert!(elapsed >= d, "slept {elapsed:?}, asked for {d:?}");
    assert!(
        elapsed < TWO_JOINED_60MS_BOUND,
        "two joined {d:?} sleeps took {elapsed:?}"
    );
}

#[test]
fn sleep_outlives_the_runtime() {
    let rt = runtime_on(SequentialExecutor);
    let d = Duration::from_millis(20);
    let sleeping = rt.sleep(d);
    drop(rt);
    let start = Instant::now();
    futures::executor::block_on(sleeping);
    let elapsed = start.elapsed();
    assert!(elapsed >= d, "slept {elapsed:?}, asked for {d:?}");
}
