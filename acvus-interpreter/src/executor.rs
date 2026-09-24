//! Executor trait - controls where and when spawned work runs.
//!
//! Two spawn paths, both handing the executor work it cannot open:
//! - `spawn_blocking`: a `BlockingJob`, a sync ExternFn's call;
//! - `spawn_async`: an `AsyncJob`, an async ExternFn's call or a spawned
//!   run of a body.
//!
//! Both return a `Handle` the executor fills with its own state, and `eval`
//! of it gives the `Done` the job produced. `sleep` is the timer
//! `Runtime::sleep` waits on (RFC-0075 rule 4).
//!
//! The trait is safe, and that is what the opaque types are for. A job's
//! value is crossed at the type the checker settled for its spawn, and only
//! the runtime reads it: an executor can run a job, hold it, or hand back a
//! `Done`, and it cannot make one except by running a job. A `Done` names
//! the job it came from, and the runtime refuses one from another job. A
//! wrong executor can be slow, hang, or panic the run; it cannot make the
//! run read a value at another type.
//!
//! ```
//! use acvus_interpreter::{AsyncJob, BlockingJob, Done, Executor, Handle};
//! use futures::future::BoxFuture;
//!
//! /// Runs every job on the awaiting task when it is evaluated.
//! struct Deferred;
//!
//! enum Work {
//!     Blocking(BlockingJob),
//!     Async(AsyncJob),
//! }
//!
//! impl Executor for Deferred {
//!     fn spawn_blocking(&self, job: BlockingJob) -> Handle {
//!         Handle::new(Work::Blocking(job))
//!     }
//!
//!     fn spawn_async(&self, job: AsyncJob) -> Handle {
//!         Handle::new(Work::Async(job))
//!     }
//!
//!     fn eval(&self, handle: Handle) -> BoxFuture<'_, Done> {
//!         Box::pin(async move {
//!             match handle.downcast::<Work>() {
//!                 Ok(Work::Blocking(job)) => job.run(),
//!                 Ok(Work::Async(job)) => job.await,
//!                 Err(_) => panic!("a handle this executor did not make"),
//!             }
//!         })
//!     }
//!
//!     fn sleep(&self, d: std::time::Duration) -> BoxFuture<'static, ()> {
//!         Box::pin(async move { std::thread::sleep(d) })
//!     }
//! }
//! ```
//!
//! An executor cannot make a `Done` of its own, nor read the value one holds:
//!
//! ```compile_fail,E0451
//! fn forged() -> acvus_interpreter::Done {
//!     acvus_interpreter::Done { job: todo!(), value: todo!() }
//! }
//! ```
//!
//! ```compile_fail,E0616
//! fn read(done: acvus_interpreter::Done) {
//!     let _ = done.value;
//! }
//! ```

use std::any::Any;
use std::future::Future;
use std::panic::resume_unwind;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use acvus_extern::{Holding, Owned};
use futures::future::BoxFuture;
use sync_wrapper::SyncWrapper;

use crate::runtime::AcvusRuntime;
use crate::value::Value;

// -- Jobs --------------------------------------------------------------

/// Which spawn a job is: the runtime names each one once, and a `Done` is
/// accepted only for the spawn that names it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct JobId(u64);

impl JobId {
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        JobId(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// What a job produced, owned until the runtime takes it.
pub struct Done {
    job: JobId,
    value: Owned<AcvusRuntime>,
}

impl Done {
    fn of(job: JobId, value: Value) -> Self {
        // SAFETY: the job moved its result out to this holder, and no other
        // holder owns it.
        let value = unsafe { Owned::from_value(Holding::new(), value) };
        Done { job, value }
    }

    /// The value, where this is what `job` produced.
    ///
    /// # Panics
    /// The executor gave back the completion of another job.
    pub(crate) fn of_job(self, job: JobId) -> Value {
        assert_eq!(
            self.job, job,
            "the executor gave back the completion of another job than the one evaluated"
        );
        // SAFETY: the runtime takes the word into the register the spawn's
        // result goes to, which owns it from then on.
        self.value.into_value(unsafe { Holding::new() })
    }
}

/// A sync call to run on a thread the executor chooses.
pub struct BlockingJob {
    job: JobId,
    work: Box<dyn FnOnce() -> Value + Send + Sync>,
}

impl BlockingJob {
    pub(crate) fn new(work: Box<dyn FnOnce() -> Value + Send + Sync>) -> Self {
        BlockingJob {
            job: JobId::next(),
            work,
        }
    }

    pub(crate) fn id(&self) -> JobId {
        self.job
    }

    pub fn run(self) -> Done {
        Done::of(self.job, (self.work)())
    }
}

/// Work that awaits, polled where the executor chooses.
pub struct AsyncJob {
    job: JobId,
    work: SyncWrapper<Pin<Box<dyn Future<Output = Value> + Send>>>,
}

impl AsyncJob {
    pub(crate) fn new(work: Pin<Box<dyn Future<Output = Value> + Send>>) -> Self {
        AsyncJob {
            job: JobId::next(),
            work: SyncWrapper::new(work),
        }
    }

    pub(crate) fn id(&self) -> JobId {
        self.job
    }
}

impl Future for AsyncJob {
    type Output = Done;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Done> {
        let job = self.job;
        self.work
            .get_mut()
            .as_mut()
            .poll(cx)
            .map(|value| Done::of(job, value))
    }
}

/// An executor's own record of a spawn, which `eval` is handed back.
pub struct Handle(Box<dyn Any + Send + Sync>);

impl Handle {
    pub fn new<T>(state: T) -> Self
    where
        T: Any + Send + Sync,
    {
        Handle(Box::new(state))
    }

    pub fn downcast<T>(self) -> Result<T, Handle>
    where
        T: Any,
    {
        self.0.downcast::<T>().map(|state| *state).map_err(Handle)
    }
}

impl std::fmt::Debug for Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Handle")
    }
}

// -- Trait -------------------------------------------------------------

/// Where and when spawned work runs: sequentially, on a pool, or on any
/// other schedule.
pub trait Executor: Send + Sync {
    fn spawn_blocking(&self, job: BlockingJob) -> Handle;

    fn spawn_async(&self, job: AsyncJob) -> Handle;

    /// The completion of the job `handle` was spawned with. A panic in the
    /// job is resumed here, on the awaiting run's thread.
    fn eval(&self, handle: Handle) -> BoxFuture<'_, Done>;

    /// A future that waits `d`.
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()>;
}

impl<E> Executor for Box<E>
where
    E: Executor + ?Sized,
{
    fn spawn_blocking(&self, job: BlockingJob) -> Handle {
        (**self).spawn_blocking(job)
    }

    fn spawn_async(&self, job: AsyncJob) -> Handle {
        (**self).spawn_async(job)
    }

    fn eval(&self, handle: Handle) -> BoxFuture<'_, Done> {
        (**self).eval(handle)
    }

    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
        (**self).sleep(d)
    }
}

// -- SequentialExecutor -----------------------------------------------

enum Deferred {
    Blocking(BlockingJob),
    Async(AsyncJob),
}

/// Simplest executor - spawn stores the job, eval runs it on the awaiting
/// task. No parallelism. Good for testing and deterministic execution.
pub struct SequentialExecutor;

impl Executor for SequentialExecutor {
    fn spawn_blocking(&self, job: BlockingJob) -> Handle {
        Handle::new(Deferred::Blocking(job))
    }

    fn spawn_async(&self, job: AsyncJob) -> Handle {
        Handle::new(Deferred::Async(job))
    }

    fn eval(&self, handle: Handle) -> BoxFuture<'_, Done> {
        Box::pin(async move {
            match handle.downcast::<Deferred>() {
                Ok(Deferred::Blocking(job)) => job.run(),
                Ok(Deferred::Async(job)) => job.await,
                Err(_) => panic!("eval: handle was not spawned by SequentialExecutor"),
            }
        })
    }

    /// Sleeps the thread: this executor runs nothing else while a run
    /// waits, so there is no other work to yield to. The thread is blocked
    /// for `d`, and so is any other task a host runs on it.
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
        Box::pin(async move { std::thread::sleep(d) })
    }
}

// -- TokioExecutor ----------------------------------------------------

/// Work starts at spawn as a tokio task and eval joins it. Two spawns
/// that stand before their evals run concurrently; the schedule the
/// compiler emitted (RFC-0007) is what decides that.
pub struct TokioExecutor;

type Joined = tokio::task::JoinHandle<Done>;

impl Executor for TokioExecutor {
    fn spawn_blocking(&self, job: BlockingJob) -> Handle {
        Handle::new(tokio::task::spawn_blocking(move || job.run()))
    }

    fn spawn_async(&self, job: AsyncJob) -> Handle {
        Handle::new(tokio::spawn(job))
    }

    fn eval(&self, handle: Handle) -> BoxFuture<'_, Done> {
        Box::pin(async move {
            let joined = handle
                .downcast::<Joined>()
                .unwrap_or_else(|_| panic!("eval: handle was not spawned by TokioExecutor"));
            match joined.await {
                Ok(done) => done,
                Err(joined) if joined.is_panic() => resume_unwind(joined.into_panic()),
                Err(joined) => panic!("spawned task failed: {joined}"),
            }
        })
    }

    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
        Box::pin(tokio::time::sleep(d))
    }
}
