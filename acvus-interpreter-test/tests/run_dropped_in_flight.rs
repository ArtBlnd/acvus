//! A run dropped while a task it spawned is in flight, or a body that returns
//! before its task's `Eval`, keeps the cells the task was lent until the task
//! has finished (RFC-0079 rule 9).
//!
//! The program lends `&n` to a spawned extern that reads `n` only when the
//! test opens its gate. The test drops the run's future, or lets the body
//! return, first. Every block this binary frees is overwritten before it is
//! returned to the system, so a read of released cells answers the fill
//! pattern rather than 41.

use std::alloc::{GlobalAlloc, Layout, System};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context as TaskContext, Poll};

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::{
    AcvusRuntime, AsyncJob, BlockingJob, Done, Executor, Handle, Interpreter, TokioExecutor, Value,
};
use acvus_interpreter_test::{Context, check_source, execute_compiled, split_context};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ir::InstKind;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use futures::channel::oneshot;
use futures::future::BoxFuture;

const FREED_FILL: u8 = 0xAB;

static LENT_CELL_ADDRESS: AtomicUsize = AtomicUsize::new(0);
static LENT_CELL_FREED: AtomicBool = AtomicBool::new(false);

struct Filling;

// SAFETY: every method forwards to the system allocator with the same
// arguments; `dealloc` writes only inside the block it is handed, which the
// caller owns until the forwarded call returns.
unsafe impl GlobalAlloc for Filling {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let lent = LENT_CELL_ADDRESS.load(Ordering::SeqCst);
        let start = ptr.addr();
        if lent != 0 && (start..start + layout.size()).contains(&lent) {
            LENT_CELL_FREED.store(true, Ordering::SeqCst);
        }
        // SAFETY: the block is the caller's, `layout.size()` bytes long.
        unsafe { ptr.write_bytes(FREED_FILL, layout.size()) };
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Filling = Filling;

#[derive(Default)]
struct Handshake {
    entered: bool,
    open: bool,
    observed: Option<i64>,
}

static GATE: Mutex<Handshake> = Mutex::new(Handshake {
    entered: false,
    open: false,
    observed: None,
});
static GATE_MOVED: Condvar = Condvar::new();

fn gate() -> MutexGuard<'static, Handshake> {
    GATE.lock().expect("no test panics while holding the gate")
}

fn wait_until(ready: impl Fn(&Handshake) -> bool) {
    let guard = gate();
    let _ready = GATE_MOVED
        .wait_while(guard, |state| !ready(state))
        .expect("no test panics while holding the gate");
}

fn move_gate(step: impl FnOnce(&mut Handshake)) {
    step(&mut gate());
    GATE_MOVED.notify_all();
}

fn read_once_opened(n: &i64) -> i64 {
    LENT_CELL_ADDRESS.store(std::ptr::from_ref(n).addr(), Ordering::SeqCst);
    move_gate(|state| state.entered = true);
    wait_until(|state| state.open);
    let value = *n;
    move_gate(|state| state.observed = Some(value));
    value
}

#[extern_fn(effect = opaque)]
fn read_when_opened(n: &i64) -> i64 {
    read_once_opened(n)
}

#[extern_fn(heavy, effect = pure)]
fn read_heavily_when_opened(n: &i64) -> i64 {
    read_once_opened(n)
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        types: [],
        fns: [read_when_opened, read_heavily_when_opened],
    }
}

const PROGRAM: &str = "let n = 41; let m = read_when_opened(&n); m + 1";

const RETURNS_PAST_ITS_TASK: &str = "let n = 41; let m = read_when_opened(&n); 7";

fn interpreter(executor: Arc<dyn Executor>) -> Interpreter {
    interpreter_of(PROGRAM, Opt::Full, executor, |_| {})
}

fn interpreter_of<F>(source: &str, opt: Opt, executor: Arc<dyn Executor>, edit: F) -> Interpreter
where
    F: FnOnce(&mut acvus_mir::ir::MirModule),
{
    let interner = Interner::new();
    let (context_types, snapshot) = split_context(&interner, Context::default());
    let ast = ParsedAst::Script(acvus_ast::parse_script(&interner, source).expect("parses"));
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(registry());
    let mut compiled =
        check_source(&interner, ast, &context_types, registries, Ty::I64, opt, |_| {})
            .unwrap_or_else(|refused| panic!("refused: {}", refused.messages.join("; ")));
    let entry = compiled
        .modules
        .get_mut(&compiled.entry_qref)
        .expect("the entry is compiled");
    edit(entry);
    execute_compiled(&interner, compiled, snapshot, executor).1
}

/// Unsplit, the heavy call is awaited where it is made rather than spawned.
fn not_split(module: &mut acvus_mir::ir::MirModule) {
    assert!(
        !module
            .main
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::Spawn { .. })),
        "the heavy call is not split"
    );
}

/// No pass leaves a `Spawn` without its `Eval`; the test removes the `Eval`
/// from the compiled body, so the body returns with its task aloft.
fn drop_the_eval(module: &mut acvus_mir::ir::MirModule) {
    let insts = &mut module.main.insts;
    let evals = insts
        .iter()
        .filter(|inst| matches!(inst.kind, InstKind::Eval { .. }))
        .count();
    assert_eq!(evals, 1, "the body evaluates the lent call once");
    insts.retain(|inst| !matches!(inst.kind, InstKind::Eval { .. }));
}

static SERIAL: Mutex<()> = Mutex::new(());

fn fresh() -> MutexGuard<'static, ()> {
    let serial = SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    *gate() = Handshake::default();
    LENT_CELL_ADDRESS.store(0, Ordering::SeqCst);
    LENT_CELL_FREED.store(false, Ordering::SeqCst);
    serial
}

type Job = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct Parking {
    parked: Mutex<Vec<Job>>,
}

impl Parking {
    fn park(&self, run: impl FnOnce() -> Done + Send + 'static) -> Handle {
        let (answer, answered) = oneshot::channel();
        self.parked
            .lock()
            .expect("no test panics while holding the queue")
            .push(Box::new(move || match answer.send(run()) {
                Ok(()) => {}
                Err(_unread) => {}
            }));
        Handle::new(answered)
    }

    fn take_one(&self) -> Job {
        let mut parked = self.parked.lock().expect("no test panics while holding the queue");
        assert_eq!(parked.len(), 1, "the run spawned the lent call and nothing else");
        parked.pop().expect("one job")
    }
}

impl Executor for Parking {
    fn spawn_blocking(&self, job: BlockingJob) -> Handle {
        self.park(move || job.run())
    }

    fn spawn_async(&self, job: AsyncJob) -> Handle {
        self.park(move || futures::executor::block_on(job))
    }

    fn eval(&self, handle: Handle) -> BoxFuture<'_, Done> {
        Box::pin(async move {
            let answered = handle
                .downcast::<oneshot::Receiver<Done>>()
                .unwrap_or_else(|_| panic!("a handle this executor did not make"));
            answered.await.expect("the parked work ran to its value")
        })
    }

    fn sleep(&self, d: std::time::Duration) -> BoxFuture<'static, ()> {
        Box::pin(async move { std::thread::sleep(d) })
    }
}

fn park_at_eval(run: &mut Pin<Box<dyn Future<Output = Value> + '_>>) {
    let waker = futures::task::noop_waker();
    let mut cx = TaskContext::from_waker(&waker);
    assert!(
        matches!(run.as_mut().poll(&mut cx), Poll::Pending),
        "the run waits on the spawned call"
    );
}

fn assert_lent_cells_outlived_the_run() {
    assert_eq!(gate().observed, Some(41), "the task read the cell it was lent");
    assert!(
        LENT_CELL_FREED.load(Ordering::SeqCst),
        "the lent cell is released once the task has finished"
    );
}

fn join_blocking_pool(runtime: tokio::runtime::Runtime) {
    drop(runtime);
}

#[test]
fn a_task_running_when_the_run_is_dropped_reads_live_cells() {
    let _serial = fresh();
    let parking = Arc::new(Parking::default());
    let mut interp = interpreter(Arc::clone(&parking) as Arc<dyn Executor>);
    let mut run: Pin<Box<dyn Future<Output = Value> + '_>> = Box::pin(async { interp.execute().await.expect("the seeds hold every context the run fetches") });
    park_at_eval(&mut run);
    let worker = std::thread::spawn(parking.take_one());
    wait_until(|state| state.entered);

    drop(run);
    assert!(
        !LENT_CELL_FREED.load(Ordering::SeqCst),
        "a dropped run keeps the cells its task was lent"
    );
    move_gate(|state| state.open = true);
    worker.join().expect("the task finishes");

    assert_lent_cells_outlived_the_run();
}

#[test]
fn a_task_started_after_the_run_is_dropped_reads_live_cells() {
    let _serial = fresh();
    let parking = Arc::new(Parking::default());
    let mut interp = interpreter(Arc::clone(&parking) as Arc<dyn Executor>);
    let mut run: Pin<Box<dyn Future<Output = Value> + '_>> = Box::pin(async { interp.execute().await.expect("the seeds hold every context the run fetches") });
    park_at_eval(&mut run);
    let job = parking.take_one();

    drop(run);
    move_gate(|state| state.open = true);
    let worker = std::thread::spawn(job);
    worker.join().expect("the task finishes");

    assert_lent_cells_outlived_the_run();
}

#[test]
fn a_tokio_task_running_when_the_run_is_cancelled_reads_live_cells() {
    let _serial = fresh();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .expect("a multi-thread runtime");
    let mut interp = interpreter(Arc::new(TokioExecutor));
    let run = runtime.spawn(async move { interp.execute().await.expect("the seeds hold every context the run fetches") });
    wait_until(|state| state.entered);

    run.abort();
    let cancelled = runtime.block_on(run);
    assert!(
        cancelled.as_ref().is_err_and(tokio::task::JoinError::is_cancelled),
        "the run was dropped while its task waited"
    );
    assert!(
        !LENT_CELL_FREED.load(Ordering::SeqCst),
        "a cancelled run keeps the cells its task was lent"
    );
    move_gate(|state| state.open = true);
    join_blocking_pool(runtime);

    assert_lent_cells_outlived_the_run();
}

#[test]
fn a_run_that_is_not_dropped_answers_the_lent_read() {
    let _serial = fresh();
    move_gate(|state| state.open = true);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .expect("a multi-thread runtime");
    let mut interp = interpreter(Arc::new(TokioExecutor));
    let value = runtime.block_on(interp.execute()).expect("the seeds hold every context the run fetches");
    assert_eq!(value.bits(), 42);
}

#[test]
fn a_body_that_returns_with_its_task_aloft_keeps_the_cells_until_the_task_ends() {
    let _serial = fresh();
    let parking = Arc::new(Parking::default());
    let mut interp = interpreter_of(
        RETURNS_PAST_ITS_TASK,
        Opt::Full,
        Arc::clone(&parking) as Arc<dyn Executor>,
        drop_the_eval,
    );
    let value = futures::executor::block_on(interp.execute()).expect("the seeds hold every context the run fetches");
    assert_eq!(value.bits(), 7);
    let job = parking.take_one();

    move_gate(|state| state.open = true);
    let worker = std::thread::spawn(job);
    worker.join().expect("the task finishes");

    assert_lent_cells_outlived_the_run();
}

#[test]
fn a_heavy_call_running_when_the_run_is_dropped_reads_live_cells() {
    let _serial = fresh();
    let parking = Arc::new(Parking::default());
    let mut interp = interpreter_of(
        "let n = 41; let m = read_heavily_when_opened(&n); m + 1",
        Opt::None,
        Arc::clone(&parking) as Arc<dyn Executor>,
        not_split,
    );
    let mut run: Pin<Box<dyn Future<Output = Value> + '_>> = Box::pin(async { interp.execute().await.expect("the seeds hold every context the run fetches") });
    park_at_eval(&mut run);
    let worker = std::thread::spawn(parking.take_one());
    wait_until(|state| state.entered);

    drop(run);
    assert!(
        !LENT_CELL_FREED.load(Ordering::SeqCst),
        "a dropped run keeps the cells its heavy call was lent"
    );
    move_gate(|state| state.open = true);
    worker.join().expect("the heavy call finishes");

    assert_lent_cells_outlived_the_run();
}
