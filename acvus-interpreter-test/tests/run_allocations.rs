//! RFC-0050 rule 4's first judgment: how many aggregates the run keeps off the
//! heap.
//!
//! The count is per iteration, taken as the difference between two loop
//! lengths, so whatever the compile and the frame allocate once does not enter
//! it. Obligation across artifacts: the shape these scripts prepare to is
//! `run_shape.rs`'s subject, and a run form that stopped taking a run would
//! pass every assertion there and fail the numbers here.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_interpreter::{AcvusRuntime, SequentialExecutor};
use acvus_interpreter_test::{
    Context, check_source, execute_compiled, int_context, run_script, split_context,
};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static RELEASES: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every call forwards to `System`, which is a sound allocator, and the
// counter is not read inside the allocation.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        RELEASES.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTER: Counting = Counting;

/// The inner loop is what makes `e` a storage whose address is taken, which is
/// the web `run_shape.rs` pins: without it `optimize::sroa` scalarizes the
/// aggregate in the MIR and no aggregate reaches the machine at all.
const RUN_RESIDENT: &str = "\
let acc = 0; let i = 0; while i < @n { \
let e = if i % 3 == 0 { E::A(i) } else { if i % 3 == 1 { E::B(i + 1) } else { E::C(i + 2) } }; \
let j = 0; while j < 1 { match e { E::A(v) => { acc = acc + v; }, \
E::B(v) => { acc = acc + v; }, E::C(v) => { acc = acc + v; } }; j = j + 1; } \
i = i + 1; } acc";

/// `RUN_RESIDENT`'s shape at the one variant type whose tags the language names
/// rather than a declaration.
const RESULT_RUN_RESIDENT: &str = "\
let acc = 0; let i = 0; while i < @n { \
let r = if i % 2 == 0 { Ok(i) } else { Err(i + 1) }; \
let j = 0; while j < 1 { match r { Ok(v) => { acc = acc + v; }, \
Err(v) => { acc = acc + v; } }; j = j + 1; } \
i = i + 1; } acc";

/// The same three-armed construction with one member crossing into an array,
/// which `prepare::runs::Sites` refuses, so every iteration realizes a variant
/// on the heap (RFC-0050 rule 4) and reads its tag and payload there.
const HEAP_RESIDENT: &str = "\
let acc = 0; let i = 0; while i < @n { \
let e = if i % 3 == 0 { E::A(i) } else { if i % 3 == 1 { E::B(i + 1) } else { E::C(i + 2) } }; \
let v = [e]; let m = len(&v); let z = m - m; \
match &v[z] { E::A(w) => { acc = acc + *w; }, E::B(w) => { acc = acc + *w; }, \
E::C(w) => { acc = acc + *w; } }; i = i + 1; } acc";

async fn allocations(source: &str, n: i64) -> usize {
    let interner = Interner::new();
    let context: Context = int_context(&interner, "n", n);
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let answer = run_script(&interner, source, context, Ty::I64).await;
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    std::hint::black_box(answer);
    after - before
}

/// `ALLOCATIONS` counts the whole process, so two measurements running at once
/// each read the other's allocations. This is what keeps them apart, and
/// without it a second test in this file silently moves the first's number.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

async fn per_iteration(source: &str) -> f64 {
    let measuring = ONE_AT_A_TIME.lock().expect("no measurement panicked");
    let (few, many) = (10_000i64, 110_000i64);
    let low = allocations(source, few).await;
    let high = allocations(source, many).await;
    drop(measuring);
    (high as f64 - low as f64) / (many - few) as f64
}

#[tokio::test]
async fn a_run_resident_variant_allocates_nothing_per_iteration() {
    let run = per_iteration(RUN_RESIDENT).await;
    println!("allocations per iteration: {run:.3}");
    assert!(
        run < 0.01,
        "a run-resident variant allocates {run} per iteration"
    );
}

#[tokio::test]
async fn a_run_resident_result_allocates_nothing_per_iteration() {
    let run = per_iteration(RESULT_RUN_RESIDENT).await;
    println!("allocations per iteration: {run:.3}");
    assert!(
        run < 0.01,
        "a run-resident Result allocates {run} per iteration"
    );
}

/// Before RFC-0050 rule 4's flat variant, a variant with a payload was two
/// allocations: a header holding the tag and a box holding the payload. This
/// script read 4.000 at `bc5841db` and reads 3.000 here; the two it keeps are
/// the array and the slice the match indexes through, which no build changes.
#[tokio::test]
async fn a_heap_resident_variant_is_one_allocation() {
    let heap = per_iteration(HEAP_RESIDENT).await;
    println!("allocations per iteration: {heap:.3}");
    assert!(
        (heap - 3.0).abs() < 0.05,
        "the heap variant script allocates {heap} per iteration"
    );
}

/// A `String` payload of a run-resident variant, read by a pattern in a loop
/// and once more after it. At `Opt::None` no pass scalarizes `e`, so every
/// read is a take of a register of the run.
const PAYLOAD_READ_BY_PATTERNS: &str = "\
let acc = 0; let i = 0; while i < @n { \
let e = if i % 2 == 0 { E::A(i.to_string()) } else { E::B((i + 1).to_string()) }; \
let j = 0; while j < 1 { match e { E::A(v) => { acc = acc + v.len(); }, \
E::B(v) => { acc = acc + v.len(); } }; j = j + 1; } \
let c = match e { E::A(t) => t, E::B(t) => t, }; acc = acc + c.len(); \
i = i + 1; } acc";

struct Balance {
    allocations: usize,
    releases: usize,
}

async fn balance(source: &str, n: i64, opt: Opt) -> Balance {
    let interner = Interner::new();
    let (context_types, snapshot) = split_context(&interner, int_context(&interner, "n", n));
    let ast = ParsedAst::Script(acvus_ast::parse_script(&interner, source).expect("parse error"));
    let compiled = check_source(
        &interner,
        ast,
        &context_types,
        acvus_ext::std_registries::<AcvusRuntime>(),
        Ty::U64,
        opt,
        |_| {},
    )
    .unwrap_or_else(|refusal| panic!("at {opt:?}: {}", refusal.messages.join("\n")));
    let (_shared, mut interp) =
        execute_compiled(&interner, compiled, snapshot, Arc::new(SequentialExecutor));
    let allocated = ALLOCATIONS.load(Ordering::Relaxed);
    let released = RELEASES.load(Ordering::Relaxed);
    let answer = interp.execute().await;
    let balance = Balance {
        allocations: ALLOCATIONS.load(Ordering::Relaxed) - allocated,
        releases: RELEASES.load(Ordering::Relaxed) - released,
    };
    std::hint::black_box(answer);
    balance
}

#[tokio::test]
async fn a_string_payload_read_by_patterns_leaves_nothing_behind() {
    for opt in [Opt::None, Opt::Full] {
        let measuring = ONE_AT_A_TIME.lock().expect("no measurement panicked");
        let (few, many) = (1_000i64, 5_000i64);
        let low = balance(PAYLOAD_READ_BY_PATTERNS, few, opt).await;
        let high = balance(PAYLOAD_READ_BY_PATTERNS, many, opt).await;
        drop(measuring);
        let span = (many - few) as f64;
        let allocated = (high.allocations as f64 - low.allocations as f64) / span;
        let left = ((high.allocations as f64 - high.releases as f64)
            - (low.allocations as f64 - low.releases as f64))
            / span;
        println!("at {opt:?}: allocations per iteration: {allocated:.3}, left behind: {left:.3}");
        assert!(
            left.abs() < 0.01,
            "at {opt:?}, an iteration leaves {left} allocations behind"
        );
    }
}
