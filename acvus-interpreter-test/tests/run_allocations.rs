//! RFC-0050 rule 4's first judgment: how many aggregates the run keeps off the
//! heap.
//!
//! The count is per iteration, taken as the difference between two loop
//! lengths, so whatever the compile and the frame allocate once does not enter
//! it. Obligation across artifacts: the shape these scripts prepare to is
//! `run_shape.rs`'s subject, and a run form that stopped taking a run would
//! pass every assertion there and fail the numbers here.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_interpreter_test::{Context, int_context, run_script};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

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

async fn allocations(source: &str, n: i64) -> usize {
    let interner = Interner::new();
    let context: Context = int_context(&interner, "n", n);
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let answer = run_script(&interner, source, context, Ty::I64).await;
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    std::hint::black_box(answer);
    after - before
}

async fn per_iteration(source: &str) -> f64 {
    let (few, many) = (1_000i64, 11_000i64);
    let low = allocations(source, few).await;
    let high = allocations(source, many).await;
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
