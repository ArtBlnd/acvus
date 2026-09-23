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
    answer: u64,
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
    Balance {
        allocations: ALLOCATIONS.load(Ordering::Relaxed) - allocated,
        releases: RELEASES.load(Ordering::Relaxed) - released,
        answer: answer.bits(),
    }
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

// -- Leaving a `for` over an owned array early (RFC-0057 rule 6) --------

/// What one iteration of `source` leaves allocated, at `opt`, with the run's
/// answer at the larger count checked against `answer`.
async fn left_per_iteration<F>(source: &str, opt: Opt, answer: F) -> f64
where
    F: Fn(i64) -> u64,
{
    let measuring = ONE_AT_A_TIME.lock().expect("no measurement panicked");
    let (few, many) = (1_000i64, 5_000i64);
    let low = balance(source, few, opt).await;
    let high = balance(source, many, opt).await;
    drop(measuring);
    assert_eq!(low.answer, answer(few), "at {opt:?}, the answer at {few}");
    assert_eq!(high.answer, answer(many), "at {opt:?}, the answer at {many}");
    ((high.allocations as f64 - high.releases as f64)
        - (low.allocations as f64 - low.releases as f64))
        / (many - few) as f64
}

/// Each script leaves a `for` over an array of owners by the edge its name
/// gives, at both levels, and every iteration of the enclosing `while` gives
/// back all it allocated: the elements the loop took are released by their
/// own scopes, and the array's `Drop` on the leaving edge releases the rest.
async fn leaves_nothing_behind<F>(source: &str, answer: F)
where
    F: Fn(i64) -> u64,
{
    for opt in [Opt::None, Opt::Full] {
        let left = left_per_iteration(source, opt, &answer).await;
        println!("at {opt:?}: left behind per iteration: {left:.3}");
        assert!(
            left.abs() < 0.01,
            "at {opt:?}, an iteration leaves {left} allocations behind"
        );
    }
}

/// Two taken, "ab" read and "c" released by its scope at the `break`, "d"
/// never taken.
const BREAK_IN_OWNERS: &str = "\
let acc = 0; let i = 0; while i < @n { \
let a = [\"ab\".to_string(), \"c\".to_string(), \"d\".to_string()]; let k = 0; \
for s in a { if k == 1 { break; }; acc = acc + len(&s); k = k + 1; } \
i = i + 1; } acc";

#[tokio::test]
async fn a_break_out_of_an_array_of_strings_releases_the_elements_not_taken() {
    leaves_nothing_behind(BREAK_IN_OWNERS, |n| 2 * n as u64).await;
}

/// The control: a `continue` does not leave the loop, so the terminator's
/// exit is the one edge out, as it is without a jump.
const CONTINUE_IN_OWNERS: &str = "\
let acc = 0; let i = 0; while i < @n { \
let a = [\"ab\".to_string(), \"c\".to_string(), \"d\".to_string()]; \
for s in a { if len(&s) == 1 { continue; }; acc = acc + len(&s); } \
i = i + 1; } acc";

#[tokio::test]
async fn a_continue_in_an_array_of_strings_releases_every_element() {
    leaves_nothing_behind(CONTINUE_IN_OWNERS, |n| 2 * n as u64).await;
}

/// The `?` leaves the lambda from its second element, "c".
const TRY_IN_OWNERS: &str = "\
let f = |k| -> { let acc = 0; \
for s in [\"ab\".to_string(), \"c\".to_string(), \"d\".to_string()] { \
let r = if len(&s) == k { Err(1) } else { Ok(len(&s)) }; acc = acc + r?; } Ok(acc) }; \
let acc = 0; let i = 0; while i < @n { \
acc = acc + match f(1) { Ok(v) => v, Err(e) => e }; \
i = i + 1; } acc";

#[tokio::test]
async fn a_try_out_of_an_array_of_strings_releases_the_elements_not_taken() {
    leaves_nothing_behind(TRY_IN_OWNERS, |n| n as u64).await;
}

/// The `return` leaves the lambda from its second element, "c".
const RETURN_IN_OWNERS: &str = "\
let f = |k| -> { for s in [\"ab\".to_string(), \"c\".to_string(), \"d\".to_string()] { \
if len(&s) == k { return 1; }; } 0 }; \
let acc = 0; let i = 0; while i < @n { acc = acc + f(1); \
i = i + 1; } acc";

#[tokio::test]
async fn a_return_out_of_an_array_of_strings_releases_the_elements_not_taken() {
    leaves_nothing_behind(RETURN_IN_OWNERS, |n| n as u64).await;
}

/// Each outer element runs the inner loop to its `break` at "yz", and the
/// outer loop breaks at "c": per turn of the `while`, 1 + 2 + 1.
const NESTED_BREAKS_IN_OWNERS: &str = "\
let acc = 0; let i = 0; while i < @n { \
for s in [\"ab\".to_string(), \"c\".to_string(), \"d\".to_string()] { \
for t in [\"x\".to_string(), \"yz\".to_string(), \"w\".to_string()] { \
if len(&t) == 2 { break; }; acc = acc + len(&t); } \
if len(&s) == 1 { break; }; acc = acc + len(&s); } \
i = i + 1; } acc";

#[tokio::test]
async fn nested_breaks_out_of_arrays_of_strings_release_the_elements_not_taken() {
    leaves_nothing_behind(NESTED_BREAKS_IN_OWNERS, |n| 4 * n as u64).await;
}

/// The element's type is open where the checker reaches the `break`: the
/// lambda's parameters are settled by the call.
const BREAK_IN_OPEN_ELEMENTS: &str = "\
let f = |x, z| -> { let k = 0; for y in [x, z] { k = k + 1; break; } k }; \
let acc = 0; let i = 0; while i < @n { \
acc = acc + f(\"a\".to_string(), \"b\".to_string()); i = i + 1; } acc";

#[tokio::test]
async fn a_break_out_of_an_array_of_open_elements_releases_the_elements_not_taken() {
    leaves_nothing_behind(BREAK_IN_OPEN_ELEMENTS, |n| n as u64).await;
}

/// An element that is itself an aggregate owning a `String`.
const BREAK_IN_OBJECTS: &str = "\
let acc = 0; let i = 0; while i < @n { \
for o in [{ s: \"ab\".to_string(), }, { s: \"c\".to_string(), }, { s: \"d\".to_string(), }] { \
acc = acc + len(&o.s); break; } \
i = i + 1; } acc";

#[tokio::test]
async fn a_break_out_of_an_array_of_objects_releases_the_elements_not_taken() {
    leaves_nothing_behind(BREAK_IN_OBJECTS, |n| 2 * n as u64).await;
}
