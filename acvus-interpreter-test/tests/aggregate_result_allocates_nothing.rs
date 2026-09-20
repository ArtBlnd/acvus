//! Rule 2's count, at one declaration and two destinations: a `-> S` call
//! whose result stays in the frame writes the registers the run allocation
//! placed and allocates nothing for it, and the same call whose result
//! crosses back into a handler by value is realized on the heap.
//!
//! Rule 6's count is here for the same reason: a call whose parameter is a
//! projection borrows the caller's aggregate in place, so its count does not
//! grow with the number of calls.
//!
//! The test lives here rather than beside the declaration because the
//! measurement needs the prepared program separated from its run: a global
//! allocator counts every allocation in the binary, and compiling and
//! preparing allocate freely.

use std::sync::Arc;

use acvus_extern::{Registry, TyArg, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
use acvus_interpreter_test::{
    Context, compile_source_with_externs, execute_compiled, split_context,
};
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

thread_local! {
    static ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static LIVE: std::cell::Cell<isize> = const { std::cell::Cell::new(0) };
}

struct Counting;

// SAFETY: every method forwards to the system allocator with the same
// arguments; the counter is a side effect on thread-local state and changes
// no pointer.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        LIVE.with(|n| n.set(n.get() + 1));
        // SAFETY: the caller's contract, forwarded.
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        LIVE.with(|n| n.set(n.get() - 1));
        // SAFETY: the caller's contract, forwarded.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new: usize) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        // SAFETY: the caller's contract, forwarded.
        unsafe { std::alloc::System.realloc(ptr, layout, new) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[derive(TyArg)]
pub struct Point {
    x: i64,
    y: i64,
}

#[derive(TyArg)]
pub struct Tagged {
    label: String,
    n: i64,
}

#[extern_fn(effect = pure)]
fn point_of(a: i64, b: i64) -> Point {
    Point { x: a, y: b }
}

#[extern_fn(effect = pure)]
fn sum_point(p: Point) -> i64 {
    p.x + p.y
}

/// Borrowed by a projection rather than taken by value, so the call reads the
/// caller's two words where they lie (RFC-0050 rule 6).
#[derive(TyArg)]
#[projection]
pub enum Step {
    Done,
    Left(i64),
}

#[extern_fn(effect = pure)]
fn left_of(s: StepRef<'_>) -> i64 {
    match s {
        StepRef::Done => 0,
        StepRef::Left(n) => *n,
    }
}

#[extern_fn(effect = pure)]
fn tagged(n: i64) -> Tagged {
    Tagged {
        label: "x".repeat(64),
        n,
    }
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        types: [],
        fns: [point_of, sum_point, tagged, left_of],
    }
}

struct Measured {
    allocations: usize,
    /// What the run left allocated behind it. A destination register that
    /// the frame did not claim leaks its `Large`; one it claimed twice, or
    /// overwrote without releasing, leaks the value it displaced.
    leaked: isize,
    answer: Value,
}

fn measure(source: &str) -> Measured {
    let interner = Interner::new();
    let (context_types, snapshot) = split_context(&interner, Context::default());
    let ast = ParsedAst::Script(acvus_ast::parse_script(&interner, source).expect("parse error"));
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(registry());
    let compiled = compile_source_with_externs(&interner, ast, &context_types, registries, Ty::I64);
    let (_shared, mut interp) = execute_compiled(
        &interner,
        compiled,
        snapshot,
        Arc::new(SequentialExecutor) as Arc<dyn acvus_interpreter::Executor>,
    );

    let before = ALLOCATIONS.with(std::cell::Cell::get);
    let live = LIVE.with(std::cell::Cell::get);
    let value = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime")
        .block_on(interp.execute());
    let after = ALLOCATIONS.with(std::cell::Cell::get);
    Measured {
        allocations: after - before,
        leaked: LIVE.with(std::cell::Cell::get) - live,
        answer: value,
    }
}

/// The count settles after the first run warms the allocator's cold paths,
/// so the number a case is judged by is the least of three.
fn least_of_three(source: &str) -> usize {
    (0..3)
        .map(|_| measure(source).allocations)
        .min()
        .expect("three runs")
}

fn least_left_behind(source: &str) -> isize {
    (0..3)
        .map(|_| measure(source).leaked)
        .min()
        .expect("three runs")
}

const ITERATIONS: i64 = 64;

fn large_component(iterations: i64) -> String {
    format!(
        "let acc = 0; let i = 0; while i < {iterations} {{ let t = tagged(i); \
         acc = acc + t.n; i = i + 1; }} acc"
    )
}

const IN_THE_FRAME: &str = "let acc = 0; let i = 0; while i < 64 { let p = point_of(i, i); \
acc = acc + p.x + p.y; i = i + 1; } acc";
const ON_THE_HEAP: &str = "let acc = 0; let i = 0; while i < 64 { \
acc = acc + sum_point(point_of(i, i)); i = i + 1; } acc";

#[test]
fn a_frame_resident_aggregate_result_allocates_nothing_and_an_escaping_one_allocates() {
    let want = (0..ITERATIONS).map(|i| i + i).sum::<i64>();
    assert_eq!(measure(IN_THE_FRAME).answer.as_int(), want);
    assert_eq!(measure(ON_THE_HEAP).answer.as_int(), want);

    let framed = least_of_three(IN_THE_FRAME);
    let heaped = least_of_three(ON_THE_HEAP);
    assert!(
        heaped >= framed + usize::try_from(ITERATIONS).expect("64 fits a usize"),
        "the escaping result is one heap object an iteration over {ITERATIONS} iterations, \
         and the two runs allocate {framed} and {heaped}"
    );
}

/// A `String` component lands in a destination register the frame did not
/// write, so the frame takes its claim after the call and drops it before the
/// next iteration overwrites the register. Without either half the run would
/// leave one `String` per iteration allocated behind it, which is what this
/// reads: what the run leaves behind, at two iteration counts an order of
/// magnitude apart.
#[test]
fn a_large_component_of_a_frame_resident_result_is_released_once_per_call() {
    assert_eq!(
        measure(&large_component(64)).answer.as_int(),
        (0..64).sum::<i64>()
    );
    assert_eq!(least_left_behind(&large_component(64)), 0);
    assert_eq!(least_left_behind(&large_component(512)), 0);
}

/// The enum is built once and lent on every iteration, so a call that names
/// the caller's two words where they lie leaves the count flat in the number
/// of calls.
fn borrowed_enum(iterations: i64) -> String {
    format!(
        "let e = if 0 < 1 {{ Step::Left(7) }} else {{ Step::Done }}; \
         let acc = 0; let i = 0; while i < {iterations} {{ \
         acc = acc + left_of(&e); i = i + 1; }} acc"
    )
}

#[test]
fn an_enum_projection_call_allocates_nothing_however_often_it_is_made() {
    assert_eq!(measure(&borrowed_enum(64)).answer.as_int(), 7 * 64);
    assert_eq!(measure(&borrowed_enum(512)).answer.as_int(), 7 * 512);
    assert_eq!(
        least_of_three(&borrowed_enum(64)),
        least_of_three(&borrowed_enum(512)),
        "an enum projection names the caller's two words, so 64 calls and 512 allocate alike"
    );
    assert_eq!(least_left_behind(&borrowed_enum(512)), 0);
}
