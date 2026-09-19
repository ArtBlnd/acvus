//! The customers measured here are `acvus-ext`'s `string::{trim, upper}`,
//! and the test lives in `acvus-interpreter-test` rather than beside them
//! because the measurement needs the prepared program separated from its
//! run: a global allocator counts every allocation in the binary, and
//! compiling and preparing allocate freely.

use std::sync::Arc;

use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
use acvus_interpreter_test::{
    compile_source_with_externs, execute_compiled, split_context, string_context,
};
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

thread_local! {
    static ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct Counting;

// SAFETY: every method forwards to the system allocator with the same
// arguments; the counter is a side effect on thread-local state and changes
// no pointer.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        // SAFETY: the caller's contract, forwarded.
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
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

fn allocations_of_the_run(source: &str) -> (usize, Value) {
    let interner = Interner::new();
    let context = string_context(&interner, "text", "  hello  ");
    let (context_types, snapshot) = split_context(&interner, context);
    let ast = ParsedAst::Script(acvus_ast::parse_script(&interner, source).expect("parse error"));
    let compiled = compile_source_with_externs(
        &interner,
        ast,
        &context_types,
        acvus_ext::std_registries::<AcvusRuntime>(),
        Ty::U64,
    );
    let (_shared, mut interp) = execute_compiled(
        &interner,
        compiled,
        snapshot,
        Arc::new(SequentialExecutor) as Arc<dyn acvus_interpreter::Executor>,
    );

    let before = ALLOCATIONS.with(std::cell::Cell::get);
    let value = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime")
        .block_on(interp.execute());
    let after = ALLOCATIONS.with(std::cell::Cell::get);
    (after - before, value)
}

/// The count settles after the first run warms the allocator's cold paths,
/// so the number a case is judged by is the least of three.
fn least_of_three(source: &str) -> usize {
    (0..3)
        .map(|_| allocations_of_the_run(source).0)
        .min()
        .expect("three runs")
}

#[test]
fn a_view_returning_call_allocates_nothing_and_an_owned_one_allocates() {
    let (_, value) = allocations_of_the_run("let v = trim(&@text); len(&v)");
    assert_eq!(value.as_int(), 5, "trim of \"  hello  \" is 5 bytes");
    let (_, value) = allocations_of_the_run("let v = upper(&@text); len(&v)");
    assert_eq!(value.as_int(), 9);

    let viewed = least_of_three("let v = trim(&@text); len(&v)");
    let owned = least_of_three("let v = upper(&@text); len(&v)");
    assert!(
        viewed < owned,
        "trim hands back a run of the argument's own bytes and upper builds new ones, \
         so the runs allocate {viewed} and {owned}"
    );
}
