//! RFC-0105 rule 4: a prepared body drops by a loop over its operations, so
//! a body of more than 40 000 operations drops on a test thread's stack, and
//! so does a long part a region owns, which the region hands to that loop.

use acvus_ext::std_registries;
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::Context;
use acvus_interpreter_test::listing::{
    listing, main_body, ops_of, ops_of_anywhere, prepared_script_with_externs,
};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// Each step is an arithmetic chain and an extern call, so twice as many
/// operations.
const STEPS: u32 = 20_000;

const OPS_AT_LEAST: usize = 40_000;

/// The loop's part: preparing a `while` body of `STEPS` statements takes
/// minutes in a debug build, and this many runs `DROP_STACK` out when a part
/// is dropped through each successor.
const PART_STEPS: u32 = 5_000;

const PART_OPS_AT_LEAST: usize = 10_000;

/// The stack the body is dropped on. A drop that loops needs the same few
/// frames for any body, so a small stack separates it from one that nests
/// per operation, whatever the harness gives a test thread.
const DROP_STACK: usize = 256 << 10;

/// Prepares `source`, asserts it holds `at_least` operations, of which at
/// most `top_level_at_most` are outside every region's parts, and drops it
/// on a thread of `DROP_STACK`.
fn prepares_and_drops(source: &str, at_least: usize, top_level_at_most: usize) {
    let interner = Interner::new();
    let mut registries = std_registries::<AcvusRuntime>();
    registries.push(acvus_wasm_probe::registry());
    let prepared =
        prepared_script_with_externs(&interner, source, Context::default(), registries, Ty::I64);
    let blocks = listing(main_body(&prepared).heads());
    let ops = ops_of_anywhere(&blocks).len();
    assert!(
        ops >= at_least,
        "the body prepares to {ops} operations, fewer than {at_least}"
    );
    let top_level = ops_of(&blocks).len();
    assert!(
        top_level <= top_level_at_most,
        "{top_level} of the body's operations are outside a region, more than {top_level_at_most}"
    );
    let dropped = std::thread::Builder::new()
        .stack_size(DROP_STACK)
        .spawn(move || drop(prepared))
        .expect("the dropping thread starts");
    dropped.join().expect("the body drops");
}

#[test]
fn a_straight_body_of_forty_thousand_operations_drops() {
    prepares_and_drops(
        &acvus_wasm_probe::straight_body(STEPS),
        OPS_AT_LEAST,
        usize::MAX,
    );
}

/// The `while` counts `i` to a bound, so it prepares as a `For` over a range,
/// whose `body` part holds the steps.
#[test]
fn a_long_loop_part_drops() {
    let mut source = String::from("let a = 1;\nlet k = 7;\nlet p = 1000003;\nlet i = 0;\n");
    source.push_str("while i < 1 {\n");
    for _ in 0..PART_STEPS {
        source.push_str("a = step(a * k % p);\n");
    }
    source.push_str("i = i + 1;\n}\na\n");
    prepares_and_drops(&source, PART_OPS_AT_LEAST, 100);
}
