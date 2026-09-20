//! RFC-0052: a call whose callee's task is `Sync` is an operation, and only a
//! call above `Sync` is a terminator.
//!
//! Each `step` below is a closure the inliner leaves alone, which is what
//! there has to be for a call to read at all (RFC-0060): the sync one is two
//! blocks, and the async one takes an `Order`. What the listing then reads is
//! the task.

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::listing::{
    ops_of, regions_named, script_listing_with_externs, terminators_depth_first,
};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[extern_fn]
async fn bump_later(x: i64) -> i64 {
    x + 1
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! { ns: "io", fns: [bump_later], });
    regs
}

const SYNC_CALL_WHILE: &str =
    "let step = |x| -> if x < 100 { x + 1 } else { x }; let i = 0; while i < 3 { i = step(i); } i";
const ASYNC_CALL_WHILE: &str =
    "let step = |x| -> bump_later(x); let i = 0; while i < 3 { i = step(i); } i";

#[tokio::test]
async fn a_sync_call_is_an_operation_inside_the_loop() {
    let interner = Interner::new();
    let blocks = script_listing_with_externs(
        &interner,
        SYNC_CALL_WHILE,
        Context::default(),
        registries(),
        Ty::I64,
    );
    let loops = regions_named(&blocks, "Loop");
    assert_eq!(loops.len(), 1, "the `while` is one `Loop` region");
    let body = loops[0].part("body").expect("a `Loop` owns its body");
    assert_eq!(
        body.ops,
        ["LayArg", "CallIndirect<false, true, true>"],
        "the body lays the one argument in the callee's first register and then \
         calls: the result is a word the frame opened, and the callee is \
         reached through a reference the caller keeps"
    );
    assert_eq!(
        terminators_depth_first(&blocks),
        ["Goto", "Return<true, false>"],
        "no call terminator is left: the entry block runs the loop and goes to \
         the return"
    );
}

#[tokio::test]
async fn a_call_above_sync_is_a_terminator_and_no_loop_is_recognized() {
    let interner = Interner::new();
    let blocks = script_listing_with_externs(
        &interner,
        ASYNC_CALL_WHILE,
        Context::default(),
        registries(),
        Ty::I64,
    );
    assert_eq!(
        regions_named(&blocks, "Loop").len(),
        0,
        "rule 4 keeps a suspending call out of a region"
    );
    let ends = terminators_depth_first(&blocks);
    assert_eq!(
        ends,
        [
            "Goto",
            "JumpIf<R0>",
            "CallIndirectAsync<false, true>",
            "Goto",
            "Return<true, false>"
        ],
        "entry, the head's compare, the call's own block ending at the \
         suspend, the block the resume lands in, and the return"
    );
    let ops = ops_of(&blocks);
    assert!(
        !ops.iter().any(|op| op.starts_with("CallIndirect<")),
        "the suspending form has no operation: {ops:?}"
    );
}

#[tokio::test]
async fn both_forms_run_to_the_same_value() {
    let interner = Interner::new();
    let sync = run_script_mode_with_externs(
        &interner,
        SYNC_CALL_WHILE,
        Context::default(),
        registries(),
        Ty::I64,
    )
    .await;
    let asynchronous = run_script_mode_with_externs(
        &interner,
        ASYNC_CALL_WHILE,
        Context::default(),
        registries(),
        Ty::I64,
    )
    .await;
    assert_eq!(sync.value.as_int(), 3);
    assert_eq!(asynchronous.value.as_int(), 3);
}
