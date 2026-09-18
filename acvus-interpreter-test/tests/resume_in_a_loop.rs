//! RFC-0052 rule 4: no suspendable operation inside a fused one.
//!
//! A `while` whose body calls an extern above `Sync` is not recognized as a
//! `Loop`. A `Loop`'s `next` runs its regions to completion inside one
//! terminator call, so a suspend in there would have nowhere to leave to;
//! the recognizer therefore refuses such a call and the body stays flat
//! blocks, with the call's own block ending at the suspend and the work
//! that reads its value beginning the block after it.
//!
//! The listing is where that is read. The count is read from the handler
//! itself, because a resume that re-entered the wrong block would call it
//! a different number of times while still, for this loop, reaching a
//! plausible sum.

use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::listing::{script_listing_with_externs, terminators_depth_first};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

static CALLS: AtomicUsize = AtomicUsize::new(0);

#[extern_fn]
fn fetch_by(x: i64) -> i64 {
    CALLS.fetch_add(1, Ordering::SeqCst);
    x * 10
}

fn io_registry() -> Registry<AcvusRuntime> {
    extern_registry! { ns: "io", fns: [fetch_by], }
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(io_registry());
    regs
}

const AWAIT_IN_A_WHILE: &str =
    "let n = 0; let s = 0; while n < 3 { s = s + fetch_by(n); n = n + 1; } s";

#[tokio::test]
async fn a_while_around_an_await_is_not_a_loop() {
    let interner = Interner::new();
    let blocks = script_listing_with_externs(
        &interner,
        AWAIT_IN_A_WHILE,
        Context::default(),
        registries(),
        Ty::I64,
    );
    let ends = terminators_depth_first(&blocks);
    assert_eq!(
        ends.iter().filter(|name| *name == "Loop").count(),
        0,
        "the recognizer refuses a body holding a call above Sync, so this \
         `while` is blocks and jumps: {ends:?}"
    );
    assert_eq!(
        ends,
        ["Goto", "JumpIf", "Eval<false>", "Goto", "Return<true>"],
        "entry, the head's compare, the call's block ending at the suspend, \
         the block after it, and the return"
    );
}

#[tokio::test]
async fn the_block_after_the_call_is_where_the_resume_lands() {
    let interner = Interner::new();
    let blocks = script_listing_with_externs(
        &interner,
        AWAIT_IN_A_WHILE,
        Context::default(),
        registries(),
        Ty::I64,
    );
    assert_eq!(
        blocks[2].ops,
        ["Mov<false, false>", "SpawnExternSync", "Merge"],
        "the call's own block puts the argument in the run it is lent, spawns \
         and merges, then leaves at the suspend"
    );
    assert_eq!(
        blocks[3].ops,
        ["Add<i64>", "Add<i64>"],
        "`s + <the call's value>` and `n + 1` run after the resume, in the \
         block the suspend named"
    );
}

#[tokio::test]
async fn the_loop_resumes_once_per_iteration_and_no_more() {
    let interner = Interner::new();
    let before = CALLS.load(Ordering::SeqCst);
    let value = run_script_mode_with_externs(
        &interner,
        AWAIT_IN_A_WHILE,
        Context::default(),
        registries(),
        Ty::I64,
    )
    .await;
    assert_eq!(
        value.value.as_int(),
        0 + 10 + 20,
        "fetch_by(0) + fetch_by(1) + fetch_by(2)"
    );
    assert_eq!(
        CALLS.load(Ordering::SeqCst) - before,
        3,
        "one suspend and one resume per iteration, three iterations"
    );
}
