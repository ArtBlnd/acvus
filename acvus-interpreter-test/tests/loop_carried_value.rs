//! A move-only value carried around a loop, where the back edge writes the
//! slot its own previous iteration left: the case that tells a definition
//! from an assignment, and the case `Machine::define`'s `debug_assert!`
//! would catch if the register selector or drop insertion got it wrong.

use acvus_interpreter_test::*;
use acvus_utils::Interner;

#[tokio::test]
async fn a_string_rebuilt_every_iteration_keeps_one_owner() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let s = \"\"; let n = 0; while n < 3 { s = s + \"ab\"; n = n + 1; } s.len()",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_int(), 6);
}

#[tokio::test]
async fn a_deque_carried_around_a_loop_keeps_every_element() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let d = deque(); let n = 0; while n < 4 { push_back(&mut d, n); n = n + 1; } len(&d)",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_int(), 4);
}

#[tokio::test]
async fn a_string_carried_through_a_branch_keeps_one_owner() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let s = \"\"; let n = 0; while n < 4 { if n < 2 { s = s + \"a\"; } else { s = s + \"bb\"; }; n = n + 1; } s.len()",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_int(), 6);
}
