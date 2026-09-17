//! Pins the pair `4216d0fc` relies on for `Option<String>`: RFC-0026's
//! clone-on-take and the drop the inserter still emits for the option.

use acvus_interpreter_test::*;
use acvus_utils::Interner;

#[tokio::test]
async fn an_option_of_a_string_matched_by_value_is_read_and_dropped() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let o = Some(\"abc\"); let n = 0; Some(s) = o { n = s.len(); }; n",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_int(), 3);
}

#[tokio::test]
async fn an_option_of_a_string_that_does_not_match_is_dropped_once() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let o = strip_prefix(\"abc\", \"zz\"); let n = 0; Some(s) = o { n = s.len(); }; n",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_int(), 0);
}
