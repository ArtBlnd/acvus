//! Pins the pair `4216d0fc` relies on for `Option<String>`: RFC-0026's
//! clone-on-take and the drop the inserter still emits for the option.
//!

use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[tokio::test]
async fn an_option_of_a_string_matched_by_value_is_read_and_dropped() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let o = Some(\"abc\".to_string()); let n = 0; if let Some(s) = o { n = s.len(); }; n",
        Context::default(),
        Ty::U64,
    )
    .await;
    assert_eq!(v.as_int(), 3);
}

#[tokio::test]
async fn an_option_of_a_string_that_does_not_match_is_dropped_once() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let o = strip_prefix(\"abc\", \"zz\"); let n = 0; if let Some(s) = o { n = s.len(); }; n",
        Context::default(),
        Ty::U64,
    )
    .await;
    assert_eq!(v.as_int(), 0);
}

#[tokio::test]
#[should_panic(expected = "a reference cannot be stored in an Option or a Result")]
async fn a_view_as_an_option_payload_is_refused_where_it_answered_202_for_802() {
    let i = Interner::new();
    run_script(
        &i,
        "let o = Some(\"abcdefgh\"); let p = \"ij\"; let q = len(&p); let n = 0; \
         if let Some(s) = o { n = s.len() * 100 + q; }; n",
        Context::default(),
        Ty::U64,
    )
    .await;
}
