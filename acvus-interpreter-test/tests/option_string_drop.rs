//! Pins the pair `4216d0fc` relies on for `Option<String>`: RFC-0026's
//! clone-on-take and the drop the inserter still emits for the option.
//!
//! The payload is written `.to_string()` because a string literal is a `&str`
//! since RFC-0062, and an `Option<&str>` is a type the checker admits and the
//! machine cannot hold: `MakeSome` and `TakeVar` move one of the runtime's
//! values and a view is two, so the length word comes from whatever register
//! follows. That is a hole in `Some`'s admission, not in this file's subject,
//! and it reads the wrong length at `8e937131` too — `let o =
//! Some("abcdefgh"); let p = "ij"; let q = len(&p); ... s.len() * 100 + q`
//! answers 202 there and 2 here where 802 is right. Closing it is one call to
//! `reject_reference_in_data` at `Some`'s construction in `acvus-mir`.

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
