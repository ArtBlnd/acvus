//! A script whose `m`, bound by `if let Some(m) = as_iter(&scores) | max`,
//! is `&Float`, then captured by `|s| -> exp(*s - m)`. In its own binary
//! because the compile of this script must end in a report, not abort the
//! test process.
//!
//! The capture is admitted now (RFC-0064 Decision 2), and the refusal that
//! remains is one signature short of the program: no `max` takes an iterator
//! of references — neither `iter::max`, whose element crosses by value, nor
//! `vec::max`, which takes the container. Closing that is RFC-0047's
//! business and not this test's; what this test holds is that the compile
//! reports rather than overflows.

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

fn context(i: &Interner) -> Context {
    let data = serde_json::json!({ "keys": [[1.0, 0.0], [0.0, 1.0]] });
    data.as_object()
        .expect("an object")
        .iter()
        .map(|(k, v)| (i.intern(k), value_from_json(i, v)))
        .collect()
}

#[tokio::test]
#[should_panic(expected = "type &Float is outside the declared bound one of i64, Float")]
async fn a_reference_captured_after_if_let_over_max_is_reported_not_overflowed() {
    let i = Interner::new();
    let _: Value = run_script_mode(
        &i,
        "let dot = |k| -> 1.0; \
         let scores = as_iter(&@keys) | map(|k| -> dot(k)) | collect; \
         let m = if let Some(m) = as_iter(&scores) | max { m } else { 0.0 }; \
         let weights = as_iter(&scores) | map(|s| -> exp(*s - m)) | collect; \
         weights.len()",
        context(&i),
        Ty::U64,
    )
    .await;
}
