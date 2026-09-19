//! Where a container's slice stands, by the kind of write the loop makes
//! to it (RFC-0047 §2).

use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::listing::{regions_named, script_listing};
use acvus_interpreter_test::{Context, run_script_mode_with_externs};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// An index is a `u64` and the language has no `u64` literal (RFC-0047,
/// "what it costs"), so `one` and `zero` are derived from `len`.
const PRELUDE: &str = "\
let a = vec([1, 2, 3, 4]); \
let n = len(&a); \
let one = n / n; \
let zero = n - n; \
let i = zero; \
let total = 0; \
";

const ELEMENT_WRITE: &str = "\
while i < n { \
a[i] = a[i] + 1; \
total = total + a[i]; \
i = i + one; \
} \
total";

const SHAPE_WRITE: &str = "\
while i < n { \
a[i] = a[i] + 1; \
total = total + a[i]; \
a.push(0); \
i = i + one; \
} \
total";

fn source(body: &str) -> String {
    format!("{PRELUDE}{body}")
}

fn slices_in_the_loop_body(body: &str) -> usize {
    let interner = Interner::new();
    let blocks = script_listing(&interner, &source(body), Context::default(), Ty::I64);
    regions_named(&blocks, "Loop")
        .into_iter()
        .map(|region| {
            let part = region.part("body").expect("a Loop holds a body");
            part.ops
                .iter()
                .filter(|name| name.contains("__extern_fn_as_slice"))
                .count()
        })
        .sum()
}

async fn total(body: &str) -> i64 {
    let interner = Interner::new();
    let registries = acvus_ext::std_registries::<AcvusRuntime>();
    let ran = run_script_mode_with_externs(
        &interner,
        &source(body),
        Context::default(),
        registries,
        Ty::I64,
    )
    .await;
    Value::as_int(&ran.value)
}

/// `1+1` through `4+1`, each element read back after the body wrote it.
const EXPECTED_TOTAL: i64 = 14;

#[tokio::test]
async fn the_push_is_the_only_thing_the_two_scripts_disagree_about() {
    assert_eq!(total(ELEMENT_WRITE).await, EXPECTED_TOTAL);
    assert_eq!(
        total(SHAPE_WRITE).await,
        EXPECTED_TOTAL,
        "the bound is read before the body runs, so what `push` appends is \
         past it and both scripts walk the same four elements"
    );
}

#[tokio::test]
async fn an_element_write_leaves_the_slice_above_the_loop() {
    assert_eq!(
        slices_in_the_loop_body(ELEMENT_WRITE),
        0,
        "`a[i] = …` moves neither the pointer nor the length, so the one \
         `as_slice_mut` above the header serves the reads as well"
    );
}

#[tokio::test]
async fn a_push_keeps_the_slice_inside_the_loop() {
    assert!(
        slices_in_the_loop_body(SHAPE_WRITE) > 0,
        "`push` may reallocate, so a slice taken above the header would \
         name storage the container no longer owns"
    );
}
