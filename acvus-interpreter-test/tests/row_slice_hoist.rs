//! Where the slice of `rows[li][i]` stands: the inner loop's own body, or
//! the block above it (RFC-0047 §8).

use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::listing::{regions_named, script_listing};
use acvus_interpreter_test::{Context, run_script_mode_with_externs};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// An index is a `u64` and the language has no `u64` literal (RFC-0047,
/// "what it costs"), so `one` and `zero` are derived from `len`.
const ROWS: &str = "\
let rows = vec([vec([1, 2]), vec([3, 4, 5])]); \
let n = len(&rows); \
let one = n / n; \
let zero = n - n; \
let li = zero; \
let total = 0; \
while li < n { \
let m = len(&rows[li]); \
let i = zero; \
while i < m { \
total = total + rows[li][i]; \
i = i + one; \
} \
li = li + one; \
} \
total";

/// 1 + 2 + 3 + 4 + 5.
const EXPECTED_TOTAL: i64 = 15;

fn slices_in_the_inner_loop_body() -> usize {
    let interner = Interner::new();
    let blocks = script_listing(&interner, ROWS, Context::default(), Ty::I64);
    let outer = regions_named(&blocks, "Loop")
        .into_iter()
        .next()
        .expect("the script runs two nested loops");
    outer
        .owns
        .iter()
        .flat_map(|part| part.regions.iter())
        .filter(|region| region.name.starts_with("Loop"))
        .flat_map(|inner| inner.owns.iter())
        .map(|part| part.ops.iter().filter(|name| *name == "CallSlice").count())
        .sum()
}

#[tokio::test]
async fn the_script_sums_every_element_of_every_row() {
    let interner = Interner::new();
    let registries = acvus_ext::std_registries::<AcvusRuntime>();
    let ran =
        run_script_mode_with_externs(&interner, ROWS, Context::default(), registries, Ty::I64)
            .await;
    assert_eq!(Value::as_int(&ran.value), EXPECTED_TOTAL);
}

#[tokio::test]
async fn the_rows_slice_is_taken_once_a_row_and_not_once_an_element() {
    assert_eq!(
        slices_in_the_inner_loop_body(),
        0,
        "`rows[li]` is the row the block above the inner loop already named, \
         so its slice is taken there"
    );
}
