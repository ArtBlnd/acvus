//! Two shared slices of two different elements of one container are two
//! borrows, and each index reads the element it names (RFC-0047 §8).

use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::{Context, run_script_mode_with_externs};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

const TWO_ROWS: &str = "\
let m = vec([vec([1, 2, 3]), vec([10, 20, 30])]); \
let z = len(&m) - len(&m); \
let one = z + 1u64; \
m[z][z] * 1000 + m[one][z]";

const M: [[i64; 3]; 2] = [[1, 2, 3], [10, 20, 30]];

const EXPECTED: i64 = M[0][0] * 1000 + M[1][0];

#[tokio::test]
async fn each_row_is_read_through_its_own_slice() {
    let interner = Interner::new();
    let registries = acvus_ext::std_registries::<AcvusRuntime>();
    let ran =
        run_script_mode_with_externs(&interner, TWO_ROWS, Context::default(), registries, Ty::I64)
            .await;
    assert_eq!(Value::as_int(&ran.value), EXPECTED);
}
