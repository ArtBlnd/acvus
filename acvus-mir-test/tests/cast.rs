//! What `expr as T` refuses (RFC-0049). The values it produces are stated
//! where they arrive, in `acvus-interpreter-test/tests/cast.rs`.

use acvus_mir_test::compile_script_optimized;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn refusal(source: &str) -> String {
    let i = Interner::new();
    let found: Vec<String> = compile_script_optimized(&i, source, &FxHashMap::default())
        .expect_err(&format!("{source} is refused"))
        .lines()
        .map(|line| line.split("] ").nth(1).unwrap_or(line).to_string())
        .collect();
    assert_eq!(found.len(), 1, "{source}: {found:?}");
    found.into_iter().next().expect("one refusal")
}

#[test]
fn a_cast_to_a_name_that_is_not_a_number_is_refused() {
    assert_eq!(
        refusal("1 as Widget"),
        "`as` converts to i8, i16, i32, i64, u8, u16, u32, u64 or f64, not to `Widget`"
    );
}

/// `float` and `int` are not spellings this language has: `f64` and the
/// eight widths of RFC-0037 are the names, so the two read as any other
/// unknown type.
#[test]
fn float_and_int_are_not_spellings() {
    assert!(refusal("1 as float").contains("not to `float`"));
    assert!(refusal("1.0 as int").contains("not to `int`"));
}

#[test]
fn a_cast_of_a_value_that_is_not_a_number_is_refused() {
    assert_eq!(
        refusal("let s = \"x\"; s as i64"),
        "`as` converts a number; String is not one"
    );
    assert_eq!(
        refusal("let b = true; b as i64"),
        "`as` converts a number; Bool is not one"
    );
}
