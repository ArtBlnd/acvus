//! The script here is the owner's finding from
//! `acvus-interpreter-test/tests/attention_shape_overflow.rs`, and the
//! context mirrors that test's `{"keys": [[1.0, 0.0], [0.0, 1.0]]}`. Both
//! must move together: this file checks at the checker's contract what that
//! one checks at the interpreter's.

use acvus_mir::ty::{LenTerm, Ty};
use acvus_mir_test::compile_script_mode_raw;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

const SCRIPT: &str = "let dot = |k| -> 1.0; \
     let scores = as_iter(&@keys) | map(|k| -> dot(k)) | collect; \
     let m = if let Some(m) = as_iter(&scores) | max { m } else { 0.0 }; \
     let weights = as_iter(&scores) | map(|s| -> exp(*s - m)) | collect; \
     weights.len()";

fn keys_2x2(i: &Interner) -> FxHashMap<Astr, Ty> {
    let row = Ty::Array(Box::new(Ty::Float), LenTerm::Known(2));
    FxHashMap::from_iter([(
        i.intern("keys"),
        Ty::Array(Box::new(row), LenTerm::Known(2)),
    )])
}

#[test]
fn an_unresolved_call_inside_a_lambda_is_reported_not_overflowed() {
    let i = Interner::new();
    compile_script_mode_raw(&i, SCRIPT, &keys_2x2(&i))
        .expect_err("the unresolved call is reported as an error");
}
