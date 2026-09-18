//! A context is a place of the page (RFC-0025): `let q = @query` moves out
//! of it, and the run assigns it again before it ends, or the move is
//! reported. The harness is `compile_script_mode_optimized`, whose error is
//! the validation error list, one line per error.

use acvus_mir::ty::{LenTerm, Ty};
use acvus_mir_test::compile_script_mode_optimized;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

const MOVED_OUT: &str =
    "context @query is moved out here and not assigned again before the run ends";

fn query(i: &Interner) -> FxHashMap<Astr, Ty> {
    FxHashMap::from_iter([(
        i.intern("query"),
        Ty::Array(Box::new(Ty::Float), LenTerm::Known(2)),
    )])
}

fn errors(source: &str) -> Vec<String> {
    let i = Interner::new();
    match compile_script_mode_optimized(&i, source, &query(&i)) {
        Ok(ir) => panic!("expected a validation error, compiled:\n{ir}"),
        Err(e) => e.lines().map(str::to_string).collect(),
    }
}

#[test]
fn a_context_taken_into_a_local_and_never_assigned_again_is_named_at_the_move() {
    let errs = errors("let q = @query; let dot = |k| -> k[0]; dot(&q)");
    assert_eq!(errs.len(), 1, "{errs:#?}");
    assert!(errs[0].ends_with(MOVED_OUT), "{errs:#?}");
}

#[test]
fn a_context_assigned_again_before_the_run_ends_is_accepted() {
    let i = Interner::new();
    compile_script_mode_optimized(&i, "let q = @query; @query = q; @query[0]", &query(&i))
        .expect("the assignment gives the context its value again");
}

#[test]
fn a_second_read_of_a_moved_out_context_is_a_use_after_move_at_that_read() {
    let errs = errors("let q = @query; let r = @query;");
    assert_eq!(errs.len(), 1, "{errs:#?}");
    assert!(
        errs[0].ends_with("use of `@query` after it was moved"),
        "{errs:#?}"
    );
}
