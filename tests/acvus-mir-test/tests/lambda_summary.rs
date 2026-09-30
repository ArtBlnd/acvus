//! RFC-0064 rule 5 at its contract: what a lambda may capture, what it may
//! leave with, and what a lambda holding a loan may not be used as.

use acvus_mir_test::{compile_script_mode_optimized, refuse_script_mode_optimized};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn admits(source: &str) {
    let i = Interner::new();
    if let Err(refusal) = compile_script_mode_optimized(&i, source, &FxHashMap::default()) {
        panic!("expected the program to compile:\n{refusal}");
    }
}

fn refuses(source: &str) -> Vec<String> {
    let i = Interner::new();
    match refuse_script_mode_optimized(&i, source, &FxHashMap::default()) {
        Ok(ir) => panic!("expected a refusal, compiled:\n{ir}"),
        Err(refusals) => refusals.iter().map(|r| r.message.clone()).collect(),
    }
}

fn refuses_with(source: &str, message: &str) {
    let messages = refuses(source);
    assert!(
        messages.iter().any(|m| m.contains(message)),
        "expected {message:?}, got {messages:?}"
    );
}

const CAPTURING: &str = "let v = [1, 2, 3];\nlet r = &v;\nlet f = |k| -> len(r) + k;\n";

// -- A lambda holds what it captures -----------------------------------

#[test]
fn a_lambda_capturing_a_reference_is_admitted() {
    admits(&format!("{CAPTURING}f(1)\n"));
}

#[test]
fn reading_the_borrowed_storage_while_the_lambda_is_live_is_admitted() {
    admits(&format!("{CAPTURING}let n = v[1];\nf(1) + n\n"));
}

#[test]
fn writing_the_borrowed_storage_while_the_lambda_is_live_is_refused() {
    refuses_with(
        &format!("{CAPTURING}v = [4, 5, 6];\nf(1)\n"),
        "`v` is written here while a reference to it is live",
    );
}

/// The lambda is the holder, not `r`: `r`'s last use is the capture, so
/// without the lambda's own loan on `v` the write above would be admitted.
#[test]
fn the_same_write_after_the_lambda_is_dead_is_admitted() {
    admits(&format!(
        "{CAPTURING}let n = f(1);\nv = [4, 5, 6];\nn + v[0]\n"
    ));
}

// -- What a lambda may leave with --------------------------------------

#[test]
fn a_lambda_returning_a_reference_to_its_parameter_is_admitted() {
    admits("let v = [1, 2, 3];\nlet f = |xs| -> &xs[0];\n*f(&v) + 1\n");
}

// -- The refusals that stay --------------------------------------------

/// A view is the register pair of RFC-0047 rule 6 and a capture is one
/// word, so this one capture stays refused where a bare reference is now
/// admitted.
#[test]
fn a_lambda_capturing_a_view_is_refused() {
    refuses_with(
        "let s = \"abc\";\nlet f = |k| -> s.len() + k;\nf(1)\n",
        "a lambda cannot capture a string or slice view",
    );
}

/// RFC-0062 rule 5, as RFC-0064 rule 5 extends it: a lambda that
/// holds a loan is a holder, and a holder is not data.
#[test]
fn a_capturing_lambda_is_not_stored_in_a_list() {
    refuses_with(
        &format!("{CAPTURING}let l = [f];\nlen(&l)\n"),
        "a reference cannot be stored in",
    );
}

#[test]
fn a_lambda_that_captured_nothing_is_stored_in_a_list() {
    admits("let f = |k| -> k + 1;\nlet l = [f];\nlen(&l)\n");
}
