//! Every access to a context the host did not declare is refused where it is
//! written: a read, a store, and a pattern that binds it.

use acvus_interpreter_test::corpus::{Outcome, Stage, attempt};
use acvus_mir::graph::optimize::Opt;

fn refused_as_undeclared(source: &str) {
    for opt in [Opt::None, Opt::Full] {
        match attempt(source, opt, Stage::Prepare) {
            Outcome::Refused(why) => assert!(
                why.contains("`@out` is not a declared context"),
                "at {opt:?}: {why}"
            ),
            other => panic!("at {opt:?}, expected a refusal, got {other:?}: {source}"),
        }
    }
}

#[test]
fn a_read_of_an_undeclared_context_is_refused() {
    refused_as_undeclared("let x = @out; 0");
}

#[test]
fn a_store_to_an_undeclared_context_is_refused() {
    refused_as_undeclared("@out = 1; 0");
}

#[test]
fn a_pattern_binding_an_undeclared_context_is_refused() {
    refused_as_undeclared("if let Some(@out) = Some(1) { 0 } else { 1 }");
}
