//! `&mut T` reaching `&T` (RFC-0029 rule 3).

use std::time::Duration;

use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_mir::graph::optimize::Opt;

const LIMIT: Duration = Duration::from_secs(30);

/// A run that crashes the process is an outcome here, not the end of the
/// test binary, so each program runs in a process of its own.
fn outcome(source: &str, opt: Opt) -> Outcome {
    acvus_interpreter_test::attempt_within!(source, opt, Stage::Run, LIMIT)
        .unwrap_or_else(|lapse| panic!("at {opt:?}, {lapse:?}: {source}"))
}

fn runs_to(source: &str, value: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, opt) {
            Outcome::Value(got) => assert_eq!(got, value, "at {opt:?}: {source}"),
            other => panic!("at {opt:?}, expected {value}, got {other:?}: {source}"),
        }
    }
}

fn refused_with(source: &str, reason: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, opt) {
            Outcome::Refused(why) => {
                assert!(
                    !why.contains("[validate:"),
                    "at {opt:?}, the MIR validator refused what the checker admitted: {why}"
                );
                assert!(why.contains(reason), "at {opt:?}: {why}");
            }
            other => panic!("at {opt:?}, expected a refusal, got {other:?}: {source}"),
        }
    }
}

#[test]
fn corpus_child() {
    corpus::child();
}

#[test]
fn a_mutable_reference_is_passed_where_a_shared_one_is_taken() {
    runs_to("let v = vec([1]); let t = &mut v; len(t)", "1");
    runs_to("let s = \"a\".to_string(); let t = &mut s; clone(t)", "a");
    runs_to("let o = { a: 7, }; let t = &mut o; clone(t).a", "7");
}

#[test]
fn the_mutable_reference_passed_as_shared_stays_mutable() {
    runs_to(
        "let v = vec([1]); let t = &mut v; let n = len(t); t.push(2); v.len() + n",
        "3",
    );
}

#[test]
fn a_shared_and_a_mutable_reference_join_to_shared() {
    runs_to(
        "let a = vec([1]); let b = vec([1, 2]); let c = true; \
         let r = if c { &mut a } else { &b }; r.len()",
        "1",
    );
    runs_to(
        "let a = vec([1]); let b = vec([1, 2]); let c = false; \
         let r = match c { true => &mut a, false => &b, }; r.len()",
        "2",
    );
}

#[test]
fn the_joined_reference_is_shared() {
    refused_with(
        "let a = vec([1]); let b = vec([1, 2]); let c = true; \
         let r = if c { &mut a } else { &b }; r.push(3); 0",
        "",
    );
}

#[test]
fn a_mutable_reference_joined_as_shared_stays_mutable_where_it_was_bound() {
    runs_to(
        "let a = vec([1]); let b = vec([2]); let t = &mut a; \
         let r = if true { t } else { &b }; let n = r.len(); t.push(5); a.len() + n",
        "3",
    );
}

#[test]
fn a_shared_reference_never_reaches_a_mutable_position() {
    refused_with("let v = vec([1]); let t = &v; t.push(2); 0", "");
    refused_with("let v = vec([1]); let t = &v; push(t, 2); 0", "");
}
