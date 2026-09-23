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
    runs_to("let s = \"a\".to_string(); let t = &mut s; clone(t)", "\"a\"");
    runs_to("let v = vec([1, 2]); let t = &mut v; contains(t, &2)", "true");
    runs_to(
        "let s = \"ab\".to_string(); let t = &mut s; contains(t, \"a\")",
        "true",
    );
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

#[test]
fn a_mutable_view_is_reborrowed_as_itself() {
    runs_to(
        "let v = vec([1, 2]); let s = v.as_slice_mut(); let r = &s; len(r)",
        "2",
    );
    runs_to("let v = vec([1, 2]); let s = v.as_slice_mut(); s.len()", "2");
    runs_to("let v = vec([1, 2]); let s = v.as_slice_mut(); len(s)", "2");
}

#[test]
fn a_trapping_arm_leaves_the_references_to_meet() {
    for source in [
        "let a = vec([1]); let e = vec([1, 2]); let c = 0; \
         let r = match c { 0 => &mut a, 1 => &e, _ => panic(\"no\".to_string()), }; r.len()",
        "let a = vec([1]); let e = vec([1, 2]); let c = 1; \
         let r = match c { 0 => panic(\"no\".to_string()), 1 => &mut a, _ => &e, }; r.len()",
    ] {
        runs_to(source, "1");
    }
}
