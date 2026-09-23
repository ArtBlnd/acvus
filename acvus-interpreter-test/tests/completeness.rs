//! Programs the RFCs admit and the checker refuses. Each is the smallest of
//! its kind in the soundness corpus, pinned to the value it runs to; each
//! fails until the checker admits it.

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

fn refused(source: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, opt) {
            Outcome::Refused(why) => assert!(
                !why.contains("[validate:"),
                "at {opt:?}, the MIR validator refused what the checker admitted: {why}"
            ),
            other => panic!("at {opt:?}, expected a refusal, got {other:?}: {source}"),
        }
    }
}

#[test]
fn corpus_child() {
    corpus::child();
}

/// RFC-0020: `==` on an extension type calls `core::eq`; RFC-0070 rule 5:
/// `Vec<T>` has each core instance. `eq(&a, &b)` runs.
#[test]
fn two_vecs_compare_with_eq() {
    runs_to("vec([1]) == vec([1])", "true");
}

/// RFC-0018 rule 10: a lambda parameter's type comes from unification;
/// RFC-0042: checking opens decisions and the solve settles them.
#[test]
fn an_operator_on_a_lambda_parameter_settles_with_the_call() {
    runs_to("let v = vec([1]); let f = |r| -> r == r; f(&v)", "true");
}

/// RFC-0064 rule 5 names where a reference may not be stored — a container
/// element, an object or tuple field, a context, a spawn — and not an
/// `Option`; RFC-0039 rule 6 lays `Option<&T>` out. `match last(&v)` runs.
#[test]
fn an_option_holds_a_reference() {
    runs_to("let x = 5; match Some(&x) { Some(r) => *r, _ => 0, }", "5");
}

/// RFC-0043 rule 2: admission is asked again once the variable settles;
/// RFC-0062 rule 3: a `&String` reaches `&str`.
#[test]
fn a_method_on_a_lambda_parameter_reaches_a_view() {
    runs_to(
        "let f = |s| -> s.contains(\"a\"); f(&\"ab\".to_string())",
        "true",
    );
}

/// RFC-0024 rule 5: a pattern on a source whose type is still open is
/// checked against its own referent and joined when the type settles.
#[test]
fn a_list_pattern_on_a_lambda_parameter() {
    runs_to(
        "let f = |r| -> match r { [1, x, _z] => *x, _ => 0, }; let a = [1, 5, 9]; f(&a)",
        "5",
    );
}

/// RFC-0062 rule 3: a string pattern admits a `String` or `&str`
/// scrutinee; RFC-0042 rule 1: a pattern is a lower bound.
#[test]
fn a_string_pattern_in_an_object_pattern_on_a_lambda_parameter() {
    runs_to(
        "let f = |r| -> match r { { k: \"nope\", } => 1, _ => 0, }; \
         let o = { k: \"kk\".to_string(), }; f(&o)",
        "0",
    );
}

/// RFC-0018 rules 4 and 10, RFC-0024 rule 7: a field read through a
/// reference whose type the call settles; the same read runs at top level.
#[test]
fn a_field_read_through_a_lambda_parameter() {
    runs_to(
        "let o = { a: { b: 13, }, }; let f = |r| -> r.a.b; f(&o)",
        "13",
    );
}

/// RFC-0018 rules 1, 4 and 10: a store through a parameter the call lends
/// mutably.
#[test]
fn a_store_through_a_lambda_parameter() {
    runs_to(
        "let x = 0; let f = |r| -> { *r = 1; 0 }; f(&mut x); x",
        "1",
    );
}

/// RFC-0018 rule 9: a holder is live to its last read, as liveness counts
/// it; the loan `r` held before it was reassigned is no longer read.
#[test]
fn a_reassigned_reference_releases_its_old_loan() {
    runs_to(
        "let a = vec([1]); let b = vec([1, 2]); let r = &a; r = &b; a.push(3); r.len()",
        "2",
    );
}

/// RFC-0018 rule 10: a captured `String` is seen as `&String`; RFC-0020:
/// `==` reads text operands and references to them alike. The comparison
/// runs when `k` is a `let`.
#[test]
fn a_comparison_with_a_captured_string() {
    runs_to(
        "let mk = |k| -> |x| -> x == k; let g = mk(\"q\".to_string()); g(\"q\".to_string())",
        "true",
    );
}

/// RFC-0043 rule 6: a receiver whose type is still open is lent when every
/// candidate takes a reference; RFC-0029 rules 3-4: that lend settles to a
/// shared reborrow of the `&mut` the call passes.
#[test]
fn a_method_on_a_mutably_lent_lambda_parameter() {
    runs_to(
        "let v = vec([1, 2]); let f = |r| -> match r.last() { Some(x) => *x, None => 0, }; f(&mut v)",
        "2",
    );
}

/// RFC-0018 rule 1: an `Option` copies exactly when its payload does, so
/// an `Option` of a word is a word to `*` (rule 4) and an index (RFC-0047
/// rule 5).
#[test]
fn an_option_of_a_word_is_read_through_a_reference() {
    runs_to("let o = Some(2); let r = &o; match *r { Some(x) => x, None => 0, }", "2");
}

/// RFC-0029 rules 3-4: a pattern against a `&mut T` binding binds as it does
/// against `&T`, each part a shared reborrow.
#[test]
fn a_pattern_against_a_mutable_reference_binding() {
    runs_to(
        "let t = (1, 2); let e = &mut t; match e { (a, _b) => *a, _ => 0, }",
        "1",
    );
}

/// References read out of one holder of a `&mut` loan are shared reads of
/// it and do not exclude one another.
#[test]
fn parts_of_an_element_a_mutable_iterator_yields() {
    runs_to(
        "let v = vec([(1, 2)]); let it = as_iter(&v); let n = 0; \
         while let Some(x) = next(&mut it) { n = match x { (a, _b) => *a, _ => 0, }; } n",
        "1",
    );
}
