//! Each form a call is written in, pinned to the value it runs to or the
//! refusal the checker gives it.

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
fn a_free_function_of_one_signature() {
    runs_to("abs(0 - 3)", "3");
}

#[test]
fn a_free_function_of_several_signatures() {
    runs_to("max(2, 5)", "5");
}

#[test]
fn a_method_of_one_signature() {
    runs_to("(0 - 3).abs()", "3");
    runs_to("let v = vec([1]); v.push(2); v.len()", "2");
}

#[test]
fn a_method_of_several_signatures() {
    runs_to("let v = vec([1, 2, 3]); v.len()", "3");
    runs_to("let v = [4, 5]; v.max()", "5");
}

/// The view a lent argument reaches is the one it lends, whichever form
/// the call is written in and however many signatures the name has.
#[test]
fn a_lent_container_reaches_a_slice_parameter_in_every_call_form() {
    for source in [
        "let v = vec([3, 1, 2]); v.max()",
        "let v = vec([3, 1, 2]); max(&v)",
        "let v = vec([3, 1, 2]); slice::max(&v)",
        "let a = [3, 1, 2]; max(&a)",
        "let v = vec([3, 1, 2]); let r = &v; max(r)",
    ] {
        runs_to(source, "3");
    }
    for source in [
        "let v = vec([3, 1, 2]); v.contains(&2)",
        "let v = vec([3, 1, 2]); let x = 2; contains(&v, &x)",
        "let s = \"ab\".to_string(); contains(&s, \"a\")",
        "let s = \"ab\".to_string(); let r = &s; contains(r, \"a\")",
    ] {
        runs_to(source, "true");
    }
}

#[test]
fn a_local_closure_is_called_directly_and_as_a_method() {
    runs_to("let v = [1, 2]; let f = |x| -> x.len(); f(&v) + v.f()", "4");
}

#[test]
fn a_pipe_stage_is_a_call_with_the_piped_value_first() {
    runs_to("(0 - 3) | abs", "3");
    runs_to("3 | max(5)", "5");
    runs_to("let v = vec([1, 2, 3]); &v | len", "3");
    runs_to("let f = |a, b| -> a + b; 1 | f(2)", "3");
    runs_to("2 | (|y| -> y * 3)", "6");
}

#[test]
fn an_index_is_the_container_s_own_as_slice() {
    runs_to("let v = vec([1, 2, 3]); v[1u64]", "2");
    runs_to("let a = [1, 2, 3]; a[2u64]", "3");
}

#[test]
fn a_for_over_a_borrow_is_the_container_s_own_as_slice() {
    runs_to(
        "let v = vec([1, 2, 3]); let t = 0; for x in &v { t = t + *x; } t",
        "6",
    );
    runs_to(
        "let a = [1, 2, 3]; let t = 0; for x in &a { t = t + *x; } t",
        "6",
    );
}

#[test]
fn an_operator_on_an_extension_value_is_a_call_of_its_shared_signature() {
    runs_to("let a = vec([1, 2]); a == a", "true");
    runs_to("let a = vec([1, 2]); a != a", "false");
    runs_to(
        "let a = decimal(\"1.5\".to_string()).unwrap(); \
         let b = decimal(\"2.5\".to_string()).unwrap(); a < b",
        "true",
    );
}

#[test]
fn a_call_of_the_wrong_arity_is_refused_in_every_form() {
    refused_with("abs(1, 2)", "function `abs` expects 1 arguments, got 2");
    refused_with("max(1, 2, 3)", "no `max` takes a call of type");
    refused_with("(0 - 1).abs(2)", "function `abs` expects 1 arguments, got 2");
    refused_with("[1].len(2)", "no `len` takes a call of type");
    refused_with("1 | abs(2)", "function `abs` expects 1 arguments, got 2");
    refused_with(
        "let f = |x| -> x; f(1, 2)",
        "this closure expects 1 arguments, got 2",
    );
    refused_with(
        "let f = |x| -> x; 1.f(2)",
        "this closure expects 1 arguments, got 2",
    );
    refused_with(
        "(|x| -> x)(1, 2)",
        "this closure expects 1 arguments, got 2",
    );
}

#[test]
fn a_lambda_s_parameter_is_the_type_an_earlier_argument_fixed() {
    runs_to("vec([1, 2]) | into_iter | map(|x| -> x * 10) | sum", "30");
    runs_to(
        "let s = vec([\"ab\".to_string(), \"c\".to_string()]); \
         let n = map(into_iter(s), |t| -> t.len()) | collect; n[0u64] + n[1u64]",
        "3",
    );
}

/// A receiver every candidate takes by value is read as a value, so an
/// element that moves is not taken out of its index; checked as a borrow,
/// it was, and the run read freed storage.
#[test]
fn a_receiver_every_signature_takes_by_value_is_not_moved_out_of_an_index() {
    refused_with(
        "let a = [Some(\"a\".to_string())]; a[0u64].unwrap()",
        "cannot move out of index",
    );
    runs_to("let a = [Some(1)]; a[0u64].unwrap()", "1");
}

/// RFC-0043 rule 2: admission is asked again once the variable settles;
/// RFC-0062 rule 3: a `&String` reaches `&str`. A receiver whose head is
/// open is held (rule 6), and `string::contains` lends the `&String` the
/// call passes, in either call form.
#[test]
fn a_method_on_a_lambda_parameter_reaches_a_view() {
    runs_to(
        "let f = |s| -> s.contains(\"a\"); f(&\"ab\".to_string())",
        "true",
    );
    runs_to(
        "let f = |s| -> contains(s, \"a\"); f(&\"ab\".to_string())",
        "true",
    );
}

/// RFC-0043 rule 6: a receiver whose type is still open is admitted in the
/// mode of the candidate the decision settles on; RFC-0029 rules 3-4: `Vec`'s
/// `last` takes the `&mut` the call passes as a shared reborrow, which
/// drives out `slice`'s view of it (rule 1), in either call form.
#[test]
fn a_method_on_a_mutably_lent_lambda_parameter() {
    runs_to(
        "let v = vec([1, 2]); let f = |r| -> match r.last() { Some(x) => *x, None => 0, }; f(&mut v)",
        "2",
    );
    runs_to(
        "let v = vec([1, 2]); let f = |r| -> match last(r) { Some(x) => *x, None => 0, }; f(&mut v)",
        "2",
    );
}

/// RFC-0043 rule 6: the mode is the settled candidate's, as at a known
/// receiver. `string::contains` lends the `String` a lambda parameter turns
/// out to be, so it is not moved and a second call reads it again;
/// `iterator::contains` takes an iterator by value, so a second call reads
/// a moved place.
#[test]
fn an_open_receiver_is_lent_or_moved_as_the_settled_candidate_takes_it() {
    runs_to(
        "let f = |s| -> s.contains(\"a\"); f(\"ab\".to_string())",
        "true",
    );
    runs_to(
        "let f = |s| -> { let a = s.contains(\"a\"); a && s.contains(\"b\") }; f(\"ab\".to_string())",
        "true",
    );
    let moved = "let f = |it| -> { let a = it.contains(2); a && it.contains(1) }; f(range(0, 3))";
    for opt in [Opt::None, Opt::Full] {
        match outcome(moved, opt) {
            Outcome::Refused(why) => assert!(
                why.contains("`it` is used here after it was moved"),
                "at {opt:?}: {why}"
            ),
            other => panic!("at {opt:?}, expected a refusal, got {other:?}: {moved}"),
        }
    }
}

/// RFC-0043 rule 2: a held receiver no candidate admits once the solve
/// names its head refuses the call, which names that head: no mode was
/// chosen, so the receiver is shown as the place is, as at a known head.
#[test]
fn an_open_receiver_no_candidate_admits_is_refused_at_its_settled_type() {
    refused_with(
        "let f = |s| -> s.len(); f(true)",
        "no `len` takes a call of type Fn(Bool)",
    );
    refused_with(
        "let b = true; b.len()",
        "no `len` takes a call of type Fn(Bool)",
    );
}
