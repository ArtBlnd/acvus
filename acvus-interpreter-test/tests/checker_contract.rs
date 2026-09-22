//! Programs whose checked type and lowered code once disagreed: each is
//! either refused by the checker or runs to the value its type says, at both
//! optimization levels.

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
fn a_call_under_a_borrow_moves_its_argument_out_of_an_index_only_where_it_may() {
    refused_with(
        "let a = [\"a\".to_string(), \"b\".to_string()]; let f = |s| -> s; let r = &f(a[0]); r.len()",
        "cannot move out of index",
    );
}

#[test]
fn an_if_without_else_is_unit() {
    runs_to("let n = 1; let v = if true { n + 1 }; v", "null");
    refused_with(
        "let v = if true { 1 }; v + 1",
        "type mismatch in `+`: Unit vs i64",
    );
}

#[test]
fn a_piped_value_is_an_argument_of_a_structural_variant() {
    refused_with(
        "let x = 5; let r = x | Shape::Circle(1); match r { Shape::Circle(n) => n, _ => 0, }",
        "Shape::Circle",
    );
}

#[test]
fn a_container_that_is_a_value_is_lent_from_a_temporary() {
    runs_to("let x = [1, 2, 3][1]; x", "2");
}

#[test]
fn a_local_closure_called_as_a_method_takes_its_receiver_as_its_parameter_does() {
    runs_to("let v = [1, 2]; let f = |x| -> x.len(); f(&v) + v.f()", "4");
}

/// A `String` copies (RFC-0018): the copy is the language's at every level.
#[test]
fn a_string_used_twice_is_copied_at_every_level() {
    runs_to("let s = \"a\".to_string(); let t = s; let u = s; t + &u", "\"aa\"");
    runs_to(
        "let s = \"abc\".to_string(); let f = |x| -> x; let a = f(s); let b = f(s); a + &b",
        "\"abcabc\"",
    );
}

#[test]
fn a_unary_operator_refuses_an_operand_it_does_not_take() {
    refused_with("let x = !5; x", "type mismatch in `!`");
    refused_with("let x = *5; x", "`*` needs a reference");
}

#[tokio::test]
async fn a_context_read_inside_a_variant_payload_has_its_slot() {
    let i = acvus_utils::Interner::new();
    let context = acvus_interpreter_test::int_context(&i, "n", 5);
    let value = acvus_interpreter_test::run_script(
        &i,
        "let x = Some(@n); x.unwrap()",
        context,
        acvus_mir::ty::Ty::I64,
    )
    .await;
    assert_eq!(value.as_int(), 5);
}

#[test]
fn an_unused_owned_element_of_a_loop_by_value_is_dropped_where_it_arrives() {
    runs_to("let a = [\"a\".to_string()]; for o in a { } 0", "0");
}

#[test]
fn a_field_of_a_loop_element_by_value_is_the_element_s() {
    runs_to("let a = [{ n: 1, }]; let t = 0; for o in a { t = o.n; } t", "1");
    runs_to("let t = 0; for o in [{ n: 1, }] { t = t + o.n; } t", "1");
}

#[test]
fn an_input_nothing_types_is_refused_where_it_is_read() {
    refused_with("$s", "nothing that reads `$s` decides its type");
    refused_with("let t = $s; let u = $s; t + &u", "cannot infer type");
}

/// Whether a lent place holds a reference is read off its settled type: a
/// lambda's parameter is still open where its body is checked.
#[test]
fn a_receiver_whose_type_settles_after_its_call_is_passed_as_what_it_is() {
    runs_to(
        "let v = vec([1]); let f = |r| -> { r.push(0); 0 }; f(&mut v); v.len()",
        "2",
    );
    runs_to(
        "let v = [1, 2]; let f = |r| -> { r[0u64] = 5; 0 }; f(&mut v); v[0u64]",
        "5",
    );
}

#[test]
fn a_name_bound_twice_in_one_pattern_is_refused() {
    refused_with("match (1, 2) { (x, x) => x, _ => 0, }", "`x` is bound twice in one pattern");
    refused_with(
        "let o = Some({ z: 1, b: 2, }); match o { Some({ z: x, b: x, }) => x, _ => 0, }",
        "`x` is bound twice in one pattern",
    );
}

/// An operator is decided where it is written: an operand whose type is
/// still open there is taken as a word, and one that settles to anything
/// else is refused rather than compared as a word.
#[test]
fn an_operator_on_an_operand_that_settles_late_is_refused() {
    refused_with(
        "let v = vec([1, 2]); let f = |r| -> r == r; f(&v)",
        "is decided where it is written",
    );
}
