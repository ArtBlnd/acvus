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

/// A refusal that is the program's only one, at both levels. The program
/// is the body `main`, which names the refusal.
fn refused_only_with(source: &str, refusal: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, opt) {
            Outcome::Refused(why) => {
                assert_eq!(why, format!("[main] {refusal}"), "at {opt:?}: {source}")
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

/// An operator on an operand still open where it is checked leaves its
/// instance to the solve: an operand that never settles, or settles at a
/// type with no instance, is refused.
#[test]
fn an_operator_on_an_open_operand_is_refused_where_no_instance_settles() {
    refused_with("let g = |a, b| -> a < b; 1", "");
    refused_with("let g = |a| -> a + a; 0", "");
    refused_with("let f = |a| -> -a; f(true)", "");
    refused_with("let f = |a| -> -a; f(\"s\".to_string())", "");
}

/// A field reached through a lambda parameter whose type the call settles
/// is refused as it is under a head known at the read: a store needs a
/// `&mut`, a value read of a non-word needs no reference (RFC-0018 rules 4
/// and 8).
#[test]
fn a_field_under_a_head_the_call_settles_is_checked_as_under_a_known_one() {
    refused_with(
        "let o = { a: 1, }; let f = |r| -> { r.a = 2; 0 }; f(&o); o.a",
        "cannot store through",
    );
    refused_with(
        "let o = { v: vec([1]), }; let f = |r| -> r.v; f(&o).len()",
        "cannot move",
    );
}

/// RFC-0018 rule 1: an `Option` copies exactly when its payload does, so
/// an `Option` of a word is a word to `*` (rule 4) and an index (RFC-0047
/// rule 5).
#[test]
fn an_option_of_a_word_is_read_through_a_reference() {
    runs_to("let o = Some(2); let r = &o; match *r { Some(x) => x, None => 0, }", "2");
}

/// RFC-0018 rule 4: `*r` yields a word and nothing else. An `Option` of a
/// word, at any depth, is one; an `Option` of a `String`, an object and a
/// tuple are not.
#[test]
fn deref_reads_a_word_and_refuses_every_other_type() {
    runs_to(
        "let o = Some(Some(true)); let r = &o; match *r { Some(Some(b)) => b, _ => false, }",
        "true",
    );
    refused_with(
        "let o = Some(\"a\".to_string()); let r = &o; match *r { Some(s) => s, None => \"\".to_string(), }",
        "`*` reads only a word",
    );
    refused_with(
        "let o = { a: 1, }; let r = &o; let c = *r; c.a",
        "`*` reads only a word",
    );
    refused_with(
        "let t = (1, 2); let r = &t; match *r { (a, b) => a + b, _ => 0, }",
        "`*` reads only a word",
    );
}

/// RFC-0047 rule 5: a place path under an element read by value copies out
/// of the element's reference, so the path must end at a type that copies.
#[test]
fn a_field_of_an_element_is_read_by_value_only_where_it_copies() {
    runs_to(
        "let xs = [{ k: Some(3), }]; let y = xs[0].k; y.unwrap_or(0)",
        "3",
    );
    refused_with(
        "let xs = [{ k: { m: 1, }, }]; let y = xs[0].k; y.m",
        "cannot move",
    );
}

/// A `for` over `&xs` binds each element's reference, and `*x` reads the
/// element only where it is a word (RFC-0018 rule 4).
#[test]
fn deref_of_a_loop_element_reference_reads_only_a_word() {
    runs_to(
        "let xs = [Some(1), Some(2), None]; let s = 0; for x in &xs { s = s + (*x).unwrap_or(10); } s",
        "13",
    );
    refused_with(
        "let xs = [{ a: 1, }]; let r = { a: 0, }; for x in &xs { r = *x; } r.a",
        "`*` reads only a word",
    );
}

const VIEW_RETURNED: &str =
    "a body does not return a reference; write `.to_string()` for the owned text";
const VIEW_IN_AGGREGATE: &str = "a reference cannot be stored in a list, object, or tuple; \
     write `.to_string()` to store the text";
const VIEW_IN_PAYLOAD: &str = "a reference cannot be stored in an Option or a Result; \
     write `.to_string()` to store the text";

/// RFC-0062 rule 6: a lambda's result crosses as one value, which a view
/// is not, whether the result's type was known at the lambda or settled by
/// a later call.
#[test]
fn a_lambda_returning_a_view_is_refused_at_a_known_and_a_settled_type() {
    refused_only_with("let f = |k| -> \"ab\"; f(1).len()", VIEW_RETURNED);
    refused_only_with("let f = |s| -> s; f(\"ab\").len()", VIEW_RETURNED);
    refused_only_with(
        "let f = |s| -> { let y = s; y }; f(\"abc\").len()",
        VIEW_RETURNED,
    );
    refused_only_with(
        "let f = |s| -> s; let v = vec([1, 2]); f(v.as_slice()).len()",
        "a body does not return a reference",
    );
}

/// RFC-0062 rule 5: a list element holds no view, whether its type was
/// known at the list or settled by a later call.
#[test]
fn a_view_stored_in_a_list_is_refused_at_a_known_and_a_settled_type() {
    refused_only_with("let a = [\"ab\", \"cd\"]; a.len()", VIEW_IN_AGGREGATE);
    refused_only_with(
        "let f = |s| -> [s, s]; let t = \"hello\".to_string(); let a = f(t.trim()); \
         a[0u64].len() + a[1u64].len()",
        VIEW_IN_AGGREGATE,
    );
    refused_only_with(
        "let f = |s| -> vec([s]); let o = f(\"hello\"); o[0u64].len()",
        VIEW_IN_AGGREGATE,
    );
}

/// RFC-0062 rule 5: an object or tuple field holds no view, whether its
/// type was known at the construction or the store, or settled by a later
/// call.
#[test]
fn a_view_stored_in_a_field_is_refused_at_a_known_and_a_settled_type() {
    refused_only_with("let o = { k: \"ab\", n: 1, }; o.n", VIEW_IN_AGGREGATE);
    refused_only_with(
        "let f = |s| -> { k: s, n: 1, }; let o = f(\"hello\"); o.k.len() + o.n",
        VIEW_IN_AGGREGATE,
    );
    refused_only_with("let o = { n: 1, }; o.k = \"ab\"; o.n", VIEW_IN_AGGREGATE);
    refused_only_with(
        "let f = |s| -> { let o = { n: 1, }; o.k = s; o.n }; f(\"ab\")",
        VIEW_IN_AGGREGATE,
    );
    refused_only_with("let t = (\"ab\", 1); 1", VIEW_IN_AGGREGATE);
    refused_only_with(
        "let f = |s| -> (s, 1); let t = f(\"ab\"); 1",
        VIEW_IN_AGGREGATE,
    );
}

/// RFC-0062 rule 5: a variant payload holds no view, whether its type was
/// known at the construction or settled by a later call.
#[test]
fn a_view_stored_in_a_payload_is_refused_at_a_known_and_a_settled_type() {
    refused_only_with("let x = Some(\"ab\"); 1", VIEW_IN_PAYLOAD);
    refused_only_with(
        "let f = |s| -> Some(s); let x = f(\"ab\"); 1",
        VIEW_IN_PAYLOAD,
    );
}

/// RFC-0064 rule 5: a structural enum's payload is data like any other, in
/// either construction form, at a known and a settled type.
#[test]
fn a_view_stored_in_an_enum_payload_is_refused_at_a_known_and_a_settled_type() {
    const VIEW_IN_VARIANT: &str =
        "a reference cannot be stored in an enum's payload; write `.to_string()` to store the text";
    refused_only_with(
        "let e = E::B(\"ab\"); match e { E::B(x) => x.len(), }",
        VIEW_IN_VARIANT,
    );
    refused_only_with(
        "let f = |s| -> E::B(s); match f(\"ab\") { E::B(x) => x.len(), }",
        VIEW_IN_VARIANT,
    );
}

/// RFC-0064 rule 5: a capture is one word, so a lambda captures no view,
/// whether the captured name's type was known at the lambda or settled by
/// a later call.
#[test]
fn a_captured_view_is_refused_at_a_known_and_a_settled_type() {
    const VIEW_CAPTURED: &str =
        "a lambda cannot capture a string or slice view; write `.to_string()` for the owned text";
    refused_only_with(
        "let s = \"ab\"; let f = |k| -> s.len() + k; f(1)",
        VIEW_CAPTURED,
    );
    refused_only_with(
        "let f = |s| -> { let g = |k| -> s.len() + k; g(1) }; f(\"ab\")",
        VIEW_CAPTURED,
    );
}

/// The control: a view that flows only into a parameter that takes a view
/// is admitted wherever the parameter's type settles, and a `String` is
/// stored where a view is not.
#[test]
fn a_view_that_flows_only_to_a_view_parameter_runs() {
    runs_to("let f = |s| -> s.len(); f(\"ab\")", "2");
    runs_to(
        "let f = |s| -> s.len(); let t = \" abc \".to_string(); f(t.trim())",
        "3",
    );
    runs_to(
        "let f = |s| -> [s, s]; let a = f(\"ab\".to_string()); a[0u64].len() + a[1u64].len()",
        "4",
    );
    runs_to(
        "let o = { n: 1, }; o.k = \"ab\".to_string(); o.n = 3; o.k.len() + o.n",
        "5",
    );
}
