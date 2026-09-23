//! `==`, `!=`, `eq(&a, &b)` and `clone(&a)` at an object, a tuple, an
//! array, an enum, an `Option` or a `Result` are its components', one by one
//! (RFC-0020). Each program runs at both optimization levels.

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

fn decimal(text: &str) -> String {
    format!("decimal(\"{text}\".to_string()).unwrap()")
}

#[test]
fn corpus_child() {
    corpus::child();
}

/// A word field compares by its bits, a `String` field by its bytes, and an
/// extension type's field by its own instance.
#[test]
fn a_structural_value_compares_and_clones_field_by_field() {
    runs_to("let o = { a: 1, }; let p = { a: 1, }; o == p", "true");
    runs_to(
        "let o = { a: { b: 1, }, }; let p = { a: { b: 2, }, }; o == p",
        "false",
    );
    runs_to("let t = (1, 2); t == (1, 2)", "true");
    runs_to("[1, 2] == [1, 2]", "true");
    runs_to("let o = { a: 1, }; clone(&o).a", "1");
    runs_to(
        "let o = { a: 1, s: \"x\".to_string(), }; let c = clone(&o); c.s",
        "\"x\"",
    );
    runs_to(
        "let o = { v: vec([1]), }; let c = clone(&o); c.v.push(2); o.v.len() + c.v.len()",
        "3",
    );
}

/// `==` grows neither side's field set: the side built without a field the
/// other has is refused where it is lent, since its construction never
/// wrote that field. An enum compares by tag first, whatever variants each
/// side's type names.
#[test]
fn objects_compare_at_one_field_set_and_enums_at_any_variant_set() {
    refused_with(
        "let o = { a: 1, }; let p = { a: 1, b: 2, }; o == p",
        "`o` has no `b` stored on every path that reaches here",
    );
    runs_to("A::X(1) == A::Y(2)", "false");
    runs_to("A::X(1) == A::X(1)", "true");
}

#[test]
fn a_field_set_grown_on_one_branch_is_refused_and_one_grown_by_a_store_compares() {
    refused_with(
        "let o = if true { { a: 1, } } else { { a: 1, b: 2, } }; o == o",
        "`o` has no `b` stored on every path that reaches here",
    );
    runs_to(
        "let o = { a: 1, }; o.b = 2; let p = { a: 1, b: 2, }; o == p",
        "true",
    );
}

#[test]
fn nested_objects_tuples_and_arrays_compare_at_every_depth() {
    let o = "{ a: { b: [1, 2], c: (1, \"x\".to_string()), }, }";
    runs_to(&format!("let o = {o}; let p = {o}; o == p"), "true");
    runs_to(
        &format!(
            "let o = {o}; let p = {{ a: {{ b: [1, 2], c: (1, \"y\".to_string()), }}, }}; o == p"
        ),
        "false",
    );
    runs_to(
        "[{ a: 1, }, { a: 2, }] == [{ a: 1, }, { a: 3, }]",
        "false",
    );
    runs_to("{ a: 1, } != { a: 2, }", "true");
    runs_to("{ a: 1, } != { a: 1, }", "false");
}

#[test]
fn enums_options_and_results_compare_by_tag_then_payload() {
    runs_to(
        "let a = A::X({ p: 1, }); let b = A::Y(\"s\".to_string()); a == b",
        "false",
    );
    runs_to("A::X({ p: 1, }) == A::X({ p: 1, })", "true");
    runs_to("{ e: A::X(1), } == { e: A::Y(1), }", "false");
    runs_to("Some(Some(1)) == Some(None)", "false");
    runs_to("Some(Some(1)) == Some(Some(1))", "true");
    runs_to("let x = Some(1); x == None", "false");
    runs_to("None == None", "true");
    runs_to(
        "let a = if true { Ok(1) } else { Err(\"e\".to_string()) }; \
         let b = if false { Ok(1) } else { Err(\"e\".to_string()) }; a == b",
        "false",
    );
    runs_to("Err(\"e\".to_string()) == Err(\"e\".to_string())", "true");
    runs_to("Ok(1) != Ok(2)", "true");
}

#[test]
fn an_unconstrained_payload_compares_nothing() {
    runs_to("let o = { a: None, }; o == o", "true");
    runs_to("(Some(1), None) == (Some(1), None)", "true");
}

/// `rust_decimal` compares values, not scales, so `1.0` and `1.00` are
/// equal under the registry's instance while their bits differ; the clone
/// keeps the scale.
#[test]
fn a_decimal_field_compares_by_its_instance() {
    let one = decimal("1.0");
    let one_hundredths = decimal("1.00");
    runs_to(&format!("{one} == {one_hundredths}"), "true");
    runs_to(
        &format!("{{ d: {one}, }} == {{ d: {one_hundredths}, }}"),
        "true",
    );
    runs_to(
        &format!("{{ d: {}, }} == {{ d: {}, }}", decimal("1.5"), decimal("2")),
        "false",
    );
    runs_to(
        &format!("let o = {{ d: {one_hundredths}, }}; clone(&o).d.to_string()"),
        "\"1.00\"",
    );
}

#[test]
fn a_string_field_compares_by_its_bytes() {
    runs_to(
        "{ s: \"a\".to_string(), } == { s: \"a\".to_string(), }",
        "true",
    );
    runs_to(
        "{ s: \"a\".to_string(), } == { s: \"b\".to_string(), }",
        "false",
    );
}

#[test]
fn a_float_field_compares_by_its_bits() {
    runs_to("{ a: 0.0 / 0.0, } == { a: 0.0 / 0.0, }", "true");
    runs_to(
        "let n = 0.0 / 0.0; ({ a: n, } == { a: n, }) == (n == n)",
        "true",
    );
    runs_to("{ a: 0.0, } == { a: -0.0, }", "false");
}

#[test]
fn a_clone_owns_its_vec_and_the_source_keeps_its_own() {
    runs_to(
        "let o = { v: vec([1]), }; let c = clone(&o); c.v.push(2); o.v.len() * 10 + c.v.len()",
        "12",
    );
    runs_to(
        "let t = (vec([1]), 2); let c = clone(&t); match c { (v, n) => v.len() + n, _ => 0, }",
        "3",
    );
    runs_to(
        "let o = Some({ s: \"q\".to_string(), }); let c = clone(&o); \
         match c { Some(x) => x.s, None => \"none\".to_string(), }",
        "\"q\"",
    );
    runs_to(
        "let o = { a: 1, s: \"x\".to_string(), }; clone(&o) == o",
        "true",
    );
    runs_to("let o = { a: 1, }; o.clone().a", "1");
}

#[test]
fn a_structural_vec_field_compares_through_the_vec_instance() {
    runs_to("{ v: vec([1, 2]), } == { v: vec([1, 2]), }", "true");
    runs_to("{ v: vec([1]), } == { v: vec([2]), }", "false");
}

#[test]
fn a_named_eq_compares_field_by_field() {
    runs_to("eq(&{ a: 1, }, &{ a: 1, })", "true");
    runs_to(
        "let o = (1, \"x\".to_string()); let p = (1, \"y\".to_string()); eq(&o, &p)",
        "false",
    );
}

#[test]
fn a_lambda_parameter_settled_to_a_structural_type() {
    runs_to(
        "let f = |a, b| -> a == b; f({ x: 1, }, { x: 1, })",
        "true",
    );
    runs_to(
        "let f = |a, b| -> a == b; let o = { x: 1, }; let p = { x: 2, }; f(&o, &p)",
        "false",
    );
    runs_to(
        "let f = |r| -> clone(r); let o = { x: 3, }; f(&o).x",
        "3",
    );
}

#[test]
fn the_empty_tuple_equals_itself() {
    runs_to("() == ()", "true");
}

#[test]
fn a_component_with_no_instance_is_refused() {
    refused_with(
        "{ f: |x| -> x + 1, } == { f: |x| -> x + 1, }",
        "`==` has no instance of core::eq for Fn(i64) -> i64",
    );
    refused_with(
        "{ a: 1, } < { a: 2, }",
        "`<` has no instance of core::cmp for {a: i64}",
    );
}

/// RFC-0020: an extern never receives a structural instance. `Vec`'s `eq`
/// requires `eq` at its element, and at `{a: i64}` only the registry's
/// instances answer, so the comparison is refused; an extern that compares
/// states its exact type and reads it through its projection.
#[test]
fn an_extern_requirement_at_a_structural_type_is_refused() {
    refused_with(
        "let v = vec([{ a: 1, }]); v == vec([{ a: 1, }])",
        "no instance of core::eq required by core::eq",
    );
}
