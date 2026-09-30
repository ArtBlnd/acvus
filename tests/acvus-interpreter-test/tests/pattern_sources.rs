//! A pattern reads the same parts whatever it is applied to: a value with
//! no storage of its own, a place, or the place behind a reference
//! (RFC-0024). Each program runs to the value its pattern says, at both
//! optimization levels.

use std::time::Duration;

use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_mir::graph::optimize::Opt;

const LIMIT: Duration = Duration::from_secs(30);

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

fn refused_with(source: &str, words: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, opt) {
            Outcome::Refused(why) => assert!(why.contains(words), "at {opt:?}: {why}"),
            other => panic!("at {opt:?}, expected a refusal, got {other:?}: {source}"),
        }
    }
}

#[test]
fn corpus_child() {
    corpus::child();
}

#[test]
fn a_pattern_on_a_call_result_reads_the_value_it_returned() {
    runs_to(
        "let f = |x| -> (x, Some(x + 1)); match f(1) { (a, Some(b)) => a * 10 + b, _ => 0, }",
        "12",
    );
    runs_to(
        "let f = |x| -> (x, Some(x + 1)); if let (a, Some(b)) = f(1) { a * 10 + b } else { 0 }",
        "12",
    );
    runs_to(
        "let f = |x| -> Some(x); match f(3) { Some(v) => v, None => 0, }",
        "3",
    );
    runs_to(
        "let f = |x| -> Some((x, x + 1)); if let Some((a, b)) = f(3) { a * 10 + b } else { 0 }",
        "34",
    );
    runs_to(
        "let f = |x| -> Some(\"a\".to_string() + &x); match f(\"b\".to_string()) { Some(s) => s, None => \"\".to_string(), }",
        "\"ab\"",
    );
}

#[test]
fn a_pattern_on_a_local_reads_its_place() {
    runs_to(
        "let o = (2, Some(3)); match o { (a, Some(b)) => a * 10 + b, _ => 0, }",
        "23",
    );
    runs_to(
        "let o = (2, None); match o { (a, Some(b)) => a * 10 + b, (a, None) => a, _ => 0, }",
        "2",
    );
}

#[test]
fn a_pattern_on_a_borrowed_local_reads_through_the_reference() {
    runs_to(
        "let o = Some(4); match &o { Some(v) => *v, None => 0, }",
        "4",
    );
    runs_to(
        "let o = (5, Some(6)); let n = match &o { (a, Some(b)) => *a * 10 + *b, _ => 0, }; match o { (a, _) => n + a, _ => 0, }",
        "61",
    );
}

/// A member the checker reads through a reference binds that reference
/// and compares a literal through it (RFC-0024 rule 3, RFC-0029 rule 3).
#[test]
fn a_reference_member_is_read_through_at_its_own_depth() {
    runs_to(
        "let a = [5, 6]; let o = get(&a, 0u64); let r = &o; let n = match r { Some(p) => *p, None => 0, }; n + a[1]",
        "11",
    );
    runs_to(
        "let a = [5, 6]; let t = (get(&a, 1u64), 1); let n = match &t { (Some(p), k) => *p * 10 + *k, _ => 0, }; n + a[0]",
        "66",
    );
    runs_to(
        "let a = [5, 6]; let o = Some(get(&a, 0u64)); let n = match &o { Some(Some(p)) => *p, _ => 0, }; n + a[1]",
        "11",
    );
    runs_to(
        "let a = [5, 6]; let o = get(&a, 0u64); let r = &o; let n = match r { Some(5) => 1, Some(x) => 2, _ => 0, }; n + a[1]",
        "7",
    );
    runs_to("let a = [5, 6]; match get(&a, 0u64) { Some(5) => 1, _ => 0, }", "1");
    runs_to(
        "let x = 5; let o = Some(&mut x); let r = &o; let n = match r { Some(p) => *p, None => 0, }; x = 3; n + x",
        "8",
    );
}

#[test]
fn a_pattern_on_a_lambda_s_reference_parameter_reads_through_it() {
    runs_to(
        "let f = |r| -> match r { Some(v) => *v, None => 0, }; let o = Some(5); f(&o)",
        "5",
    );
    runs_to(
        "let f = |r| -> match r { (a, Some(b)) => *a * 10 + *b, _ => 0, }; let o = (7, Some(8)); f(&o)",
        "78",
    );
}

#[test]
fn a_variant_inside_a_tuple_inside_an_object_is_tested_and_bound() {
    runs_to(
        "let o = { p: (1, Some(2)), q: 3, }; match o { { p: (a, Some(b)), q, } => a * 100 + b * 10 + q, _ => 0, }",
        "123",
    );
    runs_to(
        "let o = { p: (1, None), q: 3, }; match o { { p: (a, Some(b)), q, } => a * 100 + b * 10 + q, { q, } => q, _ => 0, }",
        "3",
    );
    runs_to(
        "let f = |n| -> { p: (1, Some(n)), q: 3, }; match f(2) { { p: (a, Some(b)), q, } => a * 100 + b * 10 + q, _ => 0, }",
        "123",
    );
    runs_to(
        "let o = { p: (1, Some(2)), q: 3, }; match &o { { p: (a, Some(b)), q, } => *a * 100 + *b * 10 + *q, _ => 0, }",
        "123",
    );
}

#[test]
fn a_list_pattern_reads_its_head_and_its_tail() {
    runs_to(
        "let a = [1, 2, 3, 4]; match a { [x, .., y] => x * 10 + y, _ => 0, }",
        "14",
    );
    runs_to(
        "let a = [1, 2, 3, 4]; match a { [2, ..] => 0, [.., 3, y] => y, _ => 1, }",
        "4",
    );
    runs_to(
        "let f = |n| -> [n, n + 1, n + 2]; match f(1) { [x, .., 3] => x, _ => 0, }",
        "1",
    );
    runs_to(
        "let a = [1, 2, 3]; match &a { [x, .., y] => *x + *y, _ => 0, }",
        "4",
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

/// A `..` rest at a source still open: the length the call settles is held
/// to at least the parts, and a tail part is placed by it.
#[test]
fn a_list_pattern_with_a_rest_on_a_lambda_parameter() {
    runs_to(
        "let f = |r| -> match r { [1, x, ..] => *x, _ => 0, }; let a = [1, 5, 9, 11]; f(&a)",
        "5",
    );
    runs_to(
        "let f = |r| -> match r { [1, .., y] => *y, _ => 0, }; let a = [1, 5, 9, 11]; f(&a)",
        "11",
    );
}

/// RFC-0024 rule 2: a source the call settles to a value binds its words by
/// value.
#[test]
fn a_list_pattern_on_a_lambda_parameter_passed_by_value() {
    runs_to(
        "let f = |r| -> match r { [1, x, _z] => x, _ => 0, }; let a = [1, 5, 9]; f(a)",
        "5",
    );
}

/// A list pattern's length is checked, not tested: a source the call
/// settles to a length the pattern cannot match is refused, as a known one
/// is, and does not fall to `_`.
#[test]
fn a_list_pattern_on_a_lambda_parameter_of_another_length_is_refused() {
    refused_with(
        "let f = |r| -> match r { [1, x, _z] => *x, _ => 0, }; let a = [1, 5]; f(&a)",
        "array pattern needs length 3, got 2",
    );
    refused_with(
        "let f = |r| -> match r { [1, x, y, ..] => *x, _ => 0, }; let a = [1, 5]; f(&a)",
        "array pattern needs length at least 3, got 2",
    );
}

/// RFC-0024 rule 5: a head nothing settles reads the value, and nothing
/// settles its length either; a length has no least element to take, so
/// the pattern is refused.
#[test]
fn a_list_pattern_on_a_source_nothing_settles_is_refused() {
    refused_with(
        "let f = |r| -> match r { [1, x, _z] => 1, _ => 0, }; 3",
        "array length is not known here",
    );
    refused_with(
        "let f = |r| -> match r { [1, x, ..] => 1, _ => 0, }; 3",
        "array length is not known here",
    );
}

#[test]
fn a_string_part_read_by_a_pattern_is_copied_and_the_place_keeps_it() {
    runs_to(
        "let e = A::X(\"s\".to_string()); let a = match e { A::X(t) => t, }; let b = match e { A::X(t) => t, }; a + &b",
        "\"ss\"",
    );
    runs_to(
        "let e = A::X(\"s\".to_string()); let a = match e { A::X(t) => t, }; let b = match e { A::X(t) => t, }; let c = match e { A::X(t) => t, }; a + &b + &c",
        "\"sss\"",
    );
    runs_to(
        "let o = Ok(\"o\".to_string()); let a = match o { Ok(t) => t, Err(e) => e, }; let b = match o { Ok(t) => t, Err(e) => e, }; a + &b",
        "\"oo\"",
    );
    runs_to(
        "let o = if true { Err(\"e\".to_string()) } else { Ok(\"o\".to_string()) }; let a = match o { Ok(t) => t, Err(e) => e, }; let b = match o { Ok(t) => t, Err(e) => e, }; a + &b",
        "\"ee\"",
    );
    runs_to(
        "let p = { s: \"a\".to_string(), n: 1, }; let a = match p { { s, n, } => s, _ => \"z\".to_string(), }; let b = match p { { s, n, } => s, _ => \"z\".to_string(), }; a + &b",
        "\"aa\"",
    );
    runs_to(
        "let p = (\"a\".to_string(), 1); let a = match p { (x, _) => x, _ => \"z\".to_string(), }; let b = match p { (x, _) => x, _ => \"z\".to_string(), }; a + &b",
        "\"aa\"",
    );
    runs_to(
        "let e = A::X({ s: \"q\".to_string(), n: 1, }); let a = match e { A::X({ s, n, }) => s, }; let b = match e { A::X({ s, n, }) => s, }; a + &b",
        "\"qq\"",
    );
}

#[test]
fn a_place_whose_string_part_a_pattern_read_is_then_used_whole() {
    runs_to(
        "let e = A::X(\"s\".to_string()); let a = match e { A::X(t) => t, }; let g = |x| -> match x { A::X(t) => t, }; a + &g(e)",
        "\"ss\"",
    );
    runs_to(
        "let o = Ok(\"s\".to_string()); let a = match o { Ok(t) => t, Err(e) => e, }; a + &o.unwrap()",
        "\"ss\"",
    );
    runs_to(
        "let e = A::X(\"s\".to_string()); let a = match e { A::X(t) => t, }; let v = [e]; a + &v.len().to_string()",
        "\"s1\"",
    );
}

#[test]
fn a_literal_arm_compares_the_value_it_is_applied_to() {
    runs_to(
        "let f = |x| -> x + 1; match f(1) { 1 => 10, 2 => 20, _ => 0, }",
        "20",
    );
    runs_to("let n = 2; match n { 1 => 10, 2 => 20, _ => 0, }", "20");
    runs_to(
        "let n = 2; let f = |r| -> match r { 1 => 10, 2 => 20, _ => 0, }; f(&n)",
        "20",
    );
    runs_to(
        "match \"b\".to_string() { \"a\" => 1, \"b\" => 2, _ => 0, }",
        "2",
    );
    runs_to(
        "let f = |x| -> (x, 2); match f(1) { (1, 3) => 13, (1, n) => n, _ => 0, }",
        "2",
    );
}

#[tokio::test]
async fn a_context_bind_in_an_if_let_stores_the_part_into_the_context() {
    let i = acvus_utils::Interner::new();
    let from_a_call = acvus_interpreter_test::run_script(
        &i,
        "let f = |x| -> Some(x); if let Some(@n) = f(9) { }; @n",
        acvus_interpreter_test::int_context(&i, "n", 5),
        acvus_mir::ty::Ty::I64,
    )
    .await;
    assert_eq!(from_a_call.as_int(), 9);

    let from_a_local = acvus_interpreter_test::run_script(
        &i,
        "let o = (1, Some(8)); if let (_, Some(@n)) = o { }; @n",
        acvus_interpreter_test::int_context(&i, "n", 5),
        acvus_mir::ty::Ty::I64,
    )
    .await;
    assert_eq!(from_a_local.as_int(), 8);

    let unmatched = acvus_interpreter_test::run_script(
        &i,
        "let o = None; if let Some(@n) = o { }; @n",
        acvus_interpreter_test::int_context(&i, "n", 5),
        acvus_mir::ty::Ty::I64,
    )
    .await;
    assert_eq!(unmatched.as_int(), 5);
}
