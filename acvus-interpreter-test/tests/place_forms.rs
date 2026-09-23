use std::time::Duration;

use acvus_interpreter_test::corpus::{self, Outcome, Stage};
use acvus_mir::graph::optimize::Opt;

const LIMIT: Duration = Duration::from_secs(30);

type Contexts = serde_json::Map<String, serde_json::Value>;

fn outcome(source: &str, contexts: &Contexts, opt: Opt) -> Outcome {
    acvus_interpreter_test::attempt_within!(source, contexts = contexts, opt, Stage::Run, LIMIT)
        .unwrap_or_else(|lapse| panic!("at {opt:?}, {lapse:?}: {source}"))
}

fn runs_with_to(source: &str, contexts: &Contexts, value: &str) {
    for opt in [Opt::None, Opt::Full] {
        match outcome(source, contexts, opt) {
            Outcome::Value(got) => assert_eq!(got, value, "at {opt:?}: {source}"),
            other => panic!("at {opt:?}, expected {value}, got {other:?}: {source}"),
        }
    }
}

fn runs_to(source: &str, value: &str) {
    runs_with_to(source, &Contexts::new(), value);
}

fn nested_context() -> Contexts {
    serde_json::from_str(r#"{ "c": { "a": { "b": 1, "s": "x" } } }"#)
        .expect("the context is a JSON object")
}

#[test]
fn corpus_child() {
    corpus::child();
}

#[test]
fn a_field_chain_of_a_local_is_a_place() {
    let o = "let o = { a: { b: 1, s: \"x\".to_string(), }, };";
    runs_to(&format!("{o} o.a.b"), "1");
    runs_to(&format!("{o} (o.a).b + (o).a.b"), "2");
    runs_to(&format!("{o} o.a.b = 5; o.a.b"), "5");
    runs_to(&format!("{o} let r = &o.a; r.b"), "1");
    runs_to(&format!("{o} let r = &mut o.a; r.b = 7; o.a.b"), "7");
    runs_to(&format!("{o} let r = &mut (o.a).b; *r = 8; o.a.b"), "8");
    runs_to(&format!("{o} o.a.s == \"x\""), "true");
    runs_to(&format!("{o} o.a.s + \"!\""), "\"x!\"");
    runs_to(&format!("{o} match o.a {{ {{ b: 1, }} => 10, _ => 0, }}"), "10");
}

#[test]
fn a_field_chain_of_a_context_is_a_place() {
    let contexts = nested_context();
    runs_with_to("@c.a.b", &contexts, "1");
    runs_with_to("@c.a.b = 5; @c.a.b", &contexts, "5");
    runs_with_to("let r = &@c.a; r.b", &contexts, "1");
    runs_with_to("let r = &mut @c.a; r.b = 9; @c.a.b", &contexts, "9");
    runs_with_to("@c.a.s == \"x\"", &contexts, "true");
}

#[test]
fn a_field_chain_through_a_local_reference_is_what_the_reference_names() {
    let o = "let o = { a: { b: 1, s: \"x\".to_string(), }, };";
    runs_to(&format!("{o} let r = &o; r.a.b"), "1");
    runs_to(&format!("{o} let r = &mut o; r.a.b = 3; o.a.b"), "3");
    runs_to(&format!("{o} let r = &mut o; let q = &mut r.a; q.b = 4; o.a.b"), "4");
    runs_to(&format!("{o} let r = &o; let q = &r.a; q.b"), "1");
    runs_to(&format!("{o} let r = &o; r.a.s == \"x\""), "true");
    runs_to(&format!("{o} let r = &mut o.a; let q = &mut r.b; *q = 6; o.a.b"), "6");
    runs_to(
        &format!("{o} let os = [o]; for r in &mut os {{ r.a.b = 2; }} os[0].a.b"),
        "2",
    );
}

#[test]
fn a_field_chain_through_a_lambda_parameter_is_what_the_argument_names() {
    let xs = "let xs = [{ a: { b: 1, s: \"x\".to_string(), }, }, { a: { b: 2, s: \"y\".to_string(), }, }];";
    runs_to(&format!("{xs} as_iter(&xs) | fold(0, |acc, x| -> acc + x.a.b)"), "3");
    runs_to(
        &format!("{xs} as_iter(&xs) | fold(0, |acc, x| -> {{ let q = &x.a; acc + q.b }})"),
        "3",
    );
    runs_to(
        &format!("{xs} as_iter(&xs) | fold(0, |acc, x| -> acc + (x.a).s.len())"),
        "2",
    );
}

#[test]
fn a_field_chain_of_a_call_result_is_a_temporary() {
    let mk = "let mk = |n| -> { a: { b: n, s: \"x\".to_string(), }, };";
    runs_to(&format!("{mk} mk(2).a.b"), "2");
    runs_to(&format!("{mk} (mk(2)).a.b"), "2");
    runs_to(&format!("{mk} let r = &mk(2).a; r.b"), "2");
    runs_to(&format!("{mk} let r = &mut mk(2).a; r.b = 3; r.b"), "3");
    runs_to(&format!("{mk} mk(2).a.s == \"x\""), "true");
    runs_to(&format!("{mk} match mk(2).a {{ {{ b: 2, }} => 10, _ => 0, }}"), "10");
}

#[test]
fn an_element_of_an_array_local_is_a_place() {
    let xs = "let xs = [{ a: 1, c: 1, }, { a: 2, c: 2, }];";
    runs_to(&format!("{xs} xs[1].a"), "2");
    runs_to(&format!("{xs} xs[0].a = 7; xs[0].a + xs[1].a"), "9");
    runs_to(&format!("{xs} let r = &mut xs[0]; r.a = 5; xs[0].a"), "5");
    runs_to(&format!("{xs} let r = &xs[1].a; *r"), "2");
    runs_to(&format!("{xs} let r = &mut xs[1].c; *r = 9; xs[1].c"), "9");
    runs_to(&format!("{xs} let r = &mut (xs[1]).c; *r = 8; xs[1].c"), "8");
    runs_to("let ws = [1, 2]; let r = &mut ws[0]; *r = 5; ws[0] + ws[1]", "7");
    runs_to("let ss = [\"x\".to_string()]; ss[0] == \"x\"", "true");
}

#[test]
fn an_element_through_a_reference_is_what_the_reference_names() {
    let xs = "let xs = [{ a: 1, }, { a: 2, }];";
    runs_to(&format!("{xs} let r = &xs; r[1].a"), "2");
    runs_to(&format!("{xs} let r = &mut xs; r[0].a = 3; xs[0].a"), "3");
    runs_to(&format!("{xs} let r = &mut xs; let q = &mut r[1].a; *q = 4; xs[1].a"), "4");
    runs_to(&format!("{xs} let f = |r| -> r[1].a; f(&xs)"), "2");
    runs_to(
        &format!("{xs} as_iter(&[xs]) | fold(0, |acc, r| -> acc + r[1].a)"),
        "2",
    );
}

#[test]
fn an_element_of_a_temporary_is_a_value() {
    let mk = "let mk = |n| -> [n, n + 1];";
    runs_to(&format!("{mk} mk(1)[1]"), "2");
    runs_to(&format!("{mk} (mk(1))[1]"), "2");
    runs_to(&format!("{mk} let r = &mk(1)[1]; *r"), "2");
    runs_to(&format!("{mk} let r = &mut mk(1)[1]; *r = 5; *r"), "5");
    runs_to(
        "let mk = |n| -> [{ a: n, }, { a: n + 1, }]; let r = &mk(1); r[1].a",
        "2",
    );
}

#[test]
fn a_nested_element_and_field_chain_is_one_place() {
    let o = "let o = { v: [{ c: [{ d: 1, }], }], };";
    runs_to(&format!("{o} o.v[0].c[0].d"), "1");
    runs_to(&format!("{o} o.v[0].c[0].d = 4; o.v[0].c[0].d"), "4");
    runs_to(&format!("{o} let r = &mut o.v[0].c[0]; r.d = 5; o.v[0].c[0].d"), "5");
    runs_to(&format!("{o} let r = &mut o; r.v[0].c[0].d = 6; o.v[0].c[0].d"), "6");
}

/// A `String` element of a `for x in &mut v` loop is replaced whole through
/// its `&mut` (RFC-0018 rule 2), and `+` on the `String`s concatenates
/// through `core::add`'s text instances (RFC-0020).
#[test]
fn a_mutable_loop_over_strings_replaces_each_whole() {
    runs_to(
        "let v = vec([\"a\".to_string(), \"b\".to_string()]); \
         for x in &mut v { *x = x.clone() + \"!\"; } v[0u64].clone() + &v[1u64]",
        "\"a!b!\"",
    );
}
