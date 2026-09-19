//! A declaration written `-> S` writes its components where the caller
//! placed the result: the registers the run allocation gave it, or the flat
//! body of the heap object it is realized into (RFC-0050 rules 5 and 6).

use acvus_extern::{Registry, TyArg, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter::listing::ops_of_anywhere;
use acvus_interpreter_test::listing::script_listing_with_externs;
use acvus_interpreter_test::{Context, run_script_with_externs};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[derive(TyArg)]
pub struct Point {
    x: i64,
    y: i64,
}

#[derive(TyArg)]
pub struct Tagged {
    label: String,
    n: i64,
}

#[extern_fn(effect = pure)]
fn origin() -> Point {
    Point { x: 0, y: 0 }
}

#[extern_fn(effect = pure)]
fn shifted(a: i64) -> Point {
    Point { x: a, y: a + 1 }
}

#[extern_fn(effect = pure)]
fn point_of(a: i64, b: i64) -> Point {
    Point { x: a, y: b }
}

#[extern_fn(effect = pure)]
fn summed(a: i64, b: i64, c: i64) -> Point {
    Point { x: a + b, y: c }
}

#[extern_fn(effect = pure)]
fn point_window(a: i64, b: i64, c: i64, d: i64, e: i64) -> Point {
    Point {
        x: a + b + c,
        y: d + e,
    }
}

#[extern_fn(effect = pure)]
fn tagged(label: String, n: i64) -> Tagged {
    Tagged { label, n }
}

#[extern_fn(effect = pure)]
fn label_len(t: Tagged) -> i64 {
    t.label.len() as i64
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        types: [],
        fns: [
            origin,
            shifted,
            point_of,
            summed,
            point_window,
            tagged,
            label_len,
        ],
    }
}

async fn answer(source: &str) -> i64 {
    let interner = Interner::new();
    run_script_with_externs(
        &interner,
        source,
        Context::default(),
        vec![registry(), acvus_ext::conversion_registry()],
        Ty::I64,
    )
    .await
    .value
    .as_int()
}

fn one_extern_operation(source: &str) -> String {
    let interner = Interner::new();
    let blocks = script_listing_with_externs(
        &interner,
        source,
        Context::default(),
        vec![registry(), acvus_ext::conversion_registry()],
        Ty::I64,
    );
    let ops = ops_of_anywhere(&blocks);
    let called: Vec<&String> = ops
        .iter()
        .filter(|op| op.contains("__extern_fn_"))
        .collect();
    assert_eq!(
        called.len(),
        1,
        "{source} prepares one extern call, and these are {called:?}"
    );
    called[0].clone()
}

#[tokio::test]
async fn every_arity_returns_the_struct_its_body_built() {
    assert_eq!(answer("let p = origin(); p.x + p.y").await, 0);
    assert_eq!(answer("let p = shifted(4); p.x * 10 + p.y").await, 45);
    assert_eq!(answer("let p = point_of(4, 2); p.x * 10 + p.y").await, 42);
    assert_eq!(answer("let p = summed(1, 3, 2); p.x * 10 + p.y").await, 42);
    assert_eq!(
        answer("let p = point_window(1, 1, 2, 1, 1); p.x * 10 + p.y").await,
        42
    );
}

/// The value alone does not say which form ran. This is the form itself,
/// read off the prepared operations.
#[test]
fn each_arity_takes_its_own_operation() {
    let cases = [
        ("let p = origin(); p.x", "CallRun0"),
        ("let p = shifted(4); p.x", "CallRun1"),
        ("let p = point_of(4, 2); p.x", "CallRun2"),
        ("let p = summed(1, 3, 2); p.x", "CallRun3"),
        ("let p = point_window(1, 1, 2, 1, 1); p.x", "CallRunWindow"),
    ];
    for (source, form) in cases {
        let op = one_extern_operation(source);
        assert!(
            op.starts_with(form),
            "{source} takes {form}, and the operation prepared for it is {op}"
        );
    }
}

#[tokio::test]
async fn a_large_component_is_owned_by_the_run_it_lands_in() {
    assert_eq!(
        answer("let t = tagged(\"abcd\".to_string(), 7); t.n").await,
        7
    );
    assert_eq!(
        answer("let i = 0; let k = 0; while i < 8 { let t = tagged(\"abcd\".to_string(), 1); k = k + t.n; i = i + 1; } k")
            .await,
        8
    );
}

#[tokio::test]
async fn an_escaping_result_is_realized_as_a_flat_heap_object() {
    assert_eq!(
        answer("label_len(tagged(\"abcd\".to_string(), 7))").await,
        4
    );
}
