//! Loops `optimize::for_parts` writes as a `ForParts` run in place to the
//! value the source computes (RFC-0089 rule 2), each checked against a hand
//! computation and against the listing holding the form.

use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::*;
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn registries() -> Vec<acvus_extern::Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(acvus_ext::io_registry::<AcvusRuntime>());
    registries
}

fn listing(interner: &Interner, source: &str) -> String {
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse error"));
    let compiled =
        compile_source_with_externs(interner, ast, &FxHashMap::default(), registries(), Ty::I64);
    acvus_mir::printer::dump_with(interner, &compiled.modules[&compiled.entry_qref])
}

async fn run_to(source: &str, expected: i64) {
    let interner = Interner::new();
    let shown = listing(&interner, source);
    assert!(
        shown.contains(" parts ["),
        "the loop states its parts:\n{shown}"
    );
    let ran =
        run_script_with_externs(&interner, source, Context::default(), registries(), Ty::I64).await;
    assert_eq!(ran.value.as_int(), expected, "{shown}");
}

#[tokio::test]
async fn a_sum_and_a_product_run_to_their_values() {
    // s = 1 + 2 + 3 + 4 = 10, p = 1 * 2 * 3 * 4 = 24.
    run_to(
        "let v = [1, 2, 3, 4]; let s = 0; let p = 1; \
         for x in &v { s = s + *x; p = p * *x; } s * 1000 + p",
        10_024,
    )
    .await;
}

#[tokio::test]
async fn three_recurrences_run_to_their_values() {
    // a: 0, 0, 1, 3, 12. b: 0, 1, 2, 5, 26. c: 0, 2, 6, 38, 1446.
    run_to(
        "let a = 0; let b = 0; let c = 0; \
         for i in 0..4 { a = a * a + i; b = b * b + 1; c = c * c + 2; } \
         a + b * 100 + c * 10000",
        12 + 2_600 + 14_460_000,
    )
    .await;
}

#[tokio::test]
async fn a_recurrence_and_a_sum_run_to_their_values() {
    // a: 1, 1, 2, 6, 39, 1525. s = 0 + 1 + 2 + 3 + 4 = 10.
    run_to(
        "let a = 1; let s = 0; for i in 0..5 { a = a * a + i; s = s + i; } a * 100 + s",
        152_510,
    )
    .await;
}

#[tokio::test]
async fn an_anyorder_loop_with_a_sum_runs_to_its_value() {
    // Three prints, and s = 1 + 2 + 3.
    run_to(
        "let v = [1, 2, 3]; let s = 0; \
         anyorder { for x in &v { io::print(\"a\"); s = s + *x; } } s",
        6,
    )
    .await;
}

#[tokio::test]
async fn a_guarded_sum_and_a_product_run_to_their_values() {
    // s = 3 + 4 = 7, p = 24.
    run_to(
        "let v = [1, 2, 3, 4]; let s = 0; let p = 1; \
         for x in &v { if *x > 2 { s = s + *x; } p = p * *x; } s * 100 + p",
        724,
    )
    .await;
}
