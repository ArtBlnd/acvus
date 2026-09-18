//! `if/else` as one operation: what each shape computes, and what the
//! recognizer collapses it into.
//!
//! `short_circuit.rs` holds the other half of the untaken-arm property,
//! where the arm that must not run is an extern call rather than a
//! division.

use std::sync::Arc;

use acvus_interpreter::code::{Code, Diamond, Payload, payload_name};
use acvus_interpreter::{PrepareCtx, prepare_module};
use acvus_interpreter_test::*;
use acvus_utils::Interner;

#[derive(Debug, PartialEq)]
struct DiamondShape {
    on_true_ops: usize,
    on_false_ops: usize,
    join_moves: usize,
}

fn prepared_payloads(source: &str) -> Vec<String> {
    let interner = Interner::new();
    let (context_types, _snapshot) = split_context(&interner, Context::default());
    let cr = compile_script_mode(&interner, source, &context_types);
    let ctx = PrepareCtx {
        interner: &interner,
        externs: &cr.extern_executables,
        context_names: &cr.context_names,
    };
    let module = cr.modules.get(&cr.entry_qref).expect("the entry module");
    let prepared = Arc::new(prepare_module(module, &ctx));
    let Code::Body(main) = &*prepared.main else {
        panic!("the entry module's main is a body, not a one-chain expression")
    };
    main.payloads
        .iter()
        .map(|payload| match payload {
            Payload::Diamond(Diamond { on_true, on_false }) => {
                assert_eq!(
                    on_true.join.len(),
                    on_false.join.len(),
                    "both arms of a diamond feed the same join parameters"
                );
                format!(
                    "{:?}",
                    DiamondShape {
                        on_true_ops: on_true.block.iter().count(),
                        on_false_ops: on_false.block.iter().count(),
                        join_moves: on_true.join.len(),
                    }
                )
            }
            other => payload_name(other).to_string(),
        })
        .collect()
}

fn diamonds_innermost_first(source: &str) -> Vec<String> {
    prepared_payloads(source)
        .into_iter()
        .filter(|name| name.starts_with("DiamondShape"))
        .collect()
}

fn loop_count(source: &str) -> usize {
    prepared_payloads(source)
        .iter()
        .filter(|name| *name == "Loop")
        .count()
}

fn shape(on_true_ops: usize, on_false_ops: usize, join_moves: usize) -> String {
    format!(
        "{:?}",
        DiamondShape {
            on_true_ops,
            on_false_ops,
            join_moves,
        }
    )
}

#[tokio::test]
async fn an_if_expression_yields_the_taken_arms_value() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let n = 7; let d = if n % 2 == 0 { n / 2 } else { n * 3 + 1 }; d",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_int(), 22);
}

#[tokio::test]
async fn an_if_statement_without_a_value_runs_only_the_taken_arm() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let acc = 0; let n = 0; while n < 6 { if n % 2 == 0 { acc = acc + n; }; n = n + 1; } acc",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_int(), 6);
}

#[tokio::test]
async fn an_if_nested_in_an_arm_is_a_diamond_inside_a_diamond() {
    let i = Interner::new();
    let source = "let n = 9; let r = if n > 5 { if n > 8 { 100 } else { 10 } } else { 1 }; r";
    let v = run_script(&i, source, Context::default()).await;
    assert_eq!(v.as_int(), 100);
    assert_eq!(
        diamonds_innermost_first(source),
        vec![shape(1, 1, 0), shape(2, 1, 0)],
        "the outer arm holds the inner test's compare and then the inner diamond"
    );
}

#[tokio::test]
async fn an_untaken_arm_that_would_panic_never_runs() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let z = 0; let r = if z == 0 { 5 } else { 1 / z }; r",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_int(), 5);
}

#[tokio::test]
async fn the_dividing_arm_runs_when_it_is_the_one_taken() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "let z = 2; let r = if z == 0 { 5 } else { 10 / z }; r",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_int(), 5);
}

#[tokio::test]
async fn a_diamond_in_a_body_leaves_the_while_recognizable() {
    let source = "let acc = 0; let n = 0; while n < 8 { if n % 3 == 0 { acc = acc + 1; } else if n % 3 == 1 { acc = acc + 10; } else { acc = acc + 100; }; n = n + 1; } acc";
    let i = Interner::new();
    let v = run_script(&i, source, Context::default()).await;
    assert_eq!(v.as_int(), 3 + 30 + 200);
    assert_eq!(loop_count(source), 1);
    assert_eq!(diamonds_innermost_first(source).len(), 2);
}

#[tokio::test]
async fn a_while_in_an_arm_is_one_operation_inside_the_diamond() {
    let source = "let n = 5; let acc = 0; if n > 3 { let k = 0; while k < n { acc = acc + k; k = k + 1; } } else { acc = 1; }; acc";
    let i = Interner::new();
    let v = run_script(&i, source, Context::default()).await;
    assert_eq!(v.as_int(), 10);
    assert_eq!(loop_count(source), 1);
    assert_eq!(diamonds_innermost_first(source).len(), 1);
}

#[tokio::test]
async fn a_short_circuit_condition_leaves_the_while_recognizable() {
    let source =
        "let n = 0; let acc = 0; while n < 10 && acc < 12 { acc = acc + n; n = n + 1; } acc";
    let i = Interner::new();
    let v = run_script(&i, source, Context::default()).await;
    assert_eq!(v.as_int(), 15);
    assert_eq!(loop_count(source), 1);
    assert_eq!(diamonds_innermost_first(source), vec![shape(1, 1, 0)]);
}

#[tokio::test]
async fn an_or_condition_is_the_same_diamond() {
    let source = "let n = 0; let acc = 0; while n < 3 || acc < 20 { acc = acc + n; n = n + 1; } n";
    let i = Interner::new();
    let v = run_script(&i, source, Context::default()).await;
    assert_eq!(v.as_int(), 7);
    assert_eq!(loop_count(source), 1);
    assert_eq!(diamonds_innermost_first(source).len(), 1);
}
