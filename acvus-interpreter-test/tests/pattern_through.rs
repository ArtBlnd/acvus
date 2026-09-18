//! What a pattern whose scrutinee's head is open runs as (RFC-0024). A
//! lambda's parameter has no type of its own where the pattern is written:
//! the call gives it one, and the same body then reads its scrutinee
//! through a reference or as the value it was handed. A pattern inside a
//! lambda is written as the tag form `pattern = source { body };`, since
//! `if let` is a script-mode expression and a lambda's body is an
//! expression.
//!
//! The tag form and `if let` with no `else` are one lowering, so the value
//! side of the open head runs the same either way. A body that moves an
//! option's payload out takes the option's whole value, so that path has
//! nothing left to drop; the path where the match failed drops it.

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// The value the script left in `@out`.
async fn out_of(i: &Interner, source: &str) -> f64 {
    let ran = run_script_mode_with_externs(
        i,
        source,
        [(i.intern("out"), typed(Ty::Float, Value::float(0.0)))]
            .into_iter()
            .collect(),
        acvus_ext::std_registries(),
    )
    .await;
    ran.writes
        .into_iter()
        .find(|write| write.key == "out")
        .expect("@out was written")
        .value
        .as_float()
}

const READ_THROUGH: &str = "let f = |r| -> { Some(v) = r { @out = *v; }; 0 }; ";

#[tokio::test]
async fn a_variant_pattern_on_a_lent_parameter_reads_its_payload_in_place() {
    let i = Interner::new();
    let source = format!("{READ_THROUGH}let o = Some(1.5); f(&o)");
    assert_eq!(out_of(&i, &source).await, 1.5);
}

#[tokio::test]
async fn a_borrowed_option_is_matched_through_and_stays_usable() {
    let i = Interner::new();
    let v = run_script_mode(
        &i,
        "let o = Some([1.0, 2.0]);
         let second = if let Some(v) = &o { v[1] } else { 0.0 };
         let first = if let Some(u) = o { u[0] } else { 0.0 };
         second + first",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_float(), 3.0);
}

// -- A body that moves the payload out of a value source ---------------

const MOVE_OUT: &str = "let f = |q| -> { Some(v) = Some(1.5) { @out = v; }; 0 }; ";

#[tokio::test]
async fn a_tag_form_body_moves_the_payload_out_and_the_source_is_dropped_once() {
    let i = Interner::new();
    assert_eq!(out_of(&i, &format!("{MOVE_OUT}f(0)")).await, 1.5);
}

#[tokio::test]
async fn the_same_match_written_as_an_if_let_runs_the_same() {
    let i = Interner::new();
    let source = "let o = Some(1.5); if let Some(v) = o { @out = v; }; 0";
    assert_eq!(out_of(&i, source).await, 1.5);
}

#[tokio::test]
async fn a_tag_form_match_that_fails_leaves_the_context_as_it_was() {
    let i = Interner::new();
    let source = "let f = |q| -> { Some(v) = q { @out = v; }; 0 }; f(None)";
    assert_eq!(out_of(&i, source).await, 0.0);
}

#[tokio::test]
async fn an_assignment_in_a_tag_form_body_is_the_outer_binding() {
    // `out = v;` assigns the `out` the body's enclosing block bound, so the
    // payload is live after the match. The tag form and `if let` are one
    // lowering, so the join carries it either way.
    let i = Interner::new();
    let in_a_lambda = run_script_mode(
        &i,
        "let f = |q| -> { let out = 0.0; Some(v) = Some(1.5) { out = v; }; out }; f(0)",
        Context::default(),
    )
    .await;
    assert_eq!(in_a_lambda.as_float(), 1.5);

    let at_the_top_level = run_script_mode(
        &i,
        "let out = 0.0; Some(v) = Some(1.5) { out = v; }; out",
        Context::default(),
    )
    .await;
    assert_eq!(at_the_top_level.as_float(), 1.5);

    let as_an_if_let = run_script_mode(
        &i,
        "let out = 0.0; if let Some(v) = Some(1.5) { out = v; }; out",
        Context::default(),
    )
    .await;
    assert_eq!(as_an_if_let.as_float(), 1.5);
}

#[tokio::test]
async fn a_let_in_a_tag_form_body_ends_with_the_body() {
    // `let out = v;` introduces the body's own `out`, which shadows the
    // outer one and ends with the block.
    let i = Interner::new();
    let v = run_script_mode(
        &i,
        "let f = |q| -> { let out = 0.0; Some(v) = Some(1.5) { let out = v; }; out }; f(0)",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_float(), 0.0);
}

#[tokio::test]
async fn a_vec_moved_out_of_an_option_survives_the_match_that_moved_it() {
    let i = Interner::new();
    let v = run_script_mode(
        &i,
        "let f = |q| -> {
             let out = reverse([0]);
             Some(v) = Some(reverse([1, 2, 3])) { out = v; };
             len(&out)
         }; f(0)",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_int(), 3);
}
