//! What a pattern whose scrutinee's head is open runs as (RFC-0024). A
//! lambda's parameter has no type of its own where the pattern is written:
//! the call gives it one, and the same body then reads its scrutinee
//! through a reference or as the value it was handed. A pattern inside a
//! lambda is written as the tag form `pattern = source { body };`, since
//! `if let` is a script-mode expression and a lambda's body is an
//! expression; the payload is carried out through a context, because a
//! name a tag-form body binds is that body's own (`Stmt::Bind` shadows,
//! and the script grammar has no assignment statement).
//!
//! The tag form and `if let` with no `else` are one lowering, so the value
//! side of the open head runs the same either way: a body that moves its
//! payload out leaves the source partly moved, and the source is dropped
//! once on each path out of the match.

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
         let second = if let Some(v) = &o { *get(v, 1) } else { 0.0 };
         let first = if let Some(u) = o { *get(&u, 0) } else { 0.0 };
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
async fn a_name_a_tag_form_body_binds_is_that_bodys_own() {
    // The script grammar has no assignment statement: `out = v;` is a
    // `Stmt::Bind`, and a bind in the body's scope shadows the outer `out`
    // and ends with the scope. The payload leaves through a context.
    let i = Interner::new();
    let v = run_script_mode(
        &i,
        "let f = |q| -> { out = 0.0; Some(v) = Some(1.5) { out = v; }; out }; f(0)",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_float(), 0.0);
}
