//! What a pattern whose scrutinee's head is open runs as (RFC-0024). A
//! lambda's parameter has no type of its own where the pattern is written:
//! the call gives it one, and the same body then reads its scrutinee
//! through a reference or as the value it was handed. A pattern inside a
//! lambda is written as the tag form `pattern = source { body };`, since
//! `if let` is a script-mode expression and a lambda's body is an
//! expression; the payload is carried out through a context, because a
//! variable assigned inside a tag-form body is not live after it. The
//! value side of the open head has no test here: a tag-form match that
//! moves its payload out inside a lambda traps at run time whatever the
//! head is -- `|q| -> { Some(v) = Some(1.5) { @out = v; }; 0 }`, a head
//! the checker reads at once, traps the same way.

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
