use acvus_interpreter_test::*;
use acvus_utils::Interner;

#[tokio::test]
async fn a_nested_pattern_on_a_temporary_takes_it_once() {
    let i = Interner::new();
    let out = run(
        &i,
        "{{ Some(Some(v)) = Some(Some(3)) }}{{ v.to_string() }}{{_}}none{{/}}",
        Context::default(),
    )
    .await;
    assert_eq!(out, "3");
}

#[tokio::test]
async fn a_nested_pattern_on_a_bound_value() {
    let i = Interner::new();
    let out = run(
        &i,
        "{{ r = Ok(Some(3)) }}{{ Ok(Some(v)) = r }}{{ v.to_string() }}{{_}}none{{/}}",
        Context::default(),
    )
    .await;
    assert_eq!(out, "3");
}
