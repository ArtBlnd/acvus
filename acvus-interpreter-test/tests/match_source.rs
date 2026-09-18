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

#[tokio::test]
async fn every_arm_of_a_match_on_a_temporary_reads_the_same_value() {
    let i = Interner::new();
    let out = run(
        &i,
        "{{ None = strip_prefix(\"abc\", \"a\") }}none{{ Some(v) = }}{{ v }}{{/}}",
        Context::default(),
    )
    .await;
    assert_eq!(out, "bc");
}

#[tokio::test]
async fn a_string_payload_moved_out_of_a_temporary_result_is_the_string() {
    let i = Interner::new();
    let out = run(
        &i,
        "{{ Ok(c) = int_to_char(65) }}{{ c }}{{_}}bad{{/}}",
        Context::default(),
    )
    .await;
    assert_eq!(out, "A");
}
