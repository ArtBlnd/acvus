use acvus_interpreter_test::*;
use acvus_utils::Interner;

#[tokio::test]
async fn a_nested_pattern_on_a_temporary_takes_it_once() {
    let i = Interner::new();
    let out = run(
        &i,
        "% match Some(Some(3))\n\
         % Some(Some(v)) =>\n\
         {{ v.to_string() }}\n\
         % _ =>\n\
         none\n\
         % end",
        Context::default(),
    )
    .await;
    assert_eq!(out, "3\n");
}

#[tokio::test]
async fn a_nested_pattern_on_a_bound_value() {
    let i = Interner::new();
    let out = run(
        &i,
        "% let r = Ok(Some(3))\n\
         % match r\n\
         % Ok(Some(v)) =>\n\
         {{ v.to_string() }}\n\
         % _ =>\n\
         none\n\
         % end",
        Context::default(),
    )
    .await;
    assert_eq!(out, "3\n");
}

#[tokio::test]
async fn every_arm_of_a_match_on_a_temporary_reads_the_same_value() {
    let i = Interner::new();
    let out = run(
        &i,
        "% match strip_prefix(\"abc\", \"a\")\n\
         % None =>\n\
         none\n\
         % Some(v) =>\n\
         {{ v }}\n\
         % end",
        Context::default(),
    )
    .await;
    assert_eq!(out, "bc\n");
}

#[tokio::test]
async fn a_string_payload_moved_out_of_a_temporary_result_is_the_string() {
    let i = Interner::new();
    let out = run(
        &i,
        "% match int_to_char(65)\n\
         % Ok(c) =>\n\
         {{ c.to_string() }}\n\
         % _ =>\n\
         bad\n\
         % end",
        Context::default(),
    )
    .await;
    assert_eq!(out, "A\n");
}
