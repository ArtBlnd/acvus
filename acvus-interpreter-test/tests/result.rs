use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[tokio::test]
async fn ok_and_err_are_built_and_matched() {
    let i = Interner::new();
    let src = |flag: bool| {
        format!(
            r#"let r = if {flag} {{ Ok(3) }} else {{ Err("boom") }}; if let Ok(v) = r {{ v + 1 }} else {{ -1 }}"#
        )
    };
    let v = run_script_mode(&i, &src(true), Context::default(), Ty::I64).await;
    assert_eq!(v.as_int(), 4);
    let v = run_script_mode(&i, &src(false), Context::default(), Ty::I64).await;
    assert_eq!(v.as_int(), -1);
    let v = run_script_mode(
        &i,
        r#"let r = if false { Ok(0) } else { Err("boom".to_string()) };
           if let Err(e) = r { e } else { "fine".to_string() }"#,
        Context::default(),
        Ty::String,
    )
    .await;
    assert_eq!(unsafe { v.as_str() }, "boom");
}

#[tokio::test]
async fn a_result_s_two_sides_are_typed_apart() {
    let i = Interner::new();
    let out = run(
        &i,
        r#"{{ r = Ok(7) }}{{ Ok(v) = r }}{{ v.to_string() }}{{ Err(e) = }}{{ e }}{{/}}"#,
        Context::default(),
    )
    .await;
    assert_eq!(out, "7");
}
