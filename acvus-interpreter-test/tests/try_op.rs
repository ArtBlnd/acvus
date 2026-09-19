use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

fn flags(i: &Interner, values: &[(&str, bool)]) -> Context {
    values
        .iter()
        .map(|(name, b)| (i.intern(name), typed(Ty::Bool, Value::bool_(*b))))
        .collect()
}

fn text(v: Value) -> String {
    unsafe { v.as_str() }.to_owned()
}

#[tokio::test]
async fn a_question_mark_takes_the_ok_and_returns_the_err() {
    let i = Interner::new();
    let src = r#"let r = if @ok { Ok(2) } else { Err("bad".to_string()) };
                 let plus_one = |r| -> Ok(r? + 1);
                 if let Ok(v) = plus_one(r) { v } else { -1 }"#;
    let v = run_script_mode(&i, src, flags(&i, &[("ok", true)]), Ty::I64).await;
    assert_eq!(v.as_int(), 3);
    let v = run_script_mode(&i, src, flags(&i, &[("ok", false)]), Ty::I64).await;
    assert_eq!(v.as_int(), -1);
}

#[tokio::test]
async fn the_error_enum_of_a_function_is_the_union_of_what_it_tried() {
    let i = Interner::new();
    let src = r#"let a = if @a_ok { Ok(1) } else { Err(Fail::TooBig(12)) };
                 let b = if @b_ok { Ok(2) } else { Err(Fail::Odd) };
                 let both = |a, b| -> Ok(a? + b?);
                 let r = both(a, b);
                 if let Ok(v) = r { "sum".to_string() }
                 else if let Err(Fail::TooBig(n)) = r { "big".to_string() }
                 else { "odd".to_string() }"#;
    let v = run_script_mode(
        &i,
        src,
        flags(&i, &[("a_ok", true), ("b_ok", true)]),
        Ty::String,
    )
    .await;
    assert_eq!(text(v), "sum");
    let v = run_script_mode(
        &i,
        src,
        flags(&i, &[("a_ok", false), ("b_ok", true)]),
        Ty::String,
    )
    .await;
    assert_eq!(text(v), "big");
    let v = run_script_mode(
        &i,
        src,
        flags(&i, &[("a_ok", true), ("b_ok", false)]),
        Ty::String,
    )
    .await;
    assert_eq!(text(v), "odd");
}

#[tokio::test]
async fn a_question_mark_on_an_option_returns_none() {
    let i = Interner::new();
    let src = r#"let o = if @some { Some(4) } else { None }; let v = o?; Some(v + 1)"#;
    let v = run_script_mode(
        &i,
        src,
        flags(&i, &[("some", true)]),
        Ty::Option(Box::new(Ty::I64)),
    )
    .await;
    assert_eq!(v.as_int(), 5);
    let v = run_script_mode(
        &i,
        src,
        flags(&i, &[("some", false)]),
        Ty::Option(Box::new(Ty::I64)),
    )
    .await;
    assert!(v.is_none(), "{v:?}");
}

#[tokio::test]
async fn a_result_nothing_fails_into_has_the_never_error() {
    let i = Interner::new();
    let v = run_script_mode(
        &i,
        "let r = Ok(3); if let Ok(v) = r { v } else { 0 }",
        Context::default(),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 3);
}

#[tokio::test]
async fn a_script_returns_early_through_a_question_mark() {
    let i = Interner::new();
    let src = r#"let r = if @ok { Ok(3) } else { Err("bad".to_string()) }; let v = r?; Ok(v * 10)"#;
    let v = run_script_mode(
        &i,
        src,
        flags(&i, &[("ok", true)]),
        Ty::Result(Box::new(Ty::I64), Box::new(Ty::String)),
    )
    .await;
    assert_eq!(
        unsafe { v.as_result() }.as_ref().ok().map(|v| v.as_int()),
        Some(30)
    );
    let v = run_script_mode(
        &i,
        src,
        flags(&i, &[("ok", false)]),
        Ty::Result(Box::new(Ty::I64), Box::new(Ty::String)),
    )
    .await;
    assert_eq!(
        unsafe { v.as_result() }.as_ref().err().map(|e| text_ref(e)),
        Some("bad".to_owned())
    );
}

fn text_ref(v: &Value) -> String {
    unsafe { v.as_str() }.to_owned()
}
