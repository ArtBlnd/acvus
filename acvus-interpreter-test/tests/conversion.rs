//! The conversion signatures (RFC-0019) at the contract: `to_string` and
//! `to_int` are one name each, read through a reference, with an instance
//! per scalar whose every value converts. Text is parsed by the integer
//! type's own `from_str`, and a failure is a `Result` the script matches
//! (RFC-0038).

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

async fn text(src: &str) -> String {
    let i = Interner::new();
    run(&i, src, FxHashMap::default()).await
}

async fn int(src: &str) -> i64 {
    let i = Interner::new();
    run_script(&i, src, FxHashMap::default()).await.as_int()
}

async fn script_mode_text(src: &str) -> String {
    let i = Interner::new();
    let v = run_script_mode(&i, src, FxHashMap::default()).await;
    assert!(v.is_string(), "expected a String, got {v:?}");
    // SAFETY: the witness is String.
    unsafe { v.as_str() }.to_owned()
}

fn byte_context(i: &Interner, value: u8) -> Context {
    FxHashMap::from_iter([(i.intern("b"), typed(Ty::U8, Value::byte(value)))])
}

#[tokio::test]
async fn to_string_has_an_instance_for_every_scalar() {
    assert_eq!(text("{{ x = 42 }}{{ x.to_string() }}").await, "42");
    assert_eq!(text("{{ x = 1.5 }}{{ x.to_string() }}").await, "1.5");
    assert_eq!(text("{{ x = true }}{{ x.to_string() }}").await, "true");
    assert_eq!(text("{{ x = \"hi\" }}{{ x.to_string() }}").await, "hi");

    let i = Interner::new();
    assert_eq!(
        run(&i, "{{ @b.to_string() }}", byte_context(&i, 65)).await,
        "65"
    );
}

#[tokio::test]
async fn to_int_reads_every_converting_scalar_through_a_reference() {
    assert_eq!(int("x = 1.9; x.to_int()").await, 1);
    assert_eq!(int("x = true; x.to_int()").await, 1);
    assert_eq!(int("x = 7; x.to_int()").await, 7);

    let i = Interner::new();
    assert_eq!(
        run_script(&i, "@b.to_int()", byte_context(&i, 65))
            .await
            .as_int(),
        65
    );
}

fn from_str_src(ty: &str, input: &str) -> String {
    format!(
        r#"let r = {ty}::from_str("{input}");
           if let Ok(n) = r {{ "ok " + n.to_string() }}
           else if let Err(ParseIntError::Invalid(t)) = r {{ "invalid " + t }}
           else if let Err(ParseIntError::OutOfRange(t)) = r {{ "out of range " + t }}
           else {{ "unreachable" }}"#
    )
}

#[tokio::test]
async fn from_str_parses_text_at_the_named_width() {
    assert_eq!(script_mode_text(&from_str_src("u64", "42")).await, "ok 42");
    assert_eq!(
        script_mode_text(&from_str_src("u64", "18446744073709551615")).await,
        "ok 18446744073709551615"
    );
    assert_eq!(
        script_mode_text(&from_str_src("i8", "-128")).await,
        "ok -128"
    );
}

#[tokio::test]
async fn from_str_of_text_that_is_not_an_integer_is_invalid() {
    assert_eq!(
        script_mode_text(&from_str_src("u64", "4x")).await,
        "invalid 4x"
    );
    assert_eq!(script_mode_text(&from_str_src("u64", "")).await, "invalid ");
    assert_eq!(
        *"-1".parse::<u64>().unwrap_err().kind(),
        std::num::IntErrorKind::InvalidDigit
    );
    assert_eq!(
        script_mode_text(&from_str_src("u64", "-1")).await,
        "invalid -1"
    );
}

#[tokio::test]
async fn from_str_of_an_integer_the_width_does_not_hold_is_out_of_range() {
    assert_eq!(
        script_mode_text(&from_str_src("u64", "18446744073709551616")).await,
        "out of range 18446744073709551616"
    );
    assert_eq!(
        script_mode_text(&from_str_src("i8", "128")).await,
        "out of range 128"
    );
}

#[tokio::test]
async fn char_to_int_takes_exactly_one_char() {
    let src = |input: &str| {
        format!(
            r#"let r = char_to_int("{input}");
               if let Ok(n) = r {{ "ok " + n.to_string() }}
               else if let Err(CharError::NotOneChar(s)) = r {{ "not one char [" + s + "]" }}
               else {{ "unreachable" }}"#
        )
    };
    assert_eq!(script_mode_text(&src("A")).await, "ok 65");
    assert_eq!(script_mode_text(&src("")).await, "not one char []");
    assert_eq!(script_mode_text(&src("ab")).await, "not one char [ab]");
}

#[tokio::test]
async fn int_to_char_refuses_what_is_not_a_code_point() {
    let src = |n: i64| {
        format!(
            r#"let r = int_to_char({n});
               if let Ok(c) = r {{ "ok " + c }}
               else if let Err(CharError::NotAChar(n)) = r {{ "not a char " + n.to_string() }}
               else {{ "unreachable" }}"#
        )
    };
    assert_eq!(script_mode_text(&src(65)).await, "ok A");
    assert_eq!(script_mode_text(&src(-1)).await, "not a char -1");
    assert_eq!(script_mode_text(&src(0xD800)).await, "not a char 55296");
}
