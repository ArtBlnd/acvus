use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::Interner;

fn ctx(i: &Interner, name: &str, ty: IntTy, bits: u64) -> Context {
    [(
        i.intern(name),
        typed(Ty::Int(ty), Value::from_bits(ty, bits)),
    )]
    .into_iter()
    .collect()
}

#[tokio::test]
async fn arithmetic_runs_at_the_operands_width() {
    let i = Interner::new();
    let v = run_script(&i, "@b + 5", ctx(&i, "b", IntTy::U8, 250)).await;
    assert_eq!(IntTy::U8.read(v.bits()), 255);
    let v = run_script(
        &i,
        "@n * 3",
        ctx(&i, "n", IntTy::I32, (-7i32) as u32 as u64),
    )
    .await;
    assert_eq!(IntTy::I32.read(v.bits()), -21);
    let v = run_script(&i, "@u - 1", ctx(&i, "u", IntTy::U64, u64::MAX)).await;
    assert_eq!(IntTy::U64.read(v.bits()), u64::MAX as i128 - 1);
    let v = run_script(&i, "-@n", ctx(&i, "n", IntTy::I16, 5)).await;
    assert_eq!(IntTy::I16.read(v.bits()), -5);
}

#[tokio::test]
async fn arithmetic_past_the_width_wraps_at_the_width() {
    let i = Interner::new();
    let v = run_script(&i, "@b + 10", ctx(&i, "b", IntTy::U8, 250)).await;
    assert_eq!(IntTy::U8.read(v.bits()), 4);
    let v = run_script(&i, "@b * 2", ctx(&i, "b", IntTy::U8, 200)).await;
    assert_eq!(IntTy::U8.read(v.bits()), 144);
    let v = run_script(&i, "@b - 1", ctx(&i, "b", IntTy::U8, 0)).await;
    assert_eq!(IntTy::U8.read(v.bits()), 255);
    let v = run_script(&i, "-@n", ctx(&i, "n", IntTy::I8, 0x80)).await;
    assert_eq!(IntTy::I8.read(v.bits()), -128);
    let v = run_script(&i, "@n + 1", ctx(&i, "n", IntTy::I64, i64::MAX as u64)).await;
    assert_eq!(v.as_int(), i64::MIN);
}

#[tokio::test]
#[should_panic(expected = "attempt to divide by zero")]
async fn a_division_by_zero_panics_with_rust_s_text() {
    let i = Interner::new();
    run_script(&i, "@n / 0", ctx(&i, "n", IntTy::I64, 1)).await;
}

#[tokio::test]
#[should_panic(expected = "attempt to calculate the remainder with a divisor of zero")]
async fn a_remainder_by_zero_panics_with_rust_s_text() {
    let i = Interner::new();
    run_script(&i, "@n % 0", ctx(&i, "n", IntTy::I64, 1)).await;
}

#[tokio::test]
#[should_panic(expected = "attempt to divide with overflow")]
async fn dividing_the_minimum_by_minus_one_panics_with_rust_s_text() {
    let i = Interner::new();
    run_script(&i, "@n / -1", ctx(&i, "n", IntTy::I64, i64::MIN as u64)).await;
}

#[tokio::test]
#[should_panic(expected = "attempt to calculate the remainder with overflow")]
async fn the_remainder_of_the_minimum_by_minus_one_panics_with_rust_s_text() {
    let i = Interner::new();
    run_script(&i, "@n % -1", ctx(&i, "n", IntTy::I64, i64::MIN as u64)).await;
}

#[tokio::test]
async fn to_string_has_an_instance_for_every_width() {
    let i = Interner::new();
    let v = run_script(&i, "@b.to_string()", ctx(&i, "b", IntTy::U8, 250)).await;
    assert_eq!(unsafe { v.as_str() }, "250");
    let v = run_script(&i, "@n.to_string()", ctx(&i, "n", IntTy::I8, 0xFF)).await;
    assert_eq!(unsafe { v.as_str() }, "-1");
    let v = run_script(&i, "@u.to_string()", ctx(&i, "u", IntTy::U64, u64::MAX)).await;
    assert_eq!(unsafe { v.as_str() }, "18446744073709551615");
}

#[tokio::test]
async fn a_literal_argument_takes_the_parameter_s_width() {
    let i = Interner::new();
    let v = run_script(&i, r#"repeat_str("ab", 3)"#, Context::default()).await;
    assert_eq!(unsafe { v.as_str() }, "ababab");
    let v = run_script(
        &i,
        "xs = [1, 2, 3, 4]; xs | into_iter | take(2) | fold(0, |a, x| -> a + x)",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_int(), 3);
    let v = run_script(
        &i,
        "xs = [1, 2, 3, 4]; xs | into_iter | skip(3) | fold(0, |a, x| -> a + x)",
        Context::default(),
    )
    .await;
    assert_eq!(v.as_int(), 4);
}

#[tokio::test]
async fn a_literal_matches_at_the_source_s_width() {
    let i = Interner::new();
    let src = "if let 255 = @b { \"max\" } else { \"other\" }";
    let v = run_script_mode(&i, src, ctx(&i, "b", IntTy::U8, 255)).await;
    assert_eq!(unsafe { v.as_str() }, "max");
    let v = run_script_mode(&i, src, ctx(&i, "b", IntTy::U8, 7)).await;
    assert_eq!(unsafe { v.as_str() }, "other");
}

#[tokio::test]
async fn arithmetic_inside_a_while_wraps_as_the_arithmetic_does() {
    let i = Interner::new();
    let source = "let acc = @b; let k = 0; while k < 4 { acc = acc + @b; k = k + 1; } acc";
    let v = run_script_mode(&i, source, ctx(&i, "b", IntTy::U8, 250)).await;
    assert_eq!(IntTy::U8.read(v.bits()), 226, "250 * 5 mod 256");
}

/// Code motion on `bb8207f` hoisted `i + 1` into the loop head above the
/// `JumpIf`, so the exit iteration evaluated `255 + 1`, measured
/// 2026-09-18. It raised `integer overflow` then; it returns `0` now, and
/// the returned value is what fails here either way.
#[tokio::test]
async fn an_operation_in_a_loop_body_does_not_run_on_the_exit_iteration() {
    let i = Interner::new();
    let source = "let i = 250; while i < @n { i = i + 1; } i";
    let v = run_script_mode(&i, source, ctx(&i, "n", IntTy::U8, 255)).await;
    assert_eq!(IntTy::U8.read(v.bits()), 255);
}

#[tokio::test]
async fn a_loop_that_is_not_entered_runs_none_of_its_body() {
    let i = Interner::new();
    let source = "let i = 250; while i < @n { i = i + 1; } i";
    let v = run_script_mode(&i, source, ctx(&i, "n", IntTy::U8, 200)).await;
    assert_eq!(IntTy::U8.read(v.bits()), 250);
}
