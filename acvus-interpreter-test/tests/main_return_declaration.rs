//! The host declares what `main` returns, and the compilation holds the body
//! to it (RFC-0054).

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

/// A body with two `Return`s: the `?` leaves early with the `Err`, the tail
/// leaves with the `Ok` payload.
const TWO_RETURNS: &str = r#"let r = if @ok { Ok(7) } else { Err("bad") };
                             let v = r?;
                             Ok(v + 1)"#;

fn ok_flag(i: &Interner, ok: bool) -> Context {
    FxHashMap::from_iter([(i.intern("ok"), typed(Ty::Bool, Value::bool_(ok)))])
}

#[tokio::test]
async fn a_body_returning_what_the_host_declared_runs() {
    let i = Interner::new();
    let v = run_script(&i, "1 + 2", Context::default(), Ty::I64).await;
    assert_eq!(v.as_int(), 3);
}

/// RFC-0047 T2's sabotage, made permanent: the body leaves with a `String`
/// where the host reads an `i64`, and the compilation refuses it, naming the
/// entry, the declared type and the type the body leaves with.
#[tokio::test]
#[should_panic(expected = "[main] type mismatch: expected i64, got String")]
async fn a_body_returning_other_than_the_host_declared_does_not_compile() {
    let _ = run_script(&i(), r#""no""#, Context::default(), Ty::I64).await;
}

fn i() -> Interner {
    Interner::new()
}

/// RFC-0038: no value of `!` exists, so a diverging body satisfies whatever
/// the host declared.
#[tokio::test]
#[should_panic(expected = "no")]
async fn a_diverging_body_satisfies_any_declaration() {
    let _ = run_script(&i(), r#"panic("no")"#, Context::default(), Ty::I64).await;
}

/// A `?` is a second `Return`. Under a declaration both are held to it: the
/// tail agrees with `i64` and the early return leaves with a `Result`, so the
/// second one is refused.
#[tokio::test]
#[should_panic(expected = "`?` leaves with Result<!, String> but the function returns i64")]
async fn a_second_return_disagreeing_with_the_declaration_is_refused() {
    let interner = Interner::new();
    let c = ok_flag(&interner, true);
    let _ = run_script_mode(&interner, TWO_RETURNS, c, Ty::I64).await;
}

/// The same body, declared as what its `?` leaves with, compiles and runs.
#[tokio::test]
async fn both_returns_agreeing_with_the_declaration_run() {
    let interner = Interner::new();
    let c = ok_flag(&interner, true);
    let v = run_script_mode(
        &interner,
        TWO_RETURNS,
        c,
        Ty::Result(Box::new(Ty::I64), Box::new(Ty::String)),
    )
    .await;
    // SAFETY: the host declared `Result<i64, String>`.
    assert!(unsafe { v.as_result() }.is_ok());
}
