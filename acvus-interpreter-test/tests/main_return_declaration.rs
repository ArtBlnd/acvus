//! The host declares what `main` returns, and the compilation holds the body
//! to it (RFC-0054).

use acvus_interpreter::{Composite, Value};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

/// A body with two `Return`s: the `?` leaves early with the `Err`, the tail
/// leaves with the `Ok` payload.
const TWO_RETURNS: &str = r#"let r = if @ok { Ok(7) } else { Err("bad".to_string()) };
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
    let _ = run_script(&i(), r#""no".to_string()"#, Context::default(), Ty::I64).await;
}

fn i() -> Interner {
    Interner::new()
}

/// RFC-0038: no value of `!` exists, so a diverging body satisfies whatever
/// the host declared.
#[tokio::test]
#[should_panic(expected = "no")]
async fn a_diverging_body_satisfies_any_declaration() {
    let _ = run_script(
        &i(),
        r#"panic("no".to_string())"#,
        Context::default(),
        Ty::I64,
    )
    .await;
}

/// RFC-0054: a `!` declaration is the host's "I state no return type".
#[tokio::test]
async fn a_host_declaring_never_gets_the_value_and_reads_it_by_kind() {
    let v = run_script(
        &i(),
        r#""by kind".to_string()"#,
        Context::default(),
        Ty::Never,
    )
    .await;
    assert_eq!(v.composite(), Some(Composite::String));
    // SAFETY: the vtable witnesses a String behind the pointer.
    assert_eq!(unsafe { v.as_str() }, "by kind");
}

/// A `?` is a second `Return`. Under a declaration both are held to it: the
/// tail agrees with `i64` and the early return leaves with a `Result`, so the
/// second one is refused.
#[tokio::test]
#[should_panic(expected = "`?` leaves with Result<_, String> but the function returns i64")]
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
    let variant = unsafe { v.as_variant() };
    // SAFETY: the same witness — a variant's first register is its tag.
    let tag = unsafe { variant.tag().as_tag() };
    assert_eq!(interner.resolve(tag), "Ok");
}
