//! A captured word is a copy (RFC-0018). A lambda's body sees a capture
//! whose type is a word as that word — at every width RFC-0037 names, and
//! for `Float` and `Bool` — and reads it with no `*`: bound to a name,
//! assigned into, used as an index, and as an operand. A capture of any
//! other type stays lent, which is what the last two tests measure.

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::Interner;

/// `let m = k`, `m = m + k` and `m + y` — the three forms the probe found
/// refused at every width — over one captured word.
const THE_FORMS: &str = "let k = @w; let f = |y| -> { let m = k; m = m + k; m + y }; f(k)";

fn word_context(i: &Interner, ty: IntTy, bits: u64) -> Context {
    [(
        i.intern("w"),
        typed(Ty::Int(ty), Value::from_bits(ty, bits)),
    )]
    .into_iter()
    .collect()
}

async fn three_times(i: &Interner, ty: IntTy) -> i128 {
    let v = run_script(i, THE_FORMS, word_context(i, ty, 3), Ty::Int(ty)).await;
    ty.read(v.bits())
}

#[tokio::test]
async fn a_captured_word_is_bound_assigned_and_added_at_every_width() {
    let i = Interner::new();
    for ty in [
        IntTy::I8,
        IntTy::I16,
        IntTy::I32,
        IntTy::I64,
        IntTy::U8,
        IntTy::U16,
        IntTy::U32,
        IntTy::U64,
    ] {
        assert_eq!(three_times(&i, ty).await, 9, "{ty:?}");
    }
}

#[tokio::test]
async fn a_captured_u64_is_an_index() {
    let i = Interner::new();
    let v = run_script_mode(
        &i,
        "let a = [[10, 20], [30, 40]]; \
         let z = len(&a) - len(&a); \
         as_iter(&a) | map(|row| -> row[z]) | sum",
        Context::default(),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 40);
}

#[tokio::test]
async fn a_captured_float_multiplies() {
    let i = Interner::new();
    let v = run_script_mode(
        &i,
        "let h = 0.5; let f = |y| -> h * y; f(4.0)",
        Context::default(),
        Ty::Float,
    )
    .await;
    assert_eq!(v.as_float(), 2.0);
}

#[tokio::test]
async fn a_captured_bool_conjoins() {
    let i = Interner::new();
    let v = run_script_mode(
        &i,
        "let b = true; let f = |y| -> b && y; f(true)",
        Context::default(),
        Ty::Bool,
    )
    .await;
    assert!(v.as_bool());
}

#[tokio::test]
async fn a_captured_word_is_returned_by_value() {
    let i = Interner::new();
    let v = run_script_mode(
        &i,
        "let k = 7; let f = |y| -> k; f(1)",
        Context::default(),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 7);
}

/// The same program one type apart: a captured `String` is lent, so the
/// body's `s` is a reference, and a lambda returns no reference.
#[tokio::test]
#[should_panic(expected = "a lambda cannot return a reference")]
async fn a_captured_string_is_lent_and_so_is_not_returned_by_value() {
    let i = Interner::new();
    run_script_mode(
        &i,
        "let s = \"a\".to_string(); let f = |y| -> s; f(1)",
        Context::default(),
        Ty::String,
    )
    .await;
}
