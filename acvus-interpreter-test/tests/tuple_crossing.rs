//! A Rust tuple crosses the extern boundary as the script's tuple, as an
//! argument and as a result, each element by its own crossing.

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[extern_fn(effect = pure)]
fn swap(pair: (i64, String)) -> (String, i64) {
    let (n, s) = pair;
    (s, n)
}

#[extern_fn(effect = pure)]
fn describe(pair: (String, i64)) -> String {
    let (s, n) = pair;
    format!("{s}:{n}")
}

#[extern_fn(effect = pure)]
fn nest(n: i64, b: bool, u: u8) -> (i64, (bool, u8)) {
    (n, (b, u))
}

#[extern_fn(effect = pure)]
fn flat(nested: (i64, (bool, u8))) -> i64 {
    let (n, (b, u)) = nested;
    if b { n + i64::from(u) } else { -n }
}

#[extern_fn(effect = pure)]
fn pair_if(k: i64) -> Option<(i64, bool)> {
    (k >= 0).then_some((k, k % 2 == 0))
}

#[extern_fn(effect = pure)]
fn sum_pair(maybe: Option<(i64, bool)>) -> i64 {
    match maybe {
        Some((n, true)) => n * 10,
        Some((n, false)) => n,
        None => -1,
    }
}

#[extern_fn(effect = pure)]
fn single(n: i64) -> (i64,) {
    (n,)
}

#[extern_fn(effect = pure)]
fn unsingle(one: (i64,)) -> i64 {
    one.0
}

#[extern_fn(effect = pure)]
fn eight(n: i64) -> (i64, i64, i64, i64, i64, i64, i64, i64) {
    (n, n + 1, n + 2, n + 3, n + 4, n + 5, n + 6, n + 7)
}

#[extern_fn(effect = pure)]
fn weigh_eight(all: (i64, i64, i64, i64, i64, i64, i64, i64)) -> i64 {
    let (a, b, c, d, e, f, g, h) = all;
    a + 10 * b + 100 * c + 1_000 * d + 10_000 * e + 100_000 * f + 1_000_000 * g + 10_000_000 * h
}

fn regs() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        fns: [swap, describe, nest, flat, pair_if, sum_pair, single, unsingle, eight, weigh_eight],
    });
    regs
}

fn assert_str(v: &Value, expected: &str) {
    assert!(v.is_string(), "expected a String, got {v:?}");
    // SAFETY: the witness is String.
    assert_eq!(unsafe { v.as_str() }, expected);
}

async fn run(i: &Interner, source: &str, ret: Ty) -> Value {
    run_script_mode_with_externs(i, source, Context::default(), regs(), ret)
        .await
        .value
}

#[tokio::test]
async fn a_pair_crosses_in_and_back_out_swapped() {
    let i = Interner::new();
    assert_str(
        &run(&i, "describe(swap((7, \"ab\".to_string())))", Ty::String).await,
        "ab:7",
    );
}

#[tokio::test]
async fn a_returned_pair_is_a_tuple_the_script_takes_apart() {
    let i = Interner::new();
    assert_eq!(
        run(&i, "match swap((7, \"ab\".to_string())) { (s, n) => n, _ => -1 }", Ty::I64)
            .await
            .as_int(),
        7
    );
    assert_str(
        &run(&i, "match swap((7, \"ab\".to_string())) { (s, n) => s, _ => \"\".to_string() }", Ty::String).await,
        "ab",
    );
}

#[tokio::test]
async fn a_nested_tuple_crosses_element_by_element() {
    let i = Interner::new();
    assert_eq!(run(&i, "flat(nest(3, true, 9u8))", Ty::I64).await.as_int(), 12);
    assert_eq!(run(&i, "flat(nest(3, false, 9u8))", Ty::I64).await.as_int(), -3);
    assert_eq!(
        run(
            &i,
            "match nest(3, true, 9u8) { (n, (true, u)) => n + (u as i64), (n, (false, u)) => 0 - n, _ => -1 }",
            Ty::I64,
        )
        .await
        .as_int(),
        12
    );
    assert_eq!(run(&i, "flat((5, (true, 1u8)))", Ty::I64).await.as_int(), 6);
}

#[tokio::test]
async fn a_tuple_crosses_inside_an_option() {
    let i = Interner::new();
    assert_eq!(run(&i, "sum_pair(pair_if(4))", Ty::I64).await.as_int(), 40);
    assert_eq!(run(&i, "sum_pair(pair_if(3))", Ty::I64).await.as_int(), 3);
    assert_eq!(run(&i, "sum_pair(pair_if(-1))", Ty::I64).await.as_int(), -1);
    assert_eq!(
        run(
            &i,
            "if let Some((n, even)) = pair_if(4) { if even { n } else { 0 - n } } else { -1 }",
            Ty::I64,
        )
        .await
        .as_int(),
        4
    );
    assert_eq!(run(&i, "sum_pair(Some((2, true)))", Ty::I64).await.as_int(), 20);
    assert_eq!(run(&i, "sum_pair(None)", Ty::I64).await.as_int(), -1);
}

#[tokio::test]
async fn the_narrowest_and_the_widest_arity_cross() {
    let i = Interner::new();
    assert_eq!(run(&i, "unsingle(single(5))", Ty::I64).await.as_int(), 5);
    assert_eq!(
        run(&i, "weigh_eight(eight(1))", Ty::I64).await.as_int(),
        87_654_321
    );
    assert_eq!(
        run(&i, "match eight(1) { (a, b, c, d, e, f, g, h) => h * 10 + a, _ => -1 }", Ty::I64)
            .await
            .as_int(),
        81
    );
}
