//! The form of an `Option` value: `None` is a word that counts the `Some`s
//! around it, and `Some(v)` for any other `v` is `v` itself (RFC-0022).
//! Every case here crosses the extern boundary, a pattern, or both.

use acvus_extern::{Cross, Registry, Runtime, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, InterpreterContext, Kind, SequentialExecutor, Value};
use acvus_interpreter_test::*;
use acvus_utils::Interner;

#[extern_fn(effect = pure)]
fn nest(v: Option<i64>) -> Option<Option<i64>> {
    Some(v)
}

#[extern_fn(effect = pure)]
fn no_nest() -> Option<Option<i64>> {
    None
}

#[extern_fn(effect = pure)]
fn flatten(v: Option<Option<i64>>) -> i64 {
    match v {
        Some(Some(n)) => n,
        Some(None) => -1,
        None => -2,
    }
}

#[extern_fn(effect = pure)]
fn nest3(v: Option<Option<i64>>) -> Option<Option<Option<i64>>> {
    Some(v)
}

#[extern_fn(effect = pure)]
fn flatten3(v: Option<Option<Option<i64>>>) -> i64 {
    match v {
        Some(Some(Some(n))) => n,
        Some(Some(None)) => -1,
        Some(None) => -2,
        None => -3,
    }
}

#[extern_fn(effect = pure)]
fn evens(n: i64) -> Vec<Option<i64>> {
    (0..n)
        .map(|i| if i % 2 == 0 { Some(i) } else { None })
        .collect()
}

#[extern_fn(effect = pure)]
fn count_some(v: Vec<Option<i64>>) -> i64 {
    v.iter().filter(|x| x.is_some()).count() as i64
}

#[extern_fn(effect = pure)]
fn nested_vec(n: i64) -> Vec<Option<Option<i64>>> {
    (0..n)
        .map(|i| match i % 3 {
            0 => Some(Some(i)),
            1 => Some(None),
            _ => None,
        })
        .collect()
}

#[extern_fn(effect = pure)]
fn count_outer_some(v: Vec<Option<Option<i64>>>) -> i64 {
    v.iter().filter(|x| x.is_some()).count() as i64
}

#[extern_fn(effect = pure)]
fn count_inner_some(v: Vec<Option<Option<i64>>>) -> i64 {
    v.iter().filter(|x| matches!(x, Some(Some(_)))).count() as i64
}

#[extern_fn(effect = pure)]
fn sum_nested(v: Vec<Option<Option<i64>>>) -> i64 {
    v.iter().flatten().flatten().sum()
}

#[extern_fn(effect = pure)]
fn some_if_even(k: i64) -> Option<i64> {
    (k % 2 == 0).then_some(k)
}

#[extern_fn(effect = pure)]
fn ints(n: i64) -> Vec<i64> {
    (0..n).collect()
}

#[extern_fn(effect = pure)]
fn maybe_ints(n: i64) -> Option<Vec<i64>> {
    (n > 0).then(|| (0..n).collect())
}

fn regs() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        fns: [
            nest, no_nest, flatten, nest3, flatten3, evens, count_some, nested_vec,
            count_outer_some, count_inner_some, sum_nested, ints, maybe_ints, some_if_even,
        ],
    });
    regs
}

async fn run(i: &Interner, source: &str) -> Value {
    run_script_mode_with_externs(i, source, Context::default(), regs())
        .await
        .value
}

// -- The depth word ----------------------------------------------------

#[tokio::test]
async fn a_nested_option_round_trips_through_an_extern() {
    let i = Interner::new();
    assert_eq!(run(&i, "flatten(nest(Some(1)))").await.as_int(), 1);
    assert_eq!(run(&i, "flatten(nest(None))").await.as_int(), -1);
    assert_eq!(run(&i, "flatten(no_nest())").await.as_int(), -2);
}

#[tokio::test]
async fn a_nested_option_round_trips_through_a_pattern() {
    let i = Interner::new();
    let src = |call: &str| {
        format!(
            "if let Some(inner) = {call} {{ if let Some(n) = inner {{ n }} else {{ -1 }} }} else {{ -2 }}"
        )
    };
    assert_eq!(run(&i, &src("nest(Some(9))")).await.as_int(), 9);
    assert_eq!(run(&i, &src("nest(None)")).await.as_int(), -1);
    assert_eq!(run(&i, &src("no_nest()")).await.as_int(), -2);
}

#[tokio::test]
async fn depth_three_round_trips_through_an_extern() {
    let i = Interner::new();
    assert_eq!(run(&i, "flatten3(nest3(nest(Some(5))))").await.as_int(), 5);
    assert_eq!(run(&i, "flatten3(nest3(nest(None)))").await.as_int(), -1);
    assert_eq!(run(&i, "flatten3(nest3(no_nest()))").await.as_int(), -2);
}

#[tokio::test]
async fn depth_three_round_trips_through_nested_patterns() {
    let i = Interner::new();
    let src = |call: &str| {
        format!(
            "if let Some(a) = {call} {{ \
               if let Some(b) = a {{ if let Some(n) = b {{ n }} else {{ -1 }} }} else {{ -2 }} \
             }} else {{ -3 }}"
        )
    };
    assert_eq!(run(&i, &src("nest3(nest(Some(5)))")).await.as_int(), 5);
    assert_eq!(run(&i, &src("nest3(nest(None))")).await.as_int(), -1);
    assert_eq!(run(&i, &src("nest3(no_nest())")).await.as_int(), -2);
}

#[tokio::test]
async fn unwrap_opens_each_level_of_a_nested_option() {
    let i = Interner::new();
    assert_eq!(run(&i, "unwrap(unwrap(nest(Some(4))))").await.as_int(), 4);
    assert!(run(&i, "unwrap(nest(None))").await.is_none());
    assert_eq!(
        run(&i, "unwrap_or(unwrap(nest(None)), 5)").await.as_int(),
        5
    );
    assert_eq!(
        run(&i, "unwrap_or(unwrap_or(no_nest(), None), 6)")
            .await
            .as_int(),
        6
    );
}

// -- A payload that is not a word --------------------------------------

#[tokio::test]
async fn an_option_of_a_vec_is_its_vec() {
    let i = Interner::new();
    let src = |n: i64| format!("if let Some(v) = maybe_ints({n}) {{ len(&v) }} else {{ 99 }}");
    assert_eq!(run(&i, &src(3)).await.as_int(), 3);
    assert_eq!(run(&i, &src(0)).await.as_int(), 99);
}

#[tokio::test]
async fn an_option_of_a_reference_is_its_reference() {
    let i = Interner::new();
    let src =
        |n: i64| format!("let v = ints({n}); if let Some(x) = first(&v) {{ *x }} else {{ -1 }}");
    assert_eq!(run(&i, &src(3)).await.as_int(), 0);
    assert_eq!(run(&i, &src(0)).await.as_int(), -1);
}

// -- Through a reference -----------------------------------------------

#[tokio::test]
async fn a_pattern_through_a_reference_binds_an_inner_none_as_that_none() {
    let i = Interner::new();
    let src = |call: &str| {
        format!(
            "let o = {call}; \
             if let Some(inner) = &o {{ if let Some(n) = inner {{ *n }} else {{ 7 }} }} else {{ -2 }}"
        )
    };
    assert_eq!(run(&i, &src("nest(Some(3))")).await.as_int(), 3);
    assert_eq!(run(&i, &src("nest(None)")).await.as_int(), 7);
    assert_eq!(run(&i, &src("no_nest()")).await.as_int(), -2);
}

// -- A container of options --------------------------------------------

#[tokio::test]
async fn a_vec_of_options_stores_each_element_flat() {
    let i = Interner::new();
    assert_eq!(run(&i, "count_some(evens(5))").await.as_int(), 3);
    assert_eq!(run(&i, "count_some(evens(0))").await.as_int(), 0);
}

#[tokio::test]
async fn an_element_of_a_vec_of_options_is_read_back_through_a_pattern() {
    let i = Interner::new();
    let src = |k: i64| format!("let v = evens(4); if let Some(n) = v[{k}] {{ n }} else {{ -1 }}");
    assert_eq!(run(&i, &src(0)).await.as_int(), 0);
    assert_eq!(run(&i, &src(1)).await.as_int(), -1);
    assert_eq!(run(&i, &src(2)).await.as_int(), 2);
}

#[tokio::test]
async fn a_vec_of_nested_options_round_trips_whole() {
    let i = Interner::new();
    for n in [0i64, 6] {
        let fixture = nested_vec(n);
        let outer = fixture.iter().filter(|x| x.is_some()).count() as i64;
        let inner = fixture
            .iter()
            .filter(|x| matches!(x, Some(Some(_))))
            .count() as i64;
        let sum: i64 = fixture.iter().flatten().flatten().sum();
        let ask = |f: &str| format!("{f}(nested_vec({n}))");
        assert_eq!(run(&i, &ask("count_outer_some")).await.as_int(), outer);
        assert_eq!(run(&i, &ask("count_inner_some")).await.as_int(), inner);
        assert_eq!(run(&i, &ask("sum_nested")).await.as_int(), sum);
    }
}

// -- The layout the rule rests on --------------------------------------

fn runtime(i: &Interner) -> AcvusRuntime {
    InterpreterContext::new(
        i,
        rustc_hash::FxHashMap::default(),
        std::sync::Arc::new(SequentialExecutor),
    )
    .runtime()
}

#[test]
fn none_is_a_kind_and_option_of_a_value_keeps_its_niche() {
    assert_eq!(Value::NONE.kind(), Kind::None);
    assert_eq!(size_of::<Value>(), 16);
    assert_eq!(size_of::<Option<Value>>(), 16);
    let i = Interner::new();
    let rt = runtime(&i);
    let flat = <Option<i64> as Cross<AcvusRuntime>>::erase(Some(7), &rt);
    assert_eq!(flat.kind(), Kind::I64);
    assert_eq!(
        unsafe { <Option<i64> as Cross<AcvusRuntime>>::materialize(&rt, flat) },
        Some(7)
    );
    let none = <Option<i64> as Cross<AcvusRuntime>>::erase(None, &rt);
    assert_eq!(none.kind(), Kind::None);
    assert_eq!(
        unsafe { <Option<i64> as Cross<AcvusRuntime>>::materialize(&rt, none) },
        None
    );
}

#[test]
fn a_some_is_never_none_and_opens_back_to_its_payload() {
    let i = Interner::new();
    let rt = runtime(&i);
    assert!(rt.is_none(&rt.none()));

    let mut ladder = rt.none();
    for depth in 0..4 {
        let wrapped = rt.some(ladder.copy_word());
        assert!(!rt.is_none(&wrapped), "depth {depth}: {wrapped:?}");
        assert_eq!(rt.unwrap_some(wrapped), ladder, "depth {depth}");
        ladder = rt.some(ladder);
    }

    let word = rt.some(Value::int(3));
    assert!(!rt.is_none(&word));
    assert_eq!(rt.unwrap_some(word), Value::int(3));

    let large = rt.some(Value::string("s"));
    assert!(!rt.is_none(&large));
    let back = rt.unwrap_some(large);
    assert_eq!(unsafe { back.as_str() }, "s");
}

#[tokio::test]
async fn a_mapped_option_element_by_value_is_visited_at_every_step() {
    let i = Interner::new();
    let v = run(
        &i,
        "let it = range(0, 4) | map(|k| -> some_if_even(k)); \
         let count = 0; while let Some(x) = next(&mut it) { count = count + 1; } count",
    )
    .await;
    assert_eq!(v.as_int(), 4);
}
