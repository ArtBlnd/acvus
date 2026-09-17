//! Regression tests for RFC-0041 inside a running script (R13–R17): a
//! place lent twice to one specialized call, a field place, a lend in a
//! loop, a place lent again by a nested call, an `Iter` pipeline built by
//! the standard `map`/`filter`/`collect`,
//! `contains` over an `Erased` element, and the representation a `Copy`
//! extension type takes. A test that fails is a finding, kept as it fails.

use std::any::{TypeId, type_name};
use std::sync::Arc;

use acvus_extern::{
    Erased, FromValue, Monomorphize, Registry, Runtime, extern_fn, extern_registry,
};
use acvus_interpreter::{AcvusRuntime, InterpreterContext, SequentialExecutor, Value};
use acvus_interpreter_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

trait Float: Copy + Send + Sync + 'static {
    const ZERO: Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn sqrt(self) -> Self;
}

impl Float for f64 {
    const ZERO: Self = 0.0;

    fn mul_add(self, a: Self, b: Self) -> Self {
        self + a * b
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

#[extern_fn(effect = pure)]
fn norm<T>(v: &Vec<T>) -> T
where
    T: Monomorphize<(f64,)> + Float,
{
    v.iter().fold(T::ZERO, |acc, x| acc.mul_add(*x, *x)).sqrt()
}

#[extern_fn(effect = pure)]
fn dot2<T>(a: &Vec<T>, b: &Vec<T>) -> T
where
    T: Monomorphize<(f64,)> + Float,
{
    a.iter()
        .zip(b.iter())
        .fold(T::ZERO, |acc, (x, y)| acc.mul_add(*x, *y))
}

#[extern_fn(effect = pure)]
fn scale<T>(v: &Vec<T>, k: T) -> T
where
    T: Monomorphize<(f64,)> + Float,
{
    v.iter().fold(T::ZERO, |acc, x| acc.mul_add(*x, k))
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [norm, dot2, scale],
    }
}

async fn run(source: &str) -> Value {
    let i = Interner::new();
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(registry());
    run_script_mode_with_externs(&i, source, Context::default(), registries)
        .await
        .value
}

// -- R14: R10, R11, R12 executed ----------------------------------------------

#[tokio::test]
async fn one_place_lent_twice_to_one_specialized_call_reads_the_same_storage_twice() {
    let v = run("let x = vec([3.0, 4.0]); dot2(&x, &x)").await;
    assert_eq!(v.as_float(), 25.0);
}

#[tokio::test]
async fn a_field_place_lent_to_a_specialized_parameter_is_read_through_the_cast() {
    let v = run("let o = { v: vec([3.0, 4.0]), }; norm(&o.v)").await;
    assert_eq!(v.as_float(), 5.0);
}

#[tokio::test]
async fn a_lend_inside_a_loop_is_cast_back_each_iteration_and_the_sum_is_unchanged() {
    let v = run(
        "let x = vec([3.0, 4.0]); let n = 0; let acc = 0.0; while n < 3 { acc = acc + norm(&x); n = n + 1; } acc",
    )
    .await;
    assert_eq!(v.as_float(), 15.0);
}

// -- R17: a nested call reads the place the outer call holds ------------------

#[tokio::test]
async fn a_place_lent_to_a_call_and_again_inside_a_nested_argument_computes_as_uniform() {
    let v = run("let x = vec([3.0, 4.0]); scale(&x, norm(&x))").await;
    assert_eq!(v.as_float(), 35.0, "(3 + 4) * norm([3, 4]) = 7 * 5");
}

// -- R15: the standard pipeline -----------------------------------------------

#[tokio::test]
async fn map_then_collect_has_the_source_s_length() {
    let v = run("let xs = into_iter([1, 2, 3]) | map(|x| -> x * 2) | collect; xs.len()").await;
    assert_eq!(v.as_int(), 3);
}

#[tokio::test]
async fn map_then_filter_then_collect_keeps_the_doubled_values_above_two() {
    let v = run(
        "let ys = into_iter([1, 2, 3]) | map(|x| -> x * 2) | filter(|x| -> *x > 2) | collect; ys.len() * 100 + *ys.get(0) * 10 + *ys.get(1)",
    )
    .await;
    assert_eq!(v.as_int(), 246, "two elements, 4 and 6");
}

// -- R13: contains over an Erased element from a uniform Vec literal ----------

#[tokio::test]
async fn contains_over_a_uniform_int_vec_finds_a_member_and_misses_a_stranger() {
    assert!(
        run("into_iter(vec([1, 2, 3])) | contains(2)")
            .await
            .as_bool()
    );
    assert!(
        !run("into_iter(vec([1, 2, 3])) | contains(5)")
            .await
            .as_bool()
    );
}

// -- R16: a Copy extension type outside the Inline set ------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
struct Pixel {
    x: u32,
    y: u32,
}

acvus_extern::cross_as_stored!(Pixel);

fn runtime(i: &Interner) -> AcvusRuntime {
    InterpreterContext::new(i, FxHashMap::default(), Arc::new(SequentialExecutor)).runtime()
}

#[test]
fn a_copy_struct_that_fits_the_word_but_is_not_inline_crosses_as_large_with_its_type_recorded() {
    let rt = runtime(&Interner::new());
    let pixel = Pixel { x: 1, y: 2 };
    let value = Erased::<AcvusRuntime, Pixel>::new(&rt, pixel).into_value();
    assert!(
        value.kind() == acvus_interpreter::Kind::Large,
        "a Copy type outside the Inline set is a Large box: {value:?}"
    );
    assert_eq!(rt.type_of(&value), Some(TypeId::of::<Pixel>()));
    assert_eq!(rt.type_name_of(&value), Some(type_name::<Pixel>()));
    let back = Erased::<AcvusRuntime, Pixel>::from_value(&rt, value);
    assert_eq!(*back.as_ref(&rt), pixel);
}
