//! `Erased<R, T>` inside a running script (E2, E3): an element edited
//! through `as_mut` is what the program reads afterwards, and equality over
//! an `Iter` of it goes through `as_ref` with no `Monomorphize` member list.

use std::any::type_name;
use std::sync::Arc;

use acvus_ext::Iter;
use acvus_extern::{
    EffectVar, Erased, FromValue, IdentityVar, RefMut, Registry, Runtime, extern_fn,
    extern_registry,
};
use acvus_interpreter::{AcvusRuntime, InterpreterContext, SequentialExecutor, Value};
use acvus_interpreter_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

#[extern_fn(effect = pure)]
fn upcase_first<Rt>(rt: &Rt, mut items: RefMut<Vec<Erased<Rt, String>>, Rt>)
where
    Rt: Runtime,
{
    let Some(first) = items.as_mut_slice(rt).first_mut() else {
        return;
    };
    first.as_mut(rt).make_ascii_uppercase();
}

#[extern_fn(effect = E, sync = contains_erased_now)]
async fn contains_erased<E, I, Rt>(
    rt: &Rt,
    mut it: Iter<Erased<Rt, String>, E, I, Rt>,
    needle: Erased<Rt, String>,
) -> bool
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    while let Some(item) = it.next(rt).await {
        if item.as_ref(rt) == needle.as_ref(rt) {
            return true;
        }
    }
    false
}

fn contains_erased_now<E, I, Rt>(
    rt: &Rt,
    mut it: Iter<Erased<Rt, String>, E, I, Rt>,
    needle: Erased<Rt, String>,
) -> bool
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    while let Some(item) = it.next_now(rt) {
        if item.as_ref(rt) == needle.as_ref(rt) {
            return true;
        }
    }
    false
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [upcase_first, contains_erased],
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

fn assert_str(v: &Value, expected: &str) {
    assert!(v.is_string(), "expected a String, got {v:?}");
    // SAFETY: the witness is String.
    assert_eq!(unsafe { v.as_str() }, expected);
}

// -- E2 -----------------------------------------------------------------

#[tokio::test]
async fn an_element_edited_through_as_mut_is_what_the_program_reads() {
    let v =
        run(r#"let v = split_str("ab,cd", ","); upcase_first(&mut v); into_iter(v) | join("+")"#)
            .await;
    assert_str(&v, "AB+cd");
}

// -- E3 -----------------------------------------------------------------

#[tokio::test]
async fn contains_compares_through_as_ref() {
    let hit = run(r#"into_iter(split_str("a,b,c", ",")) | contains_erased("b")"#).await;
    assert!(hit.as_bool());
    let miss = run(r#"into_iter(split_str("a,b,c", ",")) | contains_erased("z")"#).await;
    assert!(!miss.as_bool());
}

// -- The checked exit on an Inline element ------------------------------

#[tokio::test]
async fn contains_over_inline_elements_reads_them_by_their_tag() {
    assert!(run("into_iter([1, 2, 3]) | contains(3)").await.as_bool());
    assert!(!run("into_iter([1, 2, 3]) | contains(4)").await.as_bool());
    assert!(run("into_iter([1.5, 2.5]) | contains(2.5)").await.as_bool());
    assert!(
        run("into_iter([true, false]) | contains(false)")
            .await
            .as_bool()
    );
}

fn runtime(i: &Interner) -> AcvusRuntime {
    InterpreterContext::new(i, FxHashMap::default(), Arc::new(SequentialExecutor)).runtime()
}

#[test]
fn type_of_reports_the_tag_of_a_small_value() {
    use std::any::TypeId;
    let rt = runtime(&Interner::new());
    assert_eq!(rt.type_of(&Value::int(1)), Some(TypeId::of::<i64>()));
    assert_eq!(rt.type_of(&Value::float(1.0)), Some(TypeId::of::<f64>()));
    assert_eq!(rt.type_of(&Value::bool_(true)), Some(TypeId::of::<bool>()));
    assert_eq!(rt.type_of(&Value::unit()), Some(TypeId::of::<()>()));
    assert_eq!(
        rt.type_of(&Value::string("s")),
        Some(TypeId::of::<String>())
    );
    assert_eq!(rt.type_of(&Value::EMPTY), None);
    assert_eq!(rt.type_of(&Value::UNDEF), None);
    let target = Value::int(1);
    assert_eq!(rt.type_of(&Value::reference(&target)), None);
}

#[test]
#[should_panic(expected = "expected a value erased from `f64`, found one erased from `i64`")]
fn from_value_on_an_int_as_a_float_panics_naming_both_types() {
    let rt = runtime(&Interner::new());
    Erased::<AcvusRuntime, f64>::from_value(&rt, Value::int(2));
}

#[test]
fn from_value_on_an_int_as_an_int_is_the_value() {
    let rt = runtime(&Interner::new());
    let erased = Erased::<AcvusRuntime, i64>::from_value(&rt, Value::int(2));
    assert_eq!(erased.get(), 2);
}

#[test]
#[should_panic(expected = "found a value no Rust type was erased into")]
fn from_value_on_a_reference_panics_as_a_value_erased_from_no_type() {
    let rt = runtime(&Interner::new());
    let target = Value::int(2);
    Erased::<AcvusRuntime, i64>::from_value(&rt, Value::reference(&target));
}
