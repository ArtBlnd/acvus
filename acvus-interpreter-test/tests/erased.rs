//! `Erased<R, T>` inside a running script (E2, E3): an element edited
//! through `as_mut` is what the program reads afterwards, and equality over
//! a container of it goes through `as_ref` with no `Monomorphize` member
//! list.
//!
//! E3 reads the collected container and not the pipeline itself. A stage
//! consumer would have to name `acvus_ext`'s `iter::next` signature in its
//! required instance, and that signature lives in a private module, so no
//! handler outside `acvus-ext` can be written as one.

use std::sync::Arc;

use acvus_extern::Ctx;
use acvus_extern::{Erased, FromValue, Registry, Runtime, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, InterpreterContext, SequentialExecutor, Value};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

#[extern_fn(effect = pure)]
fn upcase_first<Rt>(ctx: &mut Ctx<'_, Rt>, items: &mut [Erased<Rt, String>])
where
    Rt: Runtime,
{
    let rt = ctx.rt;
    let Some(first) = items.first_mut() else {
        return;
    };
    first.as_mut(rt).make_ascii_uppercase();
}

#[extern_fn(effect = pure)]
fn contains_erased<Rt>(
    ctx: &mut Ctx<'_, Rt>,
    items: &[Erased<Rt, String>],
    needle: Erased<Rt, String>,
) -> bool
where
    Rt: Runtime,
{
    let rt = ctx.rt;
    items
        .iter()
        .any(|item| item.as_ref(rt) == needle.as_ref(rt))
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [upcase_first, contains_erased],
    }
}

async fn run(source: &str, ret: Ty) -> Value {
    let i = Interner::new();
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(registry());
    run_script_mode_with_externs(&i, source, Context::default(), registries, ret)
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
    let v = run(
        r#"let v = split("ab,cd", ",") | collect; upcase_first(&mut v); into_iter(v) | join("+".to_string())"#,
        Ty::String,
    )
    .await;
    assert_str(&v, "AB+cd");
}

// -- E3 -----------------------------------------------------------------

#[tokio::test]
async fn contains_compares_through_as_ref() {
    let hit = run(
        r#"let v = split("a,b,c", ",") | collect; contains_erased(&v, "b".to_string())"#,
        Ty::Bool,
    )
    .await;
    assert!(hit.as_bool());
    let miss = run(
        r#"let v = split("a,b,c", ",") | collect; contains_erased(&v, "z".to_string())"#,
        Ty::Bool,
    )
    .await;
    assert!(!miss.as_bool());
}

// -- The checked exit on an Inline element ------------------------------

#[tokio::test]
async fn contains_over_inline_elements_reads_them_by_their_tag() {
    assert!(
        run("into_iter([1, 2, 3]) | contains(3)", Ty::Bool)
            .await
            .as_bool()
    );
    assert!(
        !run("into_iter([1, 2, 3]) | contains(4)", Ty::Bool)
            .await
            .as_bool()
    );
    assert!(
        run("into_iter([1.5, 2.5]) | contains(2.5)", Ty::Bool)
            .await
            .as_bool()
    );
    assert!(
        run("into_iter([true, false]) | contains(false)", Ty::Bool)
            .await
            .as_bool()
    );
}

fn runtime(i: &Interner) -> AcvusRuntime {
    InterpreterContext::new(i, FxHashMap::default(), Arc::new(SequentialExecutor))
        .runtime_over_an_empty_page()
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
    assert_eq!(rt.type_of(&Value::NONE), None);
    assert_eq!(rt.type_of(&Value::UNDEF), None);
    let target = Value::int(1);
    assert_eq!(rt.type_of(&Value::reference(&target)), None);
}

/// The broken contract is a `debug_assert!` (`FromValue`'s door), so this is
/// what a debug build shows and a release build does not look for.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "expected a value erased from `f64`, found one erased from `i64`")]
fn from_value_on_an_int_as_a_float_panics_naming_both_types() {
    let rt = runtime(&Interner::new());
    // SAFETY: deliberately broken — this test is what the door refuses.
    unsafe { Erased::<AcvusRuntime, f64>::from_value(&rt, Value::int(2)) };
}

#[test]
fn from_value_on_an_int_as_an_int_is_the_value() {
    let rt = runtime(&Interner::new());
    // SAFETY: `Value::int` is the runtime's erasure of an `i64`.
    let erased = unsafe { Erased::<AcvusRuntime, i64>::from_value(&rt, Value::int(2)) };
    assert_eq!(erased.get(), 2);
}

/// The broken contract is a `debug_assert!` (`FromValue`'s door), so this is
/// what a debug build shows and a release build does not look for.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "found a value no Rust type was erased into")]
fn from_value_on_a_reference_panics_as_a_value_erased_from_no_type() {
    let rt = runtime(&Interner::new());
    let target = Value::int(2);
    // SAFETY: deliberately broken — this test is what the door refuses.
    unsafe { Erased::<AcvusRuntime, i64>::from_value(&rt, Value::reference(&target)) };
}
