//! The form of a `Result` value: the flat variant every other variant type is,
//! holding the tag word `Ok` or `Err` (RFC-0050 rule 8). A handler still writes
//! and reads Rust's `Result<T, E>` (RFC-0038), so every case here crosses that
//! boundary in one direction or both.

use acvus_extern::{OneValue, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, InterpreterContext, SequentialExecutor, Value};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};

#[extern_fn(effect = pure)]
fn halve(n: i64) -> Result<i64, String> {
    match n % 2 {
        0 => Ok(n / 2),
        _ => Err(format!("{n} is odd")),
    }
}

#[extern_fn(effect = pure)]
fn describe(r: Result<i64, String>) -> String {
    match r {
        Ok(n) => format!("ok {n}"),
        Err(e) => format!("err {e}"),
    }
}

fn regs() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        fns: [halve, describe],
    });
    regs
}

async fn run(i: &Interner, source: &str, ret: Ty) -> Value {
    run_script_mode_with_externs(i, source, Context::default(), regs(), ret)
        .await
        .value
}

async fn text(i: &Interner, source: &str) -> String {
    let v = run(i, source, Ty::String).await;
    // SAFETY: the script's declared return type is `String`.
    unsafe { v.as_str() }.to_owned()
}

#[tokio::test]
async fn a_script_matches_both_sides_of_a_result_an_extern_built() {
    let i = Interner::new();
    let src = |n: i64| {
        format!(
            r#"let r = t::halve({n});
               if let Ok(v) = r {{ "half " + v.to_string() }}
               else if let Err(e) = r {{ "no " + e }}
               else {{ "unreachable".to_string() }}"#
        )
    };
    assert_eq!(text(&i, &src(10)).await, "half 5");
    assert_eq!(text(&i, &src(7)).await, "no 7 is odd");
}

#[tokio::test]
async fn a_result_a_script_built_reaches_a_handler_as_both_sides() {
    let i = Interner::new();
    assert_eq!(
        text(
            &i,
            r#"t::describe(if true { Ok(4) } else { Err("x".to_string()) })"#
        )
        .await,
        "ok 4"
    );
    assert_eq!(
        text(
            &i,
            r#"t::describe(if false { Ok(4) } else { Err("x".to_string()) })"#
        )
        .await,
        "err x"
    );
}

#[tokio::test]
async fn a_result_round_trips_through_two_externs() {
    let i = Interner::new();
    assert_eq!(text(&i, "t::describe(t::halve(10))").await, "ok 5");
    assert_eq!(text(&i, "t::describe(t::halve(7))").await, "err 7 is odd");
}

// -- The layout the rule rests on --------------------------------------

fn runtime(i: &Interner) -> AcvusRuntime {
    InterpreterContext::new(
        i,
        rustc_hash::FxHashMap::default(),
        std::sync::Arc::new(SequentialExecutor),
    )
    .runtime_over_an_empty_page()
}

fn tag_of(value: &Value) -> Astr {
    // SAFETY: the value came from the crossing below, which erased a variant.
    let variant = unsafe { value.as_variant() };
    // SAFETY: the same witness — a variant's first register is its tag.
    unsafe { variant.tag().as_tag() }
}

#[test]
fn a_crossed_result_is_a_variant_holding_the_tag_the_side_names() {
    let i = Interner::new();
    let rt = runtime(&i);

    let ok = <Result<i64, String> as OneValue<AcvusRuntime>>::erase(Ok(7), &rt);
    assert_eq!(tag_of(&ok), i.intern("Ok"));
    assert_eq!(
        unsafe { <Result<i64, String> as OneValue<AcvusRuntime>>::materialize(&rt, ok) },
        Ok(7)
    );

    let err = <Result<i64, String> as OneValue<AcvusRuntime>>::erase(Err("bad".to_owned()), &rt);
    assert_eq!(tag_of(&err), i.intern("Err"));
    assert_eq!(
        unsafe { <Result<i64, String> as OneValue<AcvusRuntime>>::materialize(&rt, err) },
        Err("bad".to_owned())
    );
}
