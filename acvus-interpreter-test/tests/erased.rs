//! `Erased<R, String>` inside a running script (E2, E3): an element edited
//! through `as_mut` is what the program reads afterwards, and equality over
//! an `Iter` of it goes through `as_ref` with no `Monomorphize` member list.

use acvus_ext::Iter;
use acvus_extern::{EffectVar, Erased, IdentityVar, Registry, Runtime, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::*;
use acvus_utils::Interner;

#[extern_fn(effect = pure)]
fn upcase_first<Rt>(rt: &Rt, items: &mut Vec<Erased<Rt, String>>)
where
    Rt: Runtime,
{
    let Some(first) = items.first_mut() else {
        return;
    };
    first.as_mut(rt).make_ascii_uppercase();
}

#[extern_fn(effect = E)]
async fn contains_erased<E, I, Rt>(
    rt: &Rt,
    mut it: Iter<Erased<Rt, String>, E, I, Rt>,
    needle: Erased<Rt, String>,
) -> Result<bool, Rt::Error>
where
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    while let Some(item) = it.next(rt).await? {
        if item.as_ref(rt) == needle.as_ref(rt) {
            return Ok(true);
        }
    }
    Ok(false)
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
