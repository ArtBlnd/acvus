use acvus_extern::{Never, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

#[extern_fn(effect = pure)]
fn boom(message: String) -> Never {
    panic!("boom: {message}")
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        fns: [boom],
    });
    regs
}

fn flag(i: &Interner, b: bool) -> Context {
    [(i.intern("c"), typed(Ty::Bool, Value::bool_(b)))]
        .into_iter()
        .collect()
}

#[tokio::test]
async fn a_branch_that_panics_leaves_the_other_branch_s_type() {
    let i = Interner::new();
    let v = run_script_mode_with_externs(
        &i,
        r#"if @c { boom("no".to_string()) } else { 41 }"#,
        flag(&i, false),
        registries(),
        Ty::I64,
    )
    .await
    .value;
    assert_eq!(v.as_int(), 41);
}

#[tokio::test]
#[should_panic(expected = "boom: no")]
async fn a_panic_stops_the_run_with_its_message() {
    let i = Interner::new();
    run_script_mode_with_externs(
        &i,
        r#"if @c { boom("no".to_string()) } else { 41 }"#,
        flag(&i, true),
        registries(),
        Ty::I64,
    )
    .await;
}

#[tokio::test]
async fn a_move_on_the_panicking_path_does_not_reach_the_code_after() {
    let i = Interner::new();
    let v = run_script_mode_with_externs(
        &i,
        r#"let s = "kept".to_string(); if @c { let t = s; boom(t) } else { 0 }; len(&s)"#,
        flag(&i, false),
        registries(),
        Ty::U64,
    )
    .await
    .value;
    assert_eq!(v.as_int(), 4);
}
