use acvus_mir::ty::Ty;
use acvus_mir_test::{compile_script_mode_raw, compile_to_ir};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn ctx(i: &Interner, entries: &[(&str, Ty)]) -> FxHashMap<acvus_utils::Astr, Ty> {
    entries
        .iter()
        .map(|(name, ty)| (i.intern(name), ty.clone()))
        .collect()
}

#[test]
fn a_question_mark_in_a_template_is_refused() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[("r", Ty::Result(Box::new(Ty::I64), Box::new(Ty::String)))],
    );
    let err = compile_to_ir(&i, "{{ x = @r? }}{{ x.to_string() }}", &c).unwrap_err();
    assert!(err.contains("`?` needs a function to return from"), "{err}");
}

#[test]
fn a_question_mark_on_a_non_result_is_refused() {
    let i = Interner::new();
    let c = ctx(&i, &[("n", Ty::I64)]);
    let err = compile_script_mode_raw(&i, "let v = @n?; Ok(v)", &c).unwrap_err();
    assert!(
        err.contains("`?` takes a Result or an Option, not i64"),
        "{err}"
    );
}

#[test]
fn a_question_mark_s_error_must_be_the_function_s() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[("r", Ty::Result(Box::new(Ty::I64), Box::new(Ty::String)))],
    );
    let err = compile_script_mode_raw(
        &i,
        "let v = if let Ok(v) = @r { v } else { 0 }; let w = @r?; w",
        &c,
    )
    .unwrap_err();
    assert!(
        err.contains("`?` leaves with") || err.contains("type mismatch"),
        "{err}"
    );
}
