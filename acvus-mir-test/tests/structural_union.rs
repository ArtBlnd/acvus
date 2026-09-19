//! Structural types join by union (solver.md R1), and what a union that
//! cannot be recorded means at the contract: an object's missing field is
//! the definite-assignment check's error at the read or call, an enum's
//! missing variant is the type checker's, and a pattern may be narrower
//! than its source.

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{Effect, Instances, ObjectTy, ParamTerm, Poly, TyTerm};
use acvus_mir_test::compile_script_mode_ir_with;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn extern_taking(i: &Interner, name: &str, param: TyTerm<Poly>) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Instances::default(),
        },
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(i.intern("v"), param)],
            ret: Box::new(TyTerm::I64),
            captures: vec![],
            effect: Effect::PURE.into(),
        },
    }
}

fn object_ab(i: &Interner) -> TyTerm<Poly> {
    TyTerm::Object(ObjectTy::written(
        [(i.intern("a"), TyTerm::I64), (i.intern("b"), TyTerm::I64)]
            .into_iter()
            .collect(),
    ))
}

fn shape(i: &Interner, variants: &[&str]) -> TyTerm<Poly> {
    TyTerm::Enum {
        name: i.intern("Shape"),
        variants: variants
            .iter()
            .map(|v| (i.intern(v), Some(Box::new(TyTerm::I64))))
            .collect(),
    }
}

fn compile(i: &Interner, source: &str) -> Result<String, String> {
    let externs = [
        extern_taking(i, "fab", object_ab(i)),
        extern_taking(i, "g_circle", shape(i, &["Circle"])),
        extern_taking(i, "g_both", shape(i, &["Circle", "Square"])),
    ];
    compile_script_mode_ir_with(i, source, &FxHashMap::default(), &externs)
}

fn uninit_field(result: Result<String, String>, field: &str) -> String {
    let err = result.expect_err("the definite-assignment check reports the field");
    assert!(
        err.contains(&format!("has no `{field}` stored on every path")),
        "{err}"
    );
    err
}

#[test]
fn a_field_stored_on_one_path_is_uninitialized_on_the_other() {
    let i = Interner::new();
    let err = uninit_field(
        compile(
            &i,
            "let x = { a: 1, }; let y = if true { x.b = 0; fab(x) } else { fab(x) }; y",
        ),
        "b",
    );
    assert!(err.contains("[62..68]"), "reported at the else call: {err}");
}

#[test]
fn a_field_the_callee_requires_and_the_body_never_names_is_checked() {
    let i = Interner::new();
    uninit_field(compile(&i, "let x = { a: 1, }; fab(x)"), "b");
}

#[test]
fn a_value_built_in_place_is_checked_at_the_call() {
    let i = Interner::new();
    let err = uninit_field(compile(&i, "fab({ a: 1, })"), "b");
    assert!(err.contains("this value has no"), "{err}");
}

#[test]
fn a_read_of_a_field_never_stored_is_reported() {
    let i = Interner::new();
    uninit_field(compile(&i, "let x = { a: 1, }; x.b"), "b");
}

#[test]
fn a_field_stored_on_every_path_passes() {
    let i = Interner::new();
    compile(&i, "let x = { a: 1, }; x.b = 0; fab(x)").expect("initialized");
}

#[test]
fn an_enum_value_that_may_carry_a_variant_the_callee_lacks_is_a_type_error() {
    let i = Interner::new();
    let err = compile(
        &i,
        "let s = if true { Shape::Circle(1) } else { Shape::Square(2) }; g_circle(s)",
    )
    .expect_err("the extern cannot grow its type");
    assert!(err.contains("[infer:test] type mismatch"), "{err}");
}

#[test]
fn an_enum_value_narrower_than_the_callee_s_type_passes() {
    let i = Interner::new();
    compile(&i, "g_both(Shape::Circle(1))").expect("a subset of variants");
    compile(
        &i,
        "let s = if true { Shape::Circle(1) } else { Shape::Square(2) }; g_both(s)",
    )
    .expect("the same variants");
}
