//! A `derive(TyArg)` struct declares its fields, and a value of its type
//! has exactly those (RFC-0042). What that is at the contract: an object
//! that lacks one of them, or carries one the struct does not declare, is
//! refused by the field's name, while reading a field of a value of the
//! type is the read it looks like.

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{Effect, Instances, Mutability, ObjectTy, ParamTerm, Poly, TyTerm, TypeArg};
use acvus_mir_test::compile_script_mode_ir_with;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn flags_ty(i: &Interner) -> TyTerm<Poly> {
    TyTerm::Object(ObjectTy::declared(
        i.intern("Flags"),
        [(i.intern("a"), TyTerm::Bool), (i.intern("b"), TyTerm::Bool)]
            .into_iter()
            .collect(),
    ))
}

fn extern_fn(
    i: &Interner,
    name: &str,
    params: Vec<ParamTerm<Poly>>,
    ret: TyTerm<Poly>,
) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Instances::default(),
        },
        ty: TyTerm::Fn {
            params,
            ret: Box::new(ret),
            captures: vec![],
            effect: Effect::PURE.into(),
        },
    }
}

fn compile(i: &Interner, source: &str) -> Result<String, String> {
    let externs = [
        extern_fn(
            i,
            "take_flags",
            vec![ParamTerm::<Poly>::new(i.intern("v"), flags_ty(i))],
            TyTerm::I64,
        ),
        extern_fn(i, "flags", vec![], flags_ty(i)),
        extern_fn(
            i,
            "read_flags",
            vec![ParamTerm::<Poly>::new(
                i.intern("v"),
                TyTerm::Ref(Mutability::Shared, Box::new(TypeArg::uniform(flags_ty(i)))),
            )],
            TyTerm::I64,
        ),
    ];
    compile_script_mode_ir_with(i, source, &FxHashMap::default(), &externs)
}

fn refusal(i: &Interner, source: &str) -> String {
    compile(i, source).expect_err("the declaration fixes the field set")
}

#[test]
fn an_object_lacking_a_declared_field_is_refused_by_its_name() {
    let i = Interner::new();
    let err = refusal(&i, "take_flags({ a: true, })");
    assert!(
        err.contains("object lacks field `b` that `Flags` declares"),
        "{err}"
    );
}

#[test]
fn an_object_carrying_a_field_the_struct_does_not_declare_is_refused_by_its_name() {
    let i = Interner::new();
    let err = refusal(&i, "take_flags({ a: true, b: false, c: true, })");
    assert!(
        err.contains("object has field `c` that `Flags` does not declare"),
        "{err}"
    );
}

#[test]
fn a_binding_that_lacks_a_declared_field_is_refused_where_it_is_passed() {
    let i = Interner::new();
    let err = refusal(&i, "let x = { a: true, }; take_flags(x)");
    assert!(
        err.contains("object lacks field `b` that `Flags` declares"),
        "{err}"
    );
}

#[test]
fn an_object_of_every_declared_field_is_admitted() {
    let i = Interner::new();
    compile(&i, "take_flags({ a: true, b: false, })").expect("the declared field set");
    compile(&i, "let x = { b: false, a: true, }; take_flags(x)").expect("the declared field set");
}

#[test]
fn a_value_of_the_declared_type_is_admitted_and_reads_its_fields() {
    let i = Interner::new();
    compile(&i, "take_flags(flags())").expect("the declared type itself");
    compile(&i, "let f = flags(); if f.a { 1 } else { 0 }").expect("a field the struct declares");
}

#[test]
fn a_field_the_struct_does_not_declare_is_not_on_a_value_of_its_type() {
    let i = Interner::new();
    let err = refusal(&i, "let f = flags(); if f.c { 1 } else { 0 }");
    assert!(err.contains("no field `c`"), "{err}");
}

/// A lambda whose parameter is read for one field is asking the object for
/// it, which is what every value of the declared type answers.
#[test]
fn a_lambda_reading_one_field_of_a_declared_value_is_admitted() {
    let i = Interner::new();
    compile(
        &i,
        "let read = |f| -> f.a; if read(flags()) { 1 } else { 0 }",
    )
    .expect("a read of `a`");
}

/// A message about the type names the struct before its field list, so
/// two objects of the same fields read apart.
#[test]
fn a_declared_object_type_is_printed_under_its_name() {
    let i = Interner::new();
    let err = refusal(&i, "take_flags(1)");
    assert!(err.contains("Flags{a: Bool, b: Bool}"), "{err}");
}

/// A reference is a second name for a storage of the type it refers to,
/// so the field set travels with the referent.
#[test]
fn a_reference_to_an_object_lacking_a_declared_field_is_refused() {
    let i = Interner::new();
    let err = refusal(&i, "let x = { a: true, }; read_flags(&x)");
    assert!(
        err.contains("object lacks field `b` that `Flags` declares"),
        "{err}"
    );
}

#[test]
fn a_reference_to_a_value_of_the_declared_type_is_admitted() {
    let i = Interner::new();
    compile(&i, "let f = flags(); read_flags(&f)").expect("a reference to the declared type");
    compile(&i, "let x = { a: true, b: false, }; read_flags(&x)")
        .expect("a reference to the declared field set");
}
