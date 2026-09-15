//! Intent tests for RFC-0024.

use acvus_mir::ty::{LenTerm, Ty};
use acvus_mir_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn items(i: &Interner) -> FxHashMap<acvus_utils::Astr, Ty> {
    FxHashMap::from_iter([(
        i.intern("items"),
        Ty::Array(Box::new(Ty::Int), LenTerm::Known(3)),
    )])
}

fn user(i: &Interner) -> FxHashMap<acvus_utils::Astr, Ty> {
    FxHashMap::from_iter([(
        i.intern("user"),
        Ty::Object(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::Int),
        ])),
    )])
}

#[test]
fn a_list_pattern_against_a_reference_binds_element_references() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "[a, b, ..] = &@items { x = *a + *b; }; 0", &items(&i)).unwrap();
    assert!(ir.contains("ref &(*r"), "{ir}");
    assert!(ir.contains("[0]"), "{ir}");
    assert!(ir.contains("commit @items"), "{ir}");
}

#[test]
fn a_word_binding_through_a_reference_is_not_the_word() {
    let i = Interner::new();
    let err = compile_script_ir(&i, "[a, b, ..] = &@items { x = a + 1; }; 0", &items(&i)).unwrap_err();
    assert!(!err.is_empty(), "{err}");
}

#[test]
fn an_object_pattern_against_a_reference_binds_field_references() {
    let i = Interner::new();
    let ir = compile_to_ir(&i, "{{ { name, } = &@user }}{{ name }}{{/}}", &user(&i)).unwrap();
    assert!(ir.contains(".name"), "{ir}");
    assert!(ir.contains("string_concat"), "{ir}");
    assert!(ir.contains("commit @user"), "{ir}");
}

#[test]
fn a_literal_pattern_compares_through_the_reference() {
    let i = Interner::new();
    let role = FxHashMap::from_iter([(i.intern("role"), Ty::String)]);
    let ir = compile_to_ir(&i, r#"{{ "admin" = &@role }}yes{{_}}no{{/}}"#, &role).unwrap();
    assert!(ir.contains("test_literal") || ir.contains("test "), "{ir}");
    assert!(ir.contains("commit @role"), "{ir}");
}

#[test]
fn a_list_pattern_against_a_value_copies_its_words_out() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "[a, b, ..] = @items { x = a + b; }; 0", &items(&i)).unwrap();
    assert!(ir.contains("commit @items"), "{ir}");
}

#[test]
fn a_list_pattern_against_a_value_of_objects_leaves_it_partly_moved() {
    let i = Interner::new();
    let user = Ty::Object(FxHashMap::from_iter([(i.intern("age"), Ty::Int)]));
    let users = FxHashMap::from_iter([(
        i.intern("users"),
        Ty::Array(Box::new(user), LenTerm::Known(2)),
    )]);
    let err = compile_script_ir(&i, "[a, b] = @users { x = 1; }; 0", &users).unwrap_err();
    assert!(err.contains("UseAfterMove"), "{err}");
    let ir = compile_script_ir(&i, "[a, b] = &@users { x = a.age; }; 0", &users).unwrap();
    assert!(ir.contains("commit @users"), "{ir}");
}

#[test]
fn a_context_bind_through_a_reference_is_rejected() {
    let i = Interner::new();
    let ctx = FxHashMap::from_iter([
        (i.intern("user"), Ty::Object(FxHashMap::from_iter([(i.intern("age"), Ty::Int)]))),
        (i.intern("copy"), Ty::Object(FxHashMap::from_iter([(i.intern("age"), Ty::Int)]))),
    ]);
    let err = compile_script_ir(&i, "@copy = &@user; 0", &ctx).unwrap_err();
    assert!(!err.is_empty(), "{err}");
}
