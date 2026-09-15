//! Intent tests for RFC-0020.

use acvus_mir::ty::Ty;
use acvus_mir_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn string_context(i: &Interner, name: &str) -> FxHashMap<acvus_utils::Astr, Ty> {
    FxHashMap::from_iter([(i.intern(name), Ty::String)])
}

#[test]
fn equality_on_primitives_is_a_word_operation() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "1 == 2", &FxHashMap::default()).unwrap();
    assert!(!ir.contains("call"), "{ir}");
}

#[test]
fn equality_on_strings_is_a_string_eq_with_both_operands_lent() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, r#"@role == "admin""#, &string_context(&i, "role")).unwrap();
    assert!(ir.contains("ref &@role"), "{ir}");
    assert!(ir.contains("string_eq"), "{ir}");
    assert!(!ir.contains("call"), "{ir}");
    assert!(ir.contains("commit @role"), "{ir}");
}

#[test]
fn inequality_on_strings_is_the_negated_string_eq() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, r#"@role != "admin""#, &string_context(&i, "role")).unwrap();
    assert!(ir.contains("string_eq"), "{ir}");
    assert!(ir.contains("!"), "{ir}");
}

#[test]
fn a_string_has_no_instance_of_eq() {
    let i = Interner::new();
    let err = compile_script_ir(&i, "eq(&@role, &@role)", &string_context(&i, "role")).unwrap_err();
    assert!(!err.is_empty(), "{err}");
}

#[test]
fn a_primitive_has_no_instance_of_eq() {
    let i = Interner::new();
    let err = compile_script_ir(&i, "eq(&1, &2)", &FxHashMap::default()).unwrap_err();
    assert!(!err.is_empty(), "{err}");
}

#[test]
fn string_addition_is_a_string_concat_of_lent_operands() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, r#""a" + "b""#, &FxHashMap::default()).unwrap();
    assert!(ir.contains("string_concat"), "{ir}");
    assert!(!ir.contains("call"), "{ir}");
}

#[test]
fn a_template_joins_its_parts_with_a_string_concat() {
    let i = Interner::new();
    let ir = compile_to_ir(&i, "Hello, {{ &@role }}!", &string_context(&i, "role")).unwrap();
    assert!(ir.contains("string_concat"), "{ir}");
    assert!(ir.contains("ref &@role"), "{ir}");
}

#[test]
fn string_concatenation_is_a_named_function() {
    let i = Interner::new();
    compile_script_ir(&i, r#"concat(&@role, &@role)"#, &string_context(&i, "role")).unwrap();
}

#[test]
fn clone_of_a_string_is_a_string_clone() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "x = clone(&@role); x", &string_context(&i, "role")).unwrap();
    assert!(ir.contains("string_clone"), "{ir}");
    assert!(!ir.contains("call"), "{ir}");
}
