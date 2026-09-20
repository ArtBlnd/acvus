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
    let ir =
        compile_script_ir(&i, "let x = clone(&@role); x", &string_context(&i, "role")).unwrap();
    assert!(ir.contains("string_clone"), "{ir}");
    assert!(!ir.contains("call"), "{ir}");
}

// -- An operator's operand is a number (RFC-0020, RFC-0043) -----------------

mod fx_vec {
    use acvus_extern::{Registry, TypesOnly, Var, extern_fn, extern_registry, kind};

    #[extern_fn(effect = pure)]
    pub fn consume<T>(v: Vec<T>) -> i64
    where
        T: Var<kind::Type>,
    {
        let _ = v;
        unreachable!("a type-only fixture is never run")
    }

    pub fn registry() -> Registry<TypesOnly> {
        extern_registry! {
            ns: "fx_vec",
            fns: [consume],
        }
    }
}

mod fx_arr {
    use acvus_extern::{Arr, Registry, TypesOnly, Var, extern_fn, extern_registry, kind};

    #[extern_fn(effect = pure)]
    pub fn consume<T, N>(v: Arr<T, N>) -> i64
    where
        T: Var<kind::Type>,
        N: Var<kind::Length>,
    {
        let _ = v;
        unreachable!("a type-only fixture is never run")
    }

    pub fn registry() -> Registry<TypesOnly> {
        extern_registry! {
            ns: "fx_arr",
            fns: [consume],
        }
    }
}

fn script(i: &Interner, source: &str) -> Result<String, String> {
    compile_script_mode_ir_with(i, source, &FxHashMap::default(), &[])
}

/// RFC-0043: the receiver of `x.consume()` is admitted as a set.
fn script_with_value_overloads(i: &Interner, source: &str) -> Result<String, String> {
    let acvus_extern::Externs { functions, .. } =
        acvus_extern::Externs::combine(vec![fx_vec::registry(), fx_arr::registry()], i)
            .expect("the fixtures combine");
    compile_script_mode_ir_with(i, source, &FxHashMap::default(), &functions)
}

#[test]
fn an_array_never_reaches_a_multiplication_through_an_open_operand() {
    let i = Interner::new();
    let err = script(&i, "let f = |k| -> k * k; f([1.0, 2.0])").unwrap_err();
    assert!(
        err.contains(
            "type Array<Float, 2> is outside the declared bound \
             one of i8, i16, i32, i64, u8, u16, u32, u64, Float"
        ),
        "{err}"
    );
}

#[test]
fn an_array_never_reaches_a_comparison_through_an_open_operand() {
    let i = Interner::new();
    let err = script(&i, "let f = |k, m| -> k < m; f([1.0], [2.0])").unwrap_err();
    assert!(
        err.contains(
            "type Array<Float, 1> is outside the declared bound \
             one of i8, i16, i32, i64, u8, u16, u32, u64, Float"
        ),
        "{err}"
    );
}

#[test]
fn an_integer_literal_narrows_the_operator_bound_to_the_integer_widths() {
    let i = Interner::new();
    let ir = script(&i, "let f = |k| -> k + 7; f(1)").unwrap();
    assert!(ir.contains("Fn(i64) -> i64"), "{ir}");
}

#[test]
fn an_array_against_an_integer_literal_operand_is_refused_as_a_mismatch() {
    let i = Interner::new();
    let err = script(&i, "let f = |k| -> k + 7; f([1.0, 2.0])").unwrap_err();
    assert!(
        err.contains("type mismatch: expected i64, got Array<Float, 2>"),
        "{err}"
    );
}

#[test]
fn a_string_against_an_integer_literal_operand_is_refused_as_a_mismatch() {
    let i = Interner::new();
    let err = script(&i, "let f = |k| -> k < 7; f(\"a\".to_string())").unwrap_err();
    assert!(
        err.contains("type mismatch: expected i64, got String"),
        "{err}"
    );
}

#[test]
fn a_float_literal_operand_refuses_an_integer_literal_argument() {
    let i = Interner::new();
    let err = script(&i, "let f = |k| -> k + 7.0; f(1)").unwrap_err();
    assert!(
        err.contains("type mismatch: expected Float, got i64"),
        "{err}"
    );
    let ir = script(&i, "let f = |k| -> k + 7.0; f(1.0)").unwrap();
    assert!(ir.contains("Fn(Float) -> Float"), "{ir}");
}

// -- A word operator reads a written `&word` operand through (RFC-0018) ----

#[test]
fn a_reference_the_program_wrote_is_read_through_before_a_multiplication() {
    let i = Interner::new();
    let ir = script(&i, "let x = 3; let r = &x; r * 2").unwrap();
    assert!(ir.contains("take (*"), "{ir}");
    assert!(ir.contains(" * "), "{ir}");
}

#[test]
fn a_captured_word_is_copied_once_however_often_the_body_reads_it() {
    let i = Interner::new();
    let ir = script(&i, "let k = 1; let f = |y| -> k * y + k; f(2)").unwrap();
    assert_eq!(ir.matches("take (*").count(), 1, "{ir}");
    assert!(ir.contains(" * "), "{ir}");
    assert!(ir.contains(" + "), "{ir}");
}

#[test]
fn a_captured_word_compared_is_the_word_itself() {
    let i = Interner::new();
    let ir = script(&i, "let k = 1; let f = |y| -> k < y; f(2)").unwrap();
    assert_eq!(ir.matches("take (*").count(), 1, "{ir}");
}

#[test]
fn a_captured_string_is_still_concatenated_through_the_reference() {
    let i = Interner::new();
    let ir = script(
        &i,
        "let s = \"a\".to_string(); let f = |t| -> s + t; f(\"b\".to_string())",
    )
    .unwrap();
    assert!(ir.contains("string_concat"), "{ir}");
    assert!(!ir.contains("take (*"), "{ir}");
}

// -- The refusal is applied by the mode taken (RFC-0018, RFC-0043) --------

#[test]
fn a_value_mode_receiver_does_not_move_a_large_out_of_a_reference() {
    let i = Interner::new();
    let err =
        script_with_value_overloads(&i, "let o = { v: vec([1]), }; let r = &o; r.v.consume()")
            .unwrap_err();
    assert!(
        err.contains("reads only a primitive")
            && err.contains("used through the reference or cloned"),
        "{err}"
    );
}

#[test]
fn a_value_mode_receiver_of_an_owned_place_is_moved_as_before() {
    let i = Interner::new();
    let ir = script_with_value_overloads(&i, "let o = { v: vec([1]), }; o.v.consume()").unwrap();
    assert!(ir.contains("take o.v"), "{ir}");
}

// -- A literal operand fixes the parameter it meets (RFC-0020, RFC-0037) --

#[test]
fn a_string_literal_operand_fixes_the_parameter_against_an_integer_call() {
    let i = Interner::new();
    let err = script(&i, "let f = |k| -> k + \"a\".to_string(); f(1)").unwrap_err();
    assert!(
        err.contains("type mismatch: expected String, got i64"),
        "{err}"
    );
    assert!(
        err.starts_with("[infer:"),
        "the checker refuses the call, so validate is never reached: {err}"
    );
}

#[test]
fn a_string_literal_operand_leaves_the_parameter_taking_a_string() {
    let i = Interner::new();
    let ir = script(
        &i,
        "let f = |k| -> k + \"a\".to_string(); f(\"b\".to_string())",
    )
    .unwrap();
    assert!(ir.contains("Fn(String) -> String"), "{ir}");
    assert!(ir.contains("string_concat"), "{ir}");
}

#[test]
fn an_integer_literal_operand_refuses_a_float_call() {
    let i = Interner::new();
    let err = script(&i, "let g = |k| -> k + 1; g(2.5)").unwrap_err();
    assert!(
        err.contains("type mismatch: expected i64, got Float"),
        "{err}"
    );
}

#[test]
fn a_parameter_the_body_leaves_open_takes_one_width_for_every_call() {
    let i = Interner::new();
    let c = FxHashMap::from_iter([(i.intern("b"), Ty::U8), (i.intern("n"), Ty::I64)]);
    let ir = compile_script_mode_ir_with(&i, "let g = |k| -> k + 1; g(@b)", &c, &[]).unwrap();
    assert!(ir.contains("Fn(u8) -> u8"), "{ir}");
    let err =
        compile_script_mode_ir_with(&i, "let g = |k| -> k + 1; g(@b); g(@n)", &c, &[]).unwrap_err();
    assert!(err.contains("type mismatch: expected u8, got i64"), "{err}");
}
