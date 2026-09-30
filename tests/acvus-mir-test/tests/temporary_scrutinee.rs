use acvus_mir::ty::{IntTy, Ty};
use acvus_mir_test::compile_script_ir;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

fn ctx(i: &Interner, entries: &[(&str, Ty)]) -> FxHashMap<Astr, Ty> {
    entries
        .iter()
        .map(|(name, ty)| (i.intern(name), ty.clone()))
        .collect()
}

fn body(ir: &str) -> Vec<String> {
    ir.lines()
        .filter_map(|line| line.split_once('|'))
        .map(|(_, code)| code.trim().to_string())
        .collect()
}

fn operands_of(ir: &str, opcode: &str) -> Vec<String> {
    body(ir)
        .iter()
        .filter_map(|line| {
            let (_, op) = line.split_once(" = ")?;
            let rest = op.strip_prefix(opcode)?.strip_prefix(' ')?;
            Some(rest.split_whitespace().next()?.to_string())
        })
        .collect()
}

fn destinations_of(ir: &str, opcode: &str) -> Vec<String> {
    body(ir)
        .iter()
        .filter_map(|line| {
            let (dst, op) = line.split_once(" = ")?;
            op.starts_with(opcode).then(|| dst.trim().to_string())
        })
        .collect()
}

#[test]
fn an_if_let_on_a_call_tests_and_unwraps_the_register_the_call_wrote() {
    let i = Interner::new();
    let ir = compile_script_ir(
        &i,
        "let s = \"abc\"; if let Some(v) = find(&s, \"b\") { @out = v; }; 0",
        &ctx(&i, &[("out", Ty::Int(IntTy::I64))]),
    )
    .unwrap();
    assert!(!ir.contains("$source"), "{ir}");
    let called = destinations_of(&ir, "call");
    assert_eq!(called.len(), 1, "{ir}");
    assert_eq!(operands_of(&ir, "test"), called, "{ir}");
    assert_eq!(operands_of(&ir, "unwrap"), called, "{ir}");
}

#[test]
fn a_while_let_head_stores_nothing() {
    let i = Interner::new();
    let ir = compile_script_ir(
        &i,
        "let v = reverse([1, 2, 3]); let it = as_iter(&v); let acc = 0; while let Some(x) = next(&mut it) { acc = acc + *x; } acc",
        &ctx(&i, &[]),
    )
    .unwrap();
    assert!(!ir.contains("$source"), "{ir}");
    assert_eq!(destinations_of(&ir, "unwrap").len(), 1, "{ir}");
}

#[test]
fn a_result_payload_is_taken_once_from_a_temporary() {
    let i = Interner::new();
    let ir = compile_script_ir(
        &i,
        "if let Ok(c) = int_to_char(65) { @out = c.to_string(); }; 0",
        &ctx(&i, &[("out", Ty::String)]),
    )
    .unwrap();
    assert!(!ir.contains("$source"), "{ir}");
    assert_eq!(destinations_of(&ir, "unwrap").len(), 1, "{ir}");
}

#[test]
fn a_pattern_whose_test_reads_a_part_keeps_a_place() {
    let i = Interner::new();
    let ir = compile_script_ir(
        &i,
        "if let Some(Some(v)) = Some(Some(3)) { @out = v; }; 0",
        &ctx(&i, &[("out", Ty::Int(IntTy::I64))]),
    )
    .unwrap();
    assert!(ir.contains("$source"), "{ir}");
}
