//! RFC-0020: `&&` and `||` lower to the branch `if`/`else` already lowers
//! to — a diamond whose value is the join block's parameter — so no
//! instruction carries the two connectives any more.

use acvus_mir::ty::Ty;
use acvus_mir_test::compile_script_mode_optimized;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

fn ctx(i: &Interner, entries: &[(&str, Ty)]) -> FxHashMap<Astr, Ty> {
    entries
        .iter()
        .map(|(name, ty)| (i.intern(name), ty.clone()))
        .collect()
}

fn bools(i: &Interner) -> FxHashMap<Astr, Ty> {
    ctx(i, &[("a", Ty::Bool), ("b", Ty::Bool)])
}

#[test]
fn and_lowers_to_a_diamond() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(&i, "let c = @a && @b; c", &bools(&i)).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn or_lowers_to_a_diamond() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(&i, "let c = @a || @b; c", &bools(&i)).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn a_compound_loop_condition_branches_inside_the_head() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(
        &i,
        "let n = 0; while n < @max && @a { n = n + 1; } n",
        &ctx(&i, &[("max", Ty::I64), ("a", Ty::Bool)]),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn no_instruction_carries_a_connective() {
    let i = Interner::new();
    for source in [
        "let c = @a && @b; c",
        "let c = @a || @b; c",
        "let c = @a && (@b || @a); c",
    ] {
        let ir = compile_script_mode_optimized(&i, source, &bools(&i)).unwrap();
        assert!(!ir.contains(" && "), "{source} lowered to:\n{ir}");
        assert!(!ir.contains(" || "), "{source} lowered to:\n{ir}");
    }
}
