//! Where a matched scrutinee is dropped. RFC-0039 is the other half of
//! what these listings pin: an option has no storage of its own, and a
//! `Result` has a box that outlives the payload taken out of it.

use acvus_mir::ty::Ty;
use acvus_mir_test::compile_script_optimized;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

fn ctx(i: &Interner, entries: &[(&str, Ty)]) -> FxHashMap<Astr, Ty> {
    entries
        .iter()
        .map(|(name, ty)| (i.intern(name), ty.clone()))
        .collect()
}

fn legend_register_of(ir: &str, name: &str) -> String {
    let legend = format!("({name}) : ");
    ir.lines()
        .find_map(|line| {
            let (reg, _) = line.trim().strip_prefix("; ")?.split_once(' ')?;
            line.contains(&legend).then(|| reg.to_string())
        })
        .unwrap_or_else(|| panic!("no storage named {name} in\n{ir}"))
}

fn body(ir: &str) -> Vec<String> {
    ir.lines()
        .filter_map(|line| line.split_once('|'))
        .map(|(_, code)| code.trim().to_string())
        .collect()
}

fn drops_of(ir: &str, register: &str) -> usize {
    let drop = format!("drop {register}");
    body(ir).iter().filter(|line| **line == drop).count()
}

fn after_the_take_in_its_block(ir: &str, storage: &str) -> Vec<String> {
    let take = format!("take {storage}.payload");
    let lines = body(ir);
    let at = lines
        .iter()
        .position(|line| line.contains(&take))
        .unwrap_or_else(|| panic!("no payload take of {storage} in\n{ir}"));
    lines[at + 1..]
        .iter()
        .take_while(|line| !line.starts_with("jump"))
        .cloned()
        .collect()
}

#[test]
fn an_option_of_a_word_is_never_dropped() {
    let i = Interner::new();
    let ir = compile_script_optimized(
        &i,
        "let o = Some(1.5); Some(v) = o { @out = v; }; 0",
        &ctx(&i, &[("out", Ty::Float)]),
    )
    .unwrap();
    assert!(!ir.contains("drop"), "{ir}");
}

#[test]
fn an_option_of_a_vec_is_dropped_only_where_it_was_not_matched() {
    let i = Interner::new();
    let ir = compile_script_optimized(
        &i,
        "let o = Some(reverse([1, 2, 3])); Some(v) = o { @out = len(&v); }; 0",
        &ctx(&i, &[("out", Ty::Int(acvus_mir::ty::IntTy::U64))]),
    )
    .unwrap();
    let o = legend_register_of(&ir, "o");
    assert_eq!(drops_of(&ir, &o), 1, "{ir}");
    assert!(
        !after_the_take_in_its_block(&ir, "o").contains(&format!("drop {o}")),
        "{ir}"
    );
}

#[test]
fn an_option_of_an_option_of_a_vec_is_dropped_only_where_it_was_not_matched() {
    let i = Interner::new();
    let ir = compile_script_optimized(
        &i,
        "let o = Some(Some(reverse([1, 2, 3]))); Some(Some(v)) = o { @out = len(&v); }; 0",
        &ctx(&i, &[("out", Ty::Int(acvus_mir::ty::IntTy::U64))]),
    )
    .unwrap();
    let o = legend_register_of(&ir, "o");
    assert_eq!(drops_of(&ir, &o), 1, "{ir}");
    assert!(
        !after_the_take_in_its_block(&ir, "o").contains(&format!("drop {o}")),
        "{ir}"
    );
}

#[test]
fn a_result_keeps_its_box_after_its_payload_is_taken() {
    let i = Interner::new();
    let ir = compile_script_optimized(
        &i,
        "let r = decimal(\"1.5\"); Ok(v) = r { @out = 1; }; 0",
        &ctx(&i, &[("out", Ty::Int(acvus_mir::ty::IntTy::U64))]),
    )
    .unwrap();
    let r = legend_register_of(&ir, "r");
    assert!(
        after_the_take_in_its_block(&ir, "r").contains(&format!("drop {r}")),
        "{ir}"
    );
}
