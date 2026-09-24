//! RFC-0063: an `if` is a terminator.

use acvus_mir::ty::Ty;
use acvus_mir_test::{compile_script_mode_raw, compile_script_optimized};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

const DIAMOND: &str = " if ";
const JUMP_IF: &str = "jump_if ";

fn ctx(i: &Interner, entries: &[(&str, Ty)]) -> FxHashMap<Astr, Ty> {
    entries
        .iter()
        .map(|(name, ty)| (i.intern(name), ty.clone()))
        .collect()
}

fn flag_and_bound(i: &Interner) -> FxHashMap<Astr, Ty> {
    ctx(i, &[("c", Ty::Bool), ("n", Ty::I64)])
}

fn raw(source: &str) -> String {
    let i = Interner::new();
    compile_script_mode_raw(&i, source, &flag_and_bound(&i)).unwrap()
}

fn diamond_lines(ir: &str) -> Vec<&str> {
    ir.lines()
        .filter_map(|line| line.split_once('|'))
        .map(|(_, code)| code.trim())
        .filter(|code| code.starts_with("if "))
        .collect()
}

#[test]
fn an_if_with_an_else_names_the_block_both_arms_rejoin_at() {
    let ir = raw("if @c { 1 } else { 2 }");
    assert_eq!(diamond_lines(&ir), ["if r1 -> L0 else L2 join L1"], "{ir}");
    assert!(!ir.contains(JUMP_IF), "{ir}");
}

#[test]
fn an_if_with_no_else_joins_where_its_else_edge_already_went() {
    let ir = raw("let x = 0; if @c { x = 1; }; x");
    assert_eq!(diamond_lines(&ir), ["if r2 -> L0 else L1 join L1"], "{ir}");
}

#[test]
fn a_while_test_stays_a_jump_if() {
    let ir = raw("let i = 0; while i < @n { i = i + 1; } i");
    assert!(ir.contains(JUMP_IF), "{ir}");
    assert!(diamond_lines(&ir).is_empty(), "{ir}");
}

#[test]
fn a_while_let_test_stays_a_jump_if() {
    let i = Interner::new();
    let ir = compile_script_mode_raw(
        &i,
        "let opt = @o; while let Some(v) = opt { opt = None; } 0",
        &ctx(&i, &[("o", Ty::Option(Box::new(Ty::I64)))]),
    )
    .unwrap();
    assert!(ir.contains(JUMP_IF), "{ir}");
}

/// RFC-0063 rule 2 states the correction this test pins: rule 1
/// had listed `?` among the branches that rejoin by construction.
#[test]
fn a_try_stays_a_jump_if_because_its_failure_arm_returns() {
    let i = Interner::new();
    let ir = compile_script_mode_raw(
        &i,
        "let v = @r?; Ok(v + 1)",
        &ctx(
            &i,
            &[("r", Ty::Result(Box::new(Ty::I64), Box::new(Ty::String)))],
        ),
    )
    .unwrap();
    assert!(ir.contains(JUMP_IF), "{ir}");
    assert!(
        !ir.contains(DIAMOND),
        "no branch in this body rejoins: {ir}"
    );
}

#[test]
fn an_if_whose_arm_breaks_the_enclosing_loop_stays_a_jump_if() {
    let ir = raw("let i = 0; while i < @n { if @c { break; }; i = i + 1; } i");
    assert!(diamond_lines(&ir).is_empty(), "{ir}");
}

#[test]
fn an_if_whose_arm_continues_the_enclosing_loop_stays_a_jump_if() {
    let ir = raw("let i = 0; while i < @n { i = i + 1; if @c { continue; }; i = i + 1; } i");
    assert!(diamond_lines(&ir).is_empty(), "{ir}");
}

/// The cheaper rule -- "an arm holds a `break`" -- was rejected for the
/// reachability test this pins, and nothing below would fail under it.
#[test]
fn an_if_whose_arm_breaks_a_loop_of_its_own_is_still_a_diamond() {
    let ir = raw("let i = 0; if @c { while i < @n { break; } }; i");
    assert_eq!(diamond_lines(&ir).len(), 1, "{ir}");
}

fn optimized(source: &str) -> String {
    let i = Interner::new();
    compile_script_optimized(&i, source, &ctx(&i, &[("n", Ty::I64)])).unwrap()
}

/// The same program as `acvus-mir-test/tests/sroa.rs`'s `enum_match`, whose
/// snapshot pins the whole body; change one and the other must move with it.
#[test]
fn an_if_whose_arms_a_pass_scattered_and_another_rejoined_is_a_diamond_again() {
    let ir = optimized(
        "let acc = 0; let i = 0; while i < @n { \
         let e = if i % 2 == 0 { E::A(i) } else { E::B(i + 1) }; \
         match e { E::A(v) => { acc = acc + v; }, E::B(v) => { acc = acc + v; } }; \
         i = i + 1; } acc",
    );
    assert_eq!(diamond_lines(&ir), ["if r7 -> L3 else L5 join L6"], "{ir}");
    assert_eq!(
        ir.matches(JUMP_IF).count(),
        0,
        "the `while` test is a `for` terminator (RFC-0081), and no other \
         branch is a `jump_if`: {ir}"
    );
}
