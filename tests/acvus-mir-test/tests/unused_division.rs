//! An integer `/` or `%` whose value nothing reads stays where its divisor
//! may be zero or, at a signed width, its operands may be `MIN` and `-1`
//! (RFC-0037 rule 2, RFC-0048 rule 8), and goes where the interval domain
//! rules both out. An overflowing `+` and a float `/` whose values nothing
//! reads go at every level (RFC-0037 rule 3).

use acvus_mir::graph::optimize::Opt;
use acvus_mir::ir::{BinOp, InstKind};
use acvus_mir::printer::dump_with;
use acvus_mir::ty::Ty;
use acvus_mir_test::compile_script_module_at;
use acvus_utils::Interner;

fn held_ops_and_listing(source: &str, opt: Opt, op: fn(&BinOp) -> bool) -> (usize, String) {
    let i = Interner::new();
    let contexts = [
        (i.intern("x"), Ty::I64),
        (i.intern("y"), Ty::I64),
        (i.intern("f"), Ty::Float),
    ]
    .into_iter()
    .collect();
    let module = compile_script_module_at(&i, source, &contexts, opt)
        .unwrap_or_else(|e| panic!("{source}\n{e}"));
    let held = module
        .main
        .insts
        .iter()
        .filter(|inst| matches!(&inst.kind, InstKind::BinOp { op: held, .. } if op(held)))
        .count();
    (held, dump_with(&i, &module))
}

fn divides(op: &BinOp) -> bool {
    matches!(op, BinOp::Div | BinOp::Mod)
}

fn adds(op: &BinOp) -> bool {
    matches!(op, BinOp::Add(_))
}

fn stays(source: &str) {
    for opt in [Opt::None, Opt::Full] {
        let (held, listing) = held_ops_and_listing(source, opt, divides);
        assert_eq!(held, 1, "at {opt:?}, the unused division stays: {source}\n{listing}");
    }
}

fn goes(source: &str, op: fn(&BinOp) -> bool) {
    for opt in [Opt::None, Opt::Full] {
        let (held, listing) = held_ops_and_listing(source, opt, op);
        assert_eq!(held, 0, "at {opt:?}, the unused operation is gone: {source}\n{listing}");
    }
}

#[test]
fn an_unused_division_by_zero_stays() {
    stays("let d = 100 / 0; 5");
    stays("let z = 0; let d = 100 / z; 5");
}

#[test]
fn an_unused_remainder_by_zero_stays() {
    stays("let d = 100 % 0; 5");
}

#[test]
fn an_unused_division_by_a_divisor_that_may_be_zero_stays() {
    stays("let d = @x / @y; 5");
    stays("let d = @x % @y; 5");
}

#[test]
fn an_unused_signed_division_by_minus_one_stays() {
    stays("let d = @x / -1; 5");
    stays("let d = @x % -1; 5");
}

#[test]
fn an_unused_division_the_intervals_prove_is_gone() {
    goes("let d = @x / 2; 5", divides);
    goes("let d = @x % 3; 5", divides);
    goes("for i in 1..3 { let d = 10 / i; } 7", divides);
}

#[test]
fn an_unused_float_division_is_gone() {
    goes("let d = @f / 0.0; 5", divides);
    goes("let d = @f % 0.0; 5", divides);
}

#[test]
fn an_unused_overflowing_addition_is_still_gone() {
    goes("let d = @x + 9223372036854775807; 5", adds);
}
