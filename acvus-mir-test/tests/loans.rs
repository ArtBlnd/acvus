//! Intent tests for `analysis::loans`: a call that is lent a place reads
//! or writes it, and every pass that moves, removes, or allocates around
//! storage sees that (RFC-0015, RFC-0018).

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{Effect, Mutability, ParamTerm, Poly, Ty, TyTerm, TypeArg, lift_to_poly};
use acvus_mir_test::*;
use acvus_utils::Interner;

fn bump(i: &Interner, effect: Effect) -> Function {
    let params = [
        (
            "n",
            Ty::Ref(Mutability::Mut, Box::new(TypeArg::uniform(Ty::I64))),
        ),
        ("by", Ty::I64),
    ];
    Function {
        qref: QualifiedRef::root(i.intern("bump")),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Default::default(),
        },
        ty: TyTerm::Fn {
            params: params
                .iter()
                .map(|(n, ty)| ParamTerm::<Poly>::new(i.intern(n), lift_to_poly(ty)))
                .collect(),
            ret: Box::new(lift_to_poly(&Ty::I64)),
            captures: vec![],
            effect: effect.into(),
        },
    }
}

fn optimized(i: &Interner, src: &str, effect: Effect) -> String {
    compile_multi_fn_optimized(i, ("main", src), &[], &[], &[bump(i, effect)]).unwrap()
}

fn main_body(ir: &str) -> &str {
    ir.split("=== main ===")
        .nth(1)
        .unwrap()
        .split("=== ")
        .next()
        .unwrap()
}

#[test]
fn a_pure_call_lent_a_place_mutably_is_kept_when_the_place_is_read_after() {
    let i = Interner::new();
    let ir = optimized(
        &i,
        "x = 1; bump(&mut x, 2); bump(&mut x, 3); x",
        Effect::PURE,
    );
    let main = main_body(&ir);
    assert_eq!(main.matches("call #0").count(), 2, "{ir}");
    assert!(main.contains("take "), "{ir}");
}

#[test]
fn a_lent_place_and_the_reference_to_it_take_different_registers() {
    let i = Interner::new();
    let ir = optimized(&i, "x = 40; y = bump(&mut x, 2); x + y", Effect::PURE);
    let main = main_body(&ir);
    let reference = main
        .lines()
        .find(|l| l.contains("ref &mut"))
        .expect("the lend");
    let register = reference.split('=').next().unwrap().trim();
    let slot = reference.split("&mut ").nth(1).unwrap().trim();
    assert_ne!(register, slot, "{ir}");
}

#[test]
fn an_eval_does_not_sink_past_a_read_of_what_its_spawn_holds() {
    let i = Interner::new();
    let ir = optimized(&i, "x = 40; y = bump(&mut x, 2); x + y", Effect::OPAQUE);
    let main = main_body(&ir);
    let eval = main.find("eval ").expect("the eval");
    let read = main.find("take ").expect("the read of x");
    assert!(eval < read, "{ir}");
}

/// `peek(&mut Int) -> &Int`: a reference result holds its argument's loan
/// (RFC-0028).
fn peek(i: &Interner) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern("peek")),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Default::default(),
        },
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(
                i.intern("n"),
                lift_to_poly(&Ty::Ref(
                    Mutability::Mut,
                    Box::new(TypeArg::uniform(Ty::I64)),
                )),
            )],
            ret: Box::new(lift_to_poly(&Ty::Ref(
                Mutability::Shared,
                Box::new(TypeArg::uniform(Ty::I64)),
            ))),
            captures: vec![],
            effect: Effect::PURE.into(),
        },
    }
}

#[test]
fn a_reference_returned_by_a_call_keeps_the_place_it_was_lent() {
    let i = Interner::new();
    let externs = [peek(&i), bump(&i, Effect::PURE)];
    let err = compile_multi_fn_optimized(
        &i,
        ("main", "x = 1; r = peek(&mut x); bump(&mut x, 1); *r"),
        &[],
        &[],
        &externs,
    )
    .expect_err("x is lent again while r still names it");
    assert!(err.contains("BorrowConflict"), "{err}");
    compile_multi_fn_optimized(
        &i,
        (
            "main",
            "x = 1; r = peek(&mut x); y = *r; bump(&mut x, 1); y",
        ),
        &[],
        &[],
        &externs,
    )
    .expect("r is dead before x is lent again");
}
