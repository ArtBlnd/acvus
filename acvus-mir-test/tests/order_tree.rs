//! The IR dump shows what the body's yielded Order waits for (RFC-0007).

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{Effect, ParamTerm, Poly, Reissue, Ty, TyTerm, lift_to_poly};
use acvus_mir_test::*;
use acvus_utils::Interner;
use std::collections::BTreeSet;

fn commutative_io(i: &Interner, name: &str) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Default::default(),
            requires: vec![],
        },
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(
                i.intern("x"),
                lift_to_poly(&Ty::I64),
            )],
            ret: Box::new(lift_to_poly(&Ty::I64)),
            captures: vec![],
            effect: Effect::with_contexts(Reissue::Opaque, true, BTreeSet::new(), BTreeSet::new())
                .into(),
        },
    }
}

#[test]
fn two_commutative_calls_merge_and_the_dump_shows_the_tree() {
    let i = Interner::new();
    let ext = [commutative_io(&i, "io1"), commutative_io(&i, "io2")];
    let ir = compile_multi_fn_optimized(
        &i,
        ("main", "let a = io1(1); let b = io2(2); a + b"),
        &[],
        &[],
        &ext,
    )
    .unwrap();
    assert!(ir.contains("= merge "), "{ir}");
    assert!(ir.contains("; orders:"), "{ir}");
    assert!(ir.contains("merge@"), "{ir}");
    assert_eq!(ir.matches("<- spawn@").count(), 2, "{ir}");
}

#[test]
fn a_chain_of_calls_is_one_line() {
    let i = Interner::new();
    let io = Function {
        qref: QualifiedRef::root(i.intern("io")),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Default::default(),
            requires: vec![],
        },
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(
                i.intern("x"),
                lift_to_poly(&Ty::I64),
            )],
            ret: Box::new(lift_to_poly(&Ty::I64)),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
        },
    };
    let ir = compile_script_ir_with(
        &i,
        "let a = io(1); io(a)",
        &rustc_hash::FxHashMap::default(),
        &[io],
    )
    .unwrap();
    assert!(ir.contains("call@"), "{ir}");
    assert!(ir.contains("(entry)"), "{ir}");
    assert!(!ir.contains("merge@"), "{ir}");
}

#[test]
fn anyorder_in_a_script_merges_its_calls() {
    let i = Interner::new();
    let io = Function {
        qref: QualifiedRef::root(i.intern("io")),
        kind: FnKind::Extern {
            bounds: vec![],
            effect_bounds: vec![],
            instances: Default::default(),
            requires: vec![],
        },
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(
                i.intern("x"),
                lift_to_poly(&Ty::I64),
            )],
            ret: Box::new(lift_to_poly(&Ty::I64)),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
        },
    };
    let ir = compile_script_ir_with(
        &i,
        "anyorder { let a = io(1); let b = io(2); } 0",
        &rustc_hash::FxHashMap::default(),
        &[io],
    )
    .unwrap();
    assert!(ir.contains("= merge "), "{ir}");
    assert!(ir.contains("merge@"), "{ir}");
}
