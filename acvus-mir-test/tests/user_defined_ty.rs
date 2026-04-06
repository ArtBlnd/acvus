//! Tests for UserDefined types (Iterator, Sequence) — migrated from acvus-mir/src/ty.rs.
//!
//! These tests construct Iterator/Sequence as `Ty::UserDefined` with proper `QualifiedRef`
//! via `acvus_ext::std_registries`, verifying unification, effect propagation, materiality,
//! and coercion behaviors.

use std::collections::BTreeSet;

use acvus_mir::graph::types::QualifiedRef;
use acvus_mir::ty::{
    Effect, EffectSet, EffectTarget, InferEffect, InferTy, Materiality, Param, Polarity,
    PolyBuilder, PolyTy, Solver, Ty, TypeRegistry, lift_effect, lift_ty,
};
use acvus_utils::{Freeze, Interner};

use Polarity::*;

// ── Helpers ──────────────────────────────────────────────────────────

/// Create an Interner and TypeRegistry with Iterator/Sequence registered.
fn setup() -> (Interner, TypeRegistry) {
    let interner = Interner::new();
    let mut type_registry = TypeRegistry::new();
    let _std_regs = acvus_ext::std_registries(&interner, &mut type_registry);
    (interner, type_registry)
}

/// Build Iterator<T, E> as concrete Ty (for materiality/pureable tests).
fn iter_ty(interner: &Interner, elem: Ty, effect: Effect) -> Ty {
    let iter_qref = QualifiedRef::root(interner.intern("Iterator"));
    Ty::UserDefined {
        id: iter_qref,
        type_args: vec![elem],
        effect_args: vec![effect],
    }
}

/// Build Sequence<T, O, E> as concrete Ty (for materiality/pureable tests).
fn seq_ty(interner: &Interner, elem: Ty, identity: Ty, effect: Effect) -> Ty {
    let seq_qref = QualifiedRef::root(interner.intern("Sequence"));
    Ty::UserDefined {
        id: seq_qref,
        type_args: vec![elem, identity],
        effect_args: vec![effect],
    }
}

/// Build Iterator<T, E> as InferTy (for unification tests).
fn iter_ity(interner: &Interner, elem: InferTy, effect: InferEffect) -> InferTy {
    let iter_qref = QualifiedRef::root(interner.intern("Iterator"));
    InferTy::UserDefined {
        id: iter_qref,
        type_args: vec![elem],
        effect_args: vec![effect],
    }
}

/// Build Sequence<T, O, E> as InferTy (for unification tests).
fn seq_ity(interner: &Interner, elem: InferTy, identity: InferTy, effect: InferEffect) -> InferTy {
    let seq_qref = QualifiedRef::root(interner.intern("Sequence"));
    InferTy::UserDefined {
        id: seq_qref,
        type_args: vec![elem, identity],
        effect_args: vec![effect],
    }
}

/// Lift concrete Effect to InferEffect.
fn ie(effect: &Effect) -> InferEffect {
    lift_effect(effect)
}

/// Lift concrete Ty to InferTy.
fn it(ty: &Ty) -> InferTy {
    lift_ty(ty)
}

/// Wrap InferTy into ParamTerm<Infer> with a dummy name.
fn tp(interner: &Interner, ty: InferTy) -> acvus_mir::ty::ParamTerm<acvus_mir::ty::Infer> {
    acvus_mir::ty::ParamTerm::<acvus_mir::ty::Infer>::new(interner.intern("_"), ty)
}

/// A resolved effectful Effect with a Token write — represents a non-pure effect.
fn test_effectful(interner: &Interner) -> Effect {
    Effect::Resolved(EffectSet {
        reads: BTreeSet::new(),
        writes: BTreeSet::from([EffectTarget::Token(QualifiedRef::root(interner.intern("__test")))]),
    })
}

// ================================================================
// UserDefined unification (same id, different args)
// ================================================================

#[test]
fn iterator_same_effect_pure_unifies() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let a = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    let b = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
}

#[test]
fn iterator_same_effect_effectful_unifies() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let a = iter_ity(&i, it(&Ty::Int), ie(&test_effectful(&i)));
    let b = iter_ity(&i, it(&Ty::Int), ie(&test_effectful(&i)));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
}

#[test]
fn iterator_effect_mismatch_invariant_fails() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let a = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    let b = iter_ity(&i, it(&Ty::Int), ie(&test_effectful(&i)));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_err());
}

#[test]
fn iterator_type_arg_mismatch_fails() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let a = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    let b = iter_ity(&i, it(&Ty::String), ie(&Effect::pure()));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_err());
}

// ================================================================
// Effect variable binding via unification
// ================================================================

#[test]
fn iterator_effect_var_binds_to_pure() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let e = s.fresh_effect_var();
    let a = iter_ity(&i, it(&Ty::Int), e.clone());
    let b = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
    assert_eq!(s.resolve_infer_effect(&e), ie(&Effect::pure()));
}

#[test]
fn iterator_effect_var_binds_to_effectful() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let e = s.fresh_effect_var();
    let a = iter_ity(&i, it(&Ty::Int), e.clone());
    let b = iter_ity(&i, it(&Ty::Int), ie(&test_effectful(&i)));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
    assert_eq!(s.resolve_infer_effect(&e), ie(&test_effectful(&i)));
}

#[test]
fn iterator_type_param_resolves() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let t = s.fresh_ty_var();
    let a = iter_ity(&i, t.clone(), ie(&Effect::pure()));
    let b = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
    assert_eq!(s.resolve_ty(&t), it(&Ty::Int));
}

// ================================================================
// Sequence unification
// ================================================================

#[test]
fn sequence_same_identity_unifies() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o = s.alloc_identity(false);
    let a = seq_ity(&i, it(&Ty::Int), o.clone(), ie(&Effect::pure()));
    let b = seq_ity(&i, it(&Ty::Int), o, ie(&Effect::pure()));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
}

#[test]
fn sequence_identity_var_binds() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o_concrete = s.alloc_identity(false);
    let o_var = s.fresh_ty_var();
    let a = seq_ity(&i, it(&Ty::Int), o_concrete.clone(), ie(&Effect::pure()));
    let b = seq_ity(&i, it(&Ty::Int), o_var.clone(), ie(&Effect::pure()));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
    assert_eq!(s.resolve_ty(&o_var), o_concrete);
}

#[test]
fn sequence_effect_var_binds_pure() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let e = s.fresh_effect_var();
    let o = s.fresh_ty_var();
    let a = seq_ity(&i, it(&Ty::Int), o.clone(), e.clone());
    let b = seq_ity(&i, it(&Ty::Int), o, ie(&Effect::pure()));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
    assert_eq!(s.resolve_infer_effect(&e), ie(&Effect::pure()));
}

#[test]
fn sequence_effect_var_binds_effectful() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let e = s.fresh_effect_var();
    let o = s.fresh_ty_var();
    let a = seq_ity(&i, it(&Ty::Int), o.clone(), e.clone());
    let b = seq_ity(&i, it(&Ty::Int), o, ie(&test_effectful(&i)));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
    assert_eq!(s.resolve_infer_effect(&e), ie(&test_effectful(&i)));
}

#[test]
fn sequence_different_identity_invariant_fails() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o1 = s.alloc_identity(false);
    let o2 = s.alloc_identity(false);
    let a = seq_ity(&i, it(&Ty::Int), o1, ie(&Effect::pure()));
    let b = seq_ity(&i, it(&Ty::Int), o2, ie(&Effect::pure()));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_err());
}

#[test]
fn sequence_same_identity_effect_mismatch_invariant_fails() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o = s.alloc_identity(false);
    let a = seq_ity(&i, it(&Ty::Int), o.clone(), ie(&Effect::pure()));
    let b = seq_ity(&i, it(&Ty::Int), o, ie(&test_effectful(&i)));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_err());
}

// ================================================================
// Materiality — UserDefined types are Ephemeral
// ================================================================

#[test]
fn iterator_is_ephemeral() {
    let (i, _reg) = setup();
    assert_eq!(
        iter_ty(&i, Ty::Int, Effect::pure()).materiality(),
        Materiality::Ephemeral
    );
}

#[test]
fn sequence_is_ephemeral() {
    let (i, _reg) = setup();
    let mut s = Solver::new();
    let o = s.alloc_identity(false);
    // For materiality test, use concrete Ty with Identity embedded
    assert_eq!(
        iter_ty(&i, Ty::Int, Effect::pure()).materiality(),
        Materiality::Ephemeral
    );
}

#[test]
fn iterator_not_materializable() {
    let (i, _reg) = setup();
    assert!(!iter_ty(&i, Ty::Int, Effect::pure()).is_materializable());
    assert!(!iter_ty(&i, Ty::Int, test_effectful(&i)).is_materializable());
}

#[test]
fn sequence_not_materializable() {
    let (i, _reg) = setup();
    // Sequence with any concrete identity is still not materializable
    assert!(!seq_ty(&i, Ty::Int, Ty::Unit, Effect::pure()).is_materializable());
    assert!(!seq_ty(&i, Ty::Int, Ty::Unit, test_effectful(&i)).is_materializable());
}

#[test]
fn list_of_iterator_not_materializable() {
    let (i, _reg) = setup();
    let list = Ty::List(Box::new(iter_ty(&i, Ty::Int, Effect::pure())));
    assert!(!list.is_materializable());
}

// ================================================================
// is_pureable — UserDefined types are not pureable
// ================================================================

#[test]
fn iterator_not_pureable() {
    let (i, _reg) = setup();
    assert!(!iter_ty(&i, Ty::Int, Effect::pure()).is_pureable());
}

#[test]
fn sequence_not_pureable() {
    let (i, _reg) = setup();
    assert!(!seq_ty(&i, Ty::Int, Ty::Unit, Effect::pure()).is_pureable());
}

// ================================================================
// Move-only semantics — UserDefined is always move-only
// ================================================================

#[test]
fn iterator_is_move_only() {
    let (i, _reg) = setup();
    assert_eq!(
        acvus_mir::validate::move_check::is_move_only(&iter_ty(&i, Ty::Int, Effect::pure())),
        Some(true)
    );
    assert_eq!(
        acvus_mir::validate::move_check::is_move_only(&iter_ty(&i, Ty::Int, test_effectful(&i))),
        Some(true)
    );
}

#[test]
fn sequence_is_move_only() {
    let (i, _reg) = setup();
    assert_eq!(
        acvus_mir::validate::move_check::is_move_only(&seq_ty(&i, Ty::Int, Ty::Unit, Effect::pure())),
        Some(true)
    );
}

// ================================================================
// Iterator vs Sequence are different UserDefined types
// ================================================================

#[test]
fn iterator_vs_sequence_invariant_fails() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o = s.fresh_ty_var();
    let iter = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    let seq = seq_ity(&i, it(&Ty::Int), o, ie(&Effect::pure()));
    assert!(s.unify_ty(&iter, &seq, Invariant, &reg).is_err());
}

// ================================================================
// HOF effect sharing — effect var binds via UserDefined then propagates
// ================================================================

#[test]
fn hof_shared_effect_var_binds_then_callback() {
    // Simulate: filter(Iterator<Int, E>, Fn(Int → Bool, effect: E)) → Iterator<Int, E>
    // E binds to Effectful from the iterator. Pure callback unifies with E=Effectful in Covariant.
    let (i, reg) = setup();
    let mut s = Solver::new();
    let e = s.fresh_effect_var();

    // Bind e = Effectful via iterator
    let iter_sig = iter_ity(&i, it(&Ty::Int), e.clone());
    let iter_actual = iter_ity(&i, it(&Ty::Int), ie(&test_effectful(&i)));
    assert!(s.unify_ty(&iter_actual, &iter_sig, Covariant, &reg).is_ok());
    assert_eq!(s.resolve_infer_effect(&e), ie(&test_effectful(&i)));

    // Callback: Fn{effect:Pure} vs Fn{effect:e(=Effectful)}
    let actual_cb = InferTy::Fn {
        params: vec![tp(&i, it(&Ty::Int))],
        ret: Box::new(it(&Ty::Bool)),
        captures: vec![],
        effect: ie(&Effect::pure()),
        hint: None,
    };
    let expected_cb = InferTy::Fn {
        params: vec![tp(&i, it(&Ty::Int))],
        ret: Box::new(it(&Ty::Bool)),
        captures: vec![],
        effect: e,
        hint: None,
    };
    // Pure ≤ Effectful in covariant → OK
    assert!(
        s.unify_ty(&actual_cb, &expected_cb, Covariant, &reg).is_ok(),
        "Pure callback should be accepted where Effectful expected (covariant)"
    );
}

// ================================================================
// Effect subtyping: invariant rejects mismatch
// ================================================================

#[test]
fn effect_subtyping_invariant_rejects_mismatch() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let a = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    let b = iter_ity(&i, it(&Ty::Int), ie(&test_effectful(&i)));
    assert!(
        s.unify_ty(&a, &b, Invariant, &reg).is_err(),
        "Invariant should reject Pure vs Effectful"
    );
}

// ================================================================
// instantiate_pair: CastRule from/to share Param placeholders
// ================================================================

#[test]
fn instantiate_pair_shares_params() {
    // CastRule: UserDefined(A, [T]) → List<T>
    // instantiate_pair must map T in `from` and T in `to` to the same fresh Param.
    let (i, mut reg) = setup();
    let id = QualifiedRef::root(i.intern("TestType"));
    reg.register(acvus_mir::ty::UserDefinedDecl {
        qref: id,
        type_params: vec![None],
        effect_params: vec![],
    });
    let mut builder = PolyBuilder::new();
    let t = builder.fresh_ty_var();
    let from = PolyTy::UserDefined {
        id,
        type_args: vec![t.clone()],
        effect_args: vec![],
    };
    let to = PolyTy::List(Box::new(t));

    let mut s = Solver::new();
    let (inst_from, inst_to) = s.instantiate_poly_pair(&from, &to);

    // Unify inst_from with concrete → T resolves
    let concrete_from = InferTy::UserDefined {
        id,
        type_args: vec![it(&Ty::Int)],
        effect_args: vec![],
    };
    assert!(s.unify_ty(&concrete_from, &inst_from, Invariant, &reg).is_ok());

    // inst_to should now resolve to List<Int> (shared T)
    let resolved_to = s.resolve_ty(&inst_to);
    assert_eq!(resolved_to, InferTy::List(Box::new(it(&Ty::Int))));
}

// ================================================================
// ExternCast coercion: soundness + completeness
// ================================================================

#[test]
fn coerce_list_to_iterator_completeness() {
    // List<Int> ≤ Iterator<Int, Pure> via CastRule
    let (i, reg) = setup();
    let mut s = Solver::new();
    let list = it(&Ty::List(Box::new(Ty::Int)));
    let iter = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    assert!(
        s.unify_ty(&list, &iter, Covariant, &reg).is_ok(),
        "List → Iterator coercion should succeed"
    );
}

#[test]
fn coerce_list_to_iterator_param_resolution() {
    // List<Int> ≤ Iterator<T, E> → T=Int, E=Pure
    let (i, reg) = setup();
    let mut s = Solver::new();
    let list = it(&Ty::List(Box::new(Ty::Int)));
    let t = s.fresh_ty_var();
    let e = s.fresh_effect_var();
    let iter = iter_ity(&i, t.clone(), e.clone());
    assert!(s.unify_ty(&list, &iter, Covariant, &reg).is_ok());
    assert_eq!(s.resolve_ty(&t), it(&Ty::Int), "T should resolve to Int");
    assert!(
        matches!(s.resolve_infer_effect(&e), acvus_mir::ty::EffectTerm::Resolved(ref es) if es.is_pure()),
        "E should resolve to Pure"
    );
}

#[test]
fn coerce_deque_to_iterator_completeness() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o = s.alloc_identity(false);
    let deque = InferTy::Deque(Box::new(it(&Ty::Int)), Box::new(o));
    let iter = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    assert!(
        s.unify_ty(&deque, &iter, Covariant, &reg).is_ok(),
        "Deque → Iterator coercion should succeed"
    );
}

#[test]
fn coerce_deque_to_sequence_completeness() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o = s.alloc_identity(false);
    let deque = InferTy::Deque(Box::new(it(&Ty::Int)), Box::new(o.clone()));
    let seq = seq_ity(&i, it(&Ty::Int), o, ie(&Effect::pure()));
    assert!(
        s.unify_ty(&deque, &seq, Covariant, &reg).is_ok(),
        "Deque → Sequence coercion should succeed"
    );
}

#[test]
fn coerce_sequence_to_iterator_completeness() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o = s.alloc_identity(false);
    let seq = seq_ity(&i, it(&Ty::Int), o, ie(&Effect::pure()));
    let iter = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    assert!(
        s.unify_ty(&seq, &iter, Covariant, &reg).is_ok(),
        "Sequence → Iterator coercion should succeed"
    );
}

#[test]
fn coerce_iterator_to_list_soundness_rejected() {
    // Iterator → List is NOT valid (can't materialize lazy into eager implicitly)
    let (i, reg) = setup();
    let mut s = Solver::new();
    let iter = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    let list = it(&Ty::List(Box::new(Ty::Int)));
    assert!(
        s.unify_ty(&iter, &list, Covariant, &reg).is_err(),
        "Iterator → List coercion must be rejected"
    );
}

#[test]
fn coerce_iterator_to_deque_soundness_rejected() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let iter = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    let o = s.alloc_identity(false);
    let deque = InferTy::Deque(Box::new(it(&Ty::Int)), Box::new(o));
    assert!(
        s.unify_ty(&iter, &deque, Covariant, &reg).is_err(),
        "Iterator → Deque coercion must be rejected"
    );
}

#[test]
fn coerce_invariant_rejects_list_to_iterator() {
    // Invariant polarity: no coercion allowed
    let (i, reg) = setup();
    let mut s = Solver::new();
    let list = it(&Ty::List(Box::new(Ty::Int)));
    let iter = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    assert!(
        s.unify_ty(&list, &iter, Invariant, &reg).is_err(),
        "Invariant should reject List → Iterator"
    );
}

// ================================================================
// LUB: effect union for same-id UserDefined
// ================================================================

#[test]
fn iterator_effect_subtyping_covariant() {
    // Iterator<Int, Pure> ≤ Iterator<Int, IO> in Covariant — subeffect, not LUB.
    let (i, reg) = setup();
    let mut s = Solver::new();
    let pure_iter = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    let io_iter = iter_ity(&i, it(&Ty::Int), ie(&test_effectful(&i)));
    // Pure ≤ IO in Covariant → OK (subeffect)
    assert!(
        s.unify_ty(&pure_iter, &io_iter, Covariant, &reg).is_ok(),
        "Pure Iterator should be subtype of IO Iterator in Covariant"
    );
    // IO ≤ Pure in Covariant → FAIL
    let mut s2 = Solver::new();
    assert!(
        s2.unify_ty(&io_iter, &pure_iter, Covariant, &reg).is_err(),
        "IO Iterator should NOT be subtype of Pure Iterator in Covariant"
    );
}

#[test]
fn iterator_effect_var_resolves_via_lub() {
    // Param with effect var: Iterator<Int, ?E>. Unify with both Pure and IO → E = IO.
    let (i, reg) = setup();
    let mut s = Solver::new();
    let e = s.fresh_effect_var();
    let param_iter = iter_ity(&i, it(&Ty::Int), e.clone());
    let pure_iter = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    let io_iter = iter_ity(&i, it(&Ty::Int), ie(&test_effectful(&i)));
    // ?E matches Pure
    assert!(s.unify_ty(&param_iter, &pure_iter, Invariant, &reg).is_ok());
    assert!(
        matches!(s.resolve_infer_effect(&e), acvus_mir::ty::EffectTerm::Resolved(ref es) if es.is_pure()),
        "E should be Pure after first unify"
    );
    // Now ?E (=Pure) vs IO in Covariant → Pure ≤ IO → OK, E stays Pure
    assert!(s.unify_ty(&param_iter, &io_iter, Covariant, &reg).is_ok());
    assert!(
        matches!(s.resolve_infer_effect(&e), acvus_mir::ty::EffectTerm::Resolved(ref es) if es.is_pure()),
        "E stays Pure (subeffect, not LUB)"
    );
}

#[test]
fn lub_iterator_effect_invariant_rejects() {
    // Invariant: Iterator<Int, Pure> vs Iterator<Int, IO> → error
    let (i, reg) = setup();
    let mut s = Solver::new();
    let a = iter_ity(&i, it(&Ty::Int), ie(&Effect::pure()));
    let b = iter_ity(&i, it(&Ty::Int), ie(&test_effectful(&i)));
    assert!(
        s.unify_ty(&a, &b, Invariant, &reg).is_err(),
        "Invariant should reject effect mismatch"
    );
}

#[test]
fn lub_sequence_identity_mismatch_to_iterator() {
    // Same Param used where Sequence<Int, O1, Pure> and Sequence<Int, O2, Pure> expected.
    // Identity mismatch → LUB via CastRule → Iterator<Int, Pure>.
    let (i, reg) = setup();
    let mut s = Solver::new();
    let p = s.fresh_ty_var();
    let o1 = s.alloc_identity(false);
    let o2 = s.alloc_identity(false);
    let a = seq_ity(&i, it(&Ty::Int), o1, ie(&Effect::pure()));
    let b = seq_ity(&i, it(&Ty::Int), o2, ie(&Effect::pure()));
    assert!(s.unify_ty(&p, &a, Covariant, &reg).is_ok());
    let result = s.unify_ty(&p, &b, Covariant, &reg);
    assert!(
        result.is_ok(),
        "LUB via CastRule should succeed: {result:?}"
    );
    let resolved = s.resolve_ty(&p);
    // Should be Iterator (CastRule: Sequence → Iterator)
    match &resolved {
        InferTy::UserDefined { id, .. } => {
            assert_eq!(
                i.resolve(id.name),
                "Iterator",
                "LUB of identity-mismatched Sequences should be Iterator"
            );
        }
        other => panic!("expected UserDefined(Iterator), got {other:?}"),
    }
}
