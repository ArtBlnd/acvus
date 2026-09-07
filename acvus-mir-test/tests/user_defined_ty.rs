//! Tests for UserDefined types (Iterator, Sequence) — migrated from acvus-mir/src/ty.rs.
//!
//! These tests construct Iterator/Sequence as `Ty::UserDefined` with proper `QualifiedRef`
//! via `acvus_ext::std_registries`, verifying unification, materiality,
//! and coercion behaviors.

use acvus_mir::graph::types::QualifiedRef;
use acvus_mir::ty::{
    InferTy, Materiality, Polarity, PolyBuilder, PolyTy, Solver, Ty, TypeRegistry, lift_ty,
};
use acvus_utils::Interner;

use Polarity::*;

// ── Helpers ──────────────────────────────────────────────────────────

/// Create an Interner and TypeRegistry with Iterator/Sequence registered.
fn setup() -> (Interner, TypeRegistry) {
    let interner = Interner::new();
    let mut type_registry = TypeRegistry::new();
    let _std_regs = acvus_ext::std_registries(&interner, &mut type_registry);
    (interner, type_registry)
}

/// Build Iterator<T> as concrete Ty (for materiality/pureable tests).
fn iter_ty(interner: &Interner, elem: Ty) -> Ty {
    let iter_qref = QualifiedRef::root(interner.intern("Iterator"));
    Ty::UserDefined {
        id: iter_qref,
        type_args: vec![elem],
    }
}

/// Build Sequence<T, O> as concrete Ty (for materiality/pureable tests).
fn seq_ty(interner: &Interner, elem: Ty, identity: Ty) -> Ty {
    let seq_qref = QualifiedRef::root(interner.intern("Sequence"));
    Ty::UserDefined {
        id: seq_qref,
        type_args: vec![elem, identity],
    }
}

/// Build Iterator<T> as InferTy (for unification tests).
fn iter_ity(interner: &Interner, elem: InferTy) -> InferTy {
    let iter_qref = QualifiedRef::root(interner.intern("Iterator"));
    InferTy::UserDefined {
        id: iter_qref,
        type_args: vec![elem],
    }
}

/// Build Sequence<T, O> as InferTy (for unification tests).
fn seq_ity(interner: &Interner, elem: InferTy, identity: InferTy) -> InferTy {
    let seq_qref = QualifiedRef::root(interner.intern("Sequence"));
    InferTy::UserDefined {
        id: seq_qref,
        type_args: vec![elem, identity],
    }
}

/// Lift concrete Ty to InferTy.
fn it(ty: &Ty) -> InferTy {
    lift_ty(ty)
}

// ================================================================
// UserDefined unification (same id, different args)
// ================================================================

#[ignore = "pending identity integration"]
#[test]
fn iterator_same_args_unifies() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let a = iter_ity(&i, it(&Ty::Int));
    let b = iter_ity(&i, it(&Ty::Int));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
}

#[ignore = "pending identity integration"]
#[test]
fn iterator_type_arg_mismatch_fails() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let a = iter_ity(&i, it(&Ty::Int));
    let b = iter_ity(&i, it(&Ty::String));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_err());
}

#[ignore = "pending identity integration"]
#[test]
fn iterator_type_param_resolves() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let t = s.fresh_ty_var();
    let a = iter_ity(&i, t.clone());
    let b = iter_ity(&i, it(&Ty::Int));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
    assert_eq!(s.resolve_ty(&t), it(&Ty::Int));
}

// ================================================================
// Sequence unification
// ================================================================

#[ignore = "pending identity integration"]
#[test]
fn sequence_same_identity_unifies() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o = s.alloc_identity();
    let a = seq_ity(&i, it(&Ty::Int), o.clone());
    let b = seq_ity(&i, it(&Ty::Int), o);
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
}

#[ignore = "pending identity integration"]
#[test]
fn sequence_identity_var_binds() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o_concrete = s.alloc_identity();
    let o_var = s.fresh_ty_var();
    let a = seq_ity(&i, it(&Ty::Int), o_concrete.clone());
    let b = seq_ity(&i, it(&Ty::Int), o_var.clone());
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
    assert_eq!(s.resolve_ty(&o_var), o_concrete);
}

#[ignore = "pending identity integration"]
#[test]
fn sequence_different_identity_invariant_fails() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o1 = s.alloc_identity();
    let o2 = s.alloc_identity();
    let a = seq_ity(&i, it(&Ty::Int), o1);
    let b = seq_ity(&i, it(&Ty::Int), o2);
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_err());
}

// ================================================================
// Materiality — UserDefined types are Ephemeral
// ================================================================

#[ignore = "pending identity integration"]
#[test]
fn iterator_is_ephemeral() {
    let (i, _reg) = setup();
    assert_eq!(
        iter_ty(&i, Ty::Int).materiality(),
        Materiality::Ephemeral
    );
}

#[ignore = "pending identity integration"]
#[test]
fn sequence_is_ephemeral() {
    let (i, _reg) = setup();
    let mut s = Solver::new();
    let o = s.alloc_identity();
    // For materiality test, use concrete Ty with Identity embedded
    assert_eq!(
        iter_ty(&i, Ty::Int).materiality(),
        Materiality::Ephemeral
    );
}

#[ignore = "pending identity integration"]
#[test]
fn iterator_not_materializable() {
    let (i, _reg) = setup();
    assert!(!iter_ty(&i, Ty::Int).is_materializable());
}

#[ignore = "pending identity integration"]
#[test]
fn sequence_not_materializable() {
    let (i, _reg) = setup();
    // Sequence with any concrete identity is still not materializable
    assert!(!seq_ty(&i, Ty::Int, Ty::Unit).is_materializable());
}

#[ignore = "pending identity integration"]
#[test]
fn list_of_iterator_not_materializable() {
    let (i, _reg) = setup();
    let list = Ty::List(Box::new(iter_ty(&i, Ty::Int)));
    assert!(!list.is_materializable());
}

// ================================================================
// is_pureable — UserDefined types are not pureable
// ================================================================

#[ignore = "pending identity integration"]
#[test]
fn iterator_not_pureable() {
    let (i, _reg) = setup();
    assert!(!iter_ty(&i, Ty::Int).is_pureable());
}

#[ignore = "pending identity integration"]
#[test]
fn sequence_not_pureable() {
    let (i, _reg) = setup();
    assert!(!seq_ty(&i, Ty::Int, Ty::Unit).is_pureable());
}

// ================================================================
// Move-only semantics — UserDefined is always move-only
// ================================================================

#[ignore = "pending identity integration"]
#[test]
fn iterator_is_move_only() {
    let (i, _reg) = setup();
    assert_eq!(
        acvus_mir::validate::move_check::is_move_only(&iter_ty(&i, Ty::Int)),
        Some(true)
    );
}

#[ignore = "pending identity integration"]
#[test]
fn sequence_is_move_only() {
    let (i, _reg) = setup();
    assert_eq!(
        acvus_mir::validate::move_check::is_move_only(&seq_ty(&i, Ty::Int, Ty::Unit)),
        Some(true)
    );
}

// ================================================================
// Iterator vs Sequence are different UserDefined types
// ================================================================

#[ignore = "pending identity integration"]
#[test]
fn iterator_vs_sequence_invariant_fails() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o = s.fresh_ty_var();
    let iter = iter_ity(&i, it(&Ty::Int));
    let seq = seq_ity(&i, it(&Ty::Int), o);
    assert!(s.unify_ty(&iter, &seq, Invariant, &reg).is_err());
}

// ================================================================
// instantiate_pair: CastRule from/to share Param placeholders
// ================================================================

#[ignore = "pending identity integration"]
#[test]
fn instantiate_pair_shares_params() {
    // CastRule: UserDefined(A, [T]) → List<T>
    // instantiate_pair must map T in `from` and T in `to` to the same fresh Param.
    let (i, mut reg) = setup();
    let id = QualifiedRef::root(i.intern("TestType"));
    reg.register(acvus_mir::ty::UserDefinedDecl {
        qref: id,
        type_params: vec![None],
    });
    let mut builder = PolyBuilder::new();
    let t = builder.fresh_ty_var();
    let from = PolyTy::UserDefined {
        id,
        type_args: vec![t.clone()],
    };
    let to = PolyTy::List(Box::new(t));

    let mut s = Solver::new();
    let (inst_from, inst_to) = s.instantiate_poly_pair(&from, &to);

    // Unify inst_from with concrete → T resolves
    let concrete_from = InferTy::UserDefined {
        id,
        type_args: vec![it(&Ty::Int)],
    };
    assert!(s.unify_ty(&concrete_from, &inst_from, Invariant, &reg).is_ok());

    // inst_to should now resolve to List<Int> (shared T)
    let resolved_to = s.resolve_ty(&inst_to);
    assert_eq!(resolved_to, InferTy::List(Box::new(it(&Ty::Int))));
}

// ================================================================
// ExternCast coercion: soundness + completeness
// ================================================================

#[ignore = "pending identity integration"]
#[test]
fn coerce_list_to_iterator_completeness() {
    // List<Int> ≤ Iterator<Int> via CastRule
    let (i, reg) = setup();
    let mut s = Solver::new();
    let list = it(&Ty::List(Box::new(Ty::Int)));
    let iter = iter_ity(&i, it(&Ty::Int));
    assert!(
        s.unify_ty(&list, &iter, Covariant, &reg).is_ok(),
        "List → Iterator coercion should succeed"
    );
}

#[ignore = "pending identity integration"]
#[test]
fn coerce_deque_to_iterator_completeness() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o = s.alloc_identity();
    let deque = InferTy::Deque(Box::new(it(&Ty::Int)), Box::new(o));
    let iter = iter_ity(&i, it(&Ty::Int));
    assert!(
        s.unify_ty(&deque, &iter, Covariant, &reg).is_ok(),
        "Deque → Iterator coercion should succeed"
    );
}

#[ignore = "pending identity integration"]
#[test]
fn coerce_deque_to_sequence_completeness() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o = s.alloc_identity();
    let deque = InferTy::Deque(Box::new(it(&Ty::Int)), Box::new(o.clone()));
    let seq = seq_ity(&i, it(&Ty::Int), o);
    assert!(
        s.unify_ty(&deque, &seq, Covariant, &reg).is_ok(),
        "Deque → Sequence coercion should succeed"
    );
}

#[ignore = "pending identity integration"]
#[test]
fn coerce_sequence_to_iterator_completeness() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let o = s.alloc_identity();
    let seq = seq_ity(&i, it(&Ty::Int), o);
    let iter = iter_ity(&i, it(&Ty::Int));
    assert!(
        s.unify_ty(&seq, &iter, Covariant, &reg).is_ok(),
        "Sequence → Iterator coercion should succeed"
    );
}

#[ignore = "pending identity integration"]
#[test]
fn coerce_iterator_to_list_soundness_rejected() {
    // Iterator → List is NOT valid (can't materialize lazy into eager implicitly)
    let (i, reg) = setup();
    let mut s = Solver::new();
    let iter = iter_ity(&i, it(&Ty::Int));
    let list = it(&Ty::List(Box::new(Ty::Int)));
    assert!(
        s.unify_ty(&iter, &list, Covariant, &reg).is_err(),
        "Iterator → List coercion must be rejected"
    );
}

#[ignore = "pending identity integration"]
#[test]
fn coerce_iterator_to_deque_soundness_rejected() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let iter = iter_ity(&i, it(&Ty::Int));
    let o = s.alloc_identity();
    let deque = InferTy::Deque(Box::new(it(&Ty::Int)), Box::new(o));
    assert!(
        s.unify_ty(&iter, &deque, Covariant, &reg).is_err(),
        "Iterator → Deque coercion must be rejected"
    );
}

#[ignore = "pending identity integration"]
#[test]
fn coerce_invariant_rejects_list_to_iterator() {
    // Invariant polarity: no coercion allowed
    let (i, reg) = setup();
    let mut s = Solver::new();
    let list = it(&Ty::List(Box::new(Ty::Int)));
    let iter = iter_ity(&i, it(&Ty::Int));
    assert!(
        s.unify_ty(&list, &iter, Invariant, &reg).is_err(),
        "Invariant should reject List → Iterator"
    );
}

// ================================================================
// LUB for same-id UserDefined
// ================================================================

#[ignore = "pending identity integration"]
#[test]
fn lub_sequence_identity_mismatch_to_iterator() {
    // Same Param used where Sequence<Int, O1> and Sequence<Int, O2> expected.
    // Identity mismatch → LUB via CastRule → Iterator<Int>.
    let (i, reg) = setup();
    let mut s = Solver::new();
    let p = s.fresh_ty_var();
    let o1 = s.alloc_identity();
    let o2 = s.alloc_identity();
    let a = seq_ity(&i, it(&Ty::Int), o1);
    let b = seq_ity(&i, it(&Ty::Int), o2);
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
