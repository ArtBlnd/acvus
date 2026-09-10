//! Tests for the Iterator UserDefined type.
//!
//! These tests construct Iterator as `Ty::UserDefined` with proper `QualifiedRef`
//! via `acvus_ext::std_registries`, verifying unification, materiality,
//! and coercion behaviors.

use acvus_mir::graph::types::QualifiedRef;
use acvus_mir::ty::{
    InferTy, Materiality, Polarity, PolyBuilder, PolyTy, Solver, Ty, TypeRegistry, lift_ty,
};
use acvus_utils::Interner;

use Polarity::*;

// -- Helpers ----------------------------------------------------------

/// Create an Interner and TypeRegistry with Iterator registered.
fn setup() -> (Interner, TypeRegistry) {
    let interner = Interner::new();
    let mut type_registry = TypeRegistry::new();
    for registry in acvus_ext::std_registries::<acvus_extern::TypesOnly>() {
        registry.register(&interner, &mut type_registry);
    }
    (interner, type_registry)
}

/// Build Iterator<T> as concrete Ty (for materiality/pureable tests).
fn iter_ty(interner: &Interner, elem: Ty) -> Ty {
    let iter_qref = QualifiedRef::root(interner.intern("Iterator"));
    Ty::UserDefined {
        id: iter_qref,
        type_args: vec![elem],
        effect_args: vec![acvus_mir::ty::Effect::PURE.into()],
        identity_args: vec![acvus_mir::ty::IdentityTerm::Known(
            <acvus_mir::ty::IdentityId as acvus_utils::LocalIdOps>::from_raw(0),
        )],
    }
}

/// Build Iterator<T> as InferTy (for unification tests).
fn iter_ity(interner: &Interner, elem: InferTy) -> InferTy {
    let iter_qref = QualifiedRef::root(interner.intern("Iterator"));
    InferTy::UserDefined {
        id: iter_qref,
        type_args: vec![elem],
        effect_args: vec![acvus_mir::ty::Effect::PURE.into()],
        identity_args: vec![acvus_mir::ty::IdentityTerm::Known(
            <acvus_mir::ty::IdentityId as acvus_utils::LocalIdOps>::from_raw(0),
        )],
    }
}

/// `Iterator<T>` whose source is still open, so a coercion may name it.
fn iter_ity_open(interner: &Interner, s: &mut Solver, elem: InferTy) -> InferTy {
    let iter_qref = QualifiedRef::root(interner.intern("Iterator"));
    InferTy::UserDefined {
        id: iter_qref,
        type_args: vec![elem],
        effect_args: vec![acvus_mir::ty::Effect::PURE.into()],
        identity_args: vec![s.fresh_identity_var()],
    }
}

/// Lift concrete Ty to InferTy.
fn it(ty: &Ty) -> InferTy {
    lift_ty(ty)
}

// ================================================================
// UserDefined unification (same id, different args)
// ================================================================

#[test]
fn iterator_same_args_unifies() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let a = iter_ity(&i, it(&Ty::Int));
    let b = iter_ity(&i, it(&Ty::Int));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_ok());
}

#[test]
fn iterator_type_arg_mismatch_fails() {
    let (i, reg) = setup();
    let mut s = Solver::new();
    let a = iter_ity(&i, it(&Ty::Int));
    let b = iter_ity(&i, it(&Ty::String));
    assert!(s.unify_ty(&a, &b, Invariant, &reg).is_err());
}

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
// Materiality - UserDefined types are Ephemeral
// ================================================================

#[test]
fn iterator_is_ephemeral() {
    let (i, _reg) = setup();
    assert_eq!(iter_ty(&i, Ty::Int).materiality(), Materiality::Ephemeral);
}

#[test]
fn iterator_not_materializable() {
    let (i, _reg) = setup();
    assert!(!iter_ty(&i, Ty::Int).is_materializable());
}

#[test]
fn list_of_iterator_not_materializable() {
    let (i, _reg) = setup();
    let list = Ty::Array(
        Box::new(iter_ty(&i, Ty::Int)),
        acvus_mir::ty::LenTerm::Known(3),
    );
    assert!(!list.is_materializable());
}

// ================================================================
// is_pureable - UserDefined types are not pureable
// ================================================================

#[test]
fn iterator_not_pureable() {
    let (i, _reg) = setup();
    assert!(!iter_ty(&i, Ty::Int).is_pureable());
}

// ================================================================
// Move-only semantics - UserDefined is always move-only
// ================================================================

#[test]
fn iterator_is_move_only() {
    let (i, _reg) = setup();
    assert_eq!(
        acvus_mir::validate::move_check::is_move_only(&iter_ty(&i, Ty::Int)),
        Some(true)
    );
}

// ================================================================
// instantiate_pair: CastRule from/to share Param placeholders
// ================================================================

#[test]
fn instantiate_pair_shares_params() {
    // CastRule: UserDefined(A, [T]) -> List<T>
    // instantiate_pair must map T in `from` and T in `to` to the same fresh Param.
    let (i, mut reg) = setup();
    let id = QualifiedRef::root(i.intern("TestType"));
    reg.register(acvus_mir::ty::UserDefinedDecl {
        qref: id,
        type_params: vec![acvus_mir::ty::TyVarBound::Any],
        effect_params: 0,
        identity_params: 0,
    });
    let mut builder = PolyBuilder::new();
    let t = builder.fresh_ty_var();
    let from = PolyTy::UserDefined {
        id,
        type_args: vec![t.clone()],
        effect_args: vec![],
        identity_args: vec![],
    };
    let to = PolyTy::Array(Box::new(t), acvus_mir::ty::LenTerm::Known(3));

    let mut s = Solver::new();
    let (inst_from, inst_to) = s.instantiate_poly_pair(&from, &to);

    // Unify inst_from with concrete -> T resolves
    let concrete_from = InferTy::UserDefined {
        id,
        type_args: vec![it(&Ty::Int)],
        effect_args: vec![],
        identity_args: vec![],
    };
    assert!(
        s.unify_ty(&concrete_from, &inst_from, Invariant, &reg)
            .is_ok()
    );

    // inst_to should now resolve to List<Int> (shared T)
    let resolved_to = s.resolve_ty(&inst_to);
    assert_eq!(
        resolved_to,
        InferTy::Array(Box::new(it(&Ty::Int)), acvus_mir::ty::LenTerm::Known(3))
    );
}

// ================================================================
// ExternCast coercion: soundness + completeness
// ================================================================

#[test]
fn coerce_list_to_iterator_completeness() {
    // List<Int> <= Iterator<Int> via CastRule
    let (i, reg) = setup();
    let mut s = Solver::new();
    let list = it(&Ty::Array(
        Box::new(Ty::Int),
        acvus_mir::ty::LenTerm::Known(3),
    ));
    let iter = iter_ity_open(&i, &mut s, it(&Ty::Int));
    assert!(
        s.unify_ty(&list, &iter, Covariant, &reg).is_ok(),
        "List -> Iterator coercion should succeed"
    );
}

#[test]
fn coerce_iterator_to_list_soundness_rejected() {
    // Iterator -> List is NOT valid (can't materialize lazy into eager implicitly)
    let (i, reg) = setup();
    let mut s = Solver::new();
    let iter = iter_ity(&i, it(&Ty::Int));
    let list = it(&Ty::Array(
        Box::new(Ty::Int),
        acvus_mir::ty::LenTerm::Known(3),
    ));
    assert!(
        s.unify_ty(&iter, &list, Covariant, &reg).is_err(),
        "Iterator -> List coercion must be rejected"
    );
}

#[test]
fn coerce_invariant_rejects_list_to_iterator() {
    // Invariant polarity: no coercion allowed
    let (i, reg) = setup();
    let mut s = Solver::new();
    let list = it(&Ty::Array(
        Box::new(Ty::Int),
        acvus_mir::ty::LenTerm::Known(3),
    ));
    let iter = iter_ity(&i, it(&Ty::Int));
    assert!(
        s.unify_ty(&list, &iter, Invariant, &reg).is_err(),
        "Invariant should reject List -> Iterator"
    );
}

// ================================================================
// LUB for same-id UserDefined
// ================================================================
