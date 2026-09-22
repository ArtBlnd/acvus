//! Tests for the owned iterator source `Items` as a UserDefined type.
//!
//! These tests construct `Items` as `Ty::UserDefined` with proper `QualifiedRef`
//! via `acvus_ext::std_registries`, verifying unification, the data
//! predicate, and coercion behaviors.

use acvus_mir::graph::types::QualifiedRef;
use acvus_mir::solver::{Answer, Conversion, Decision, Unsettled};
use acvus_mir::ty::{
    InferTy, PolyBuilder, PolyTy, Solver, Sources, Ty, TypeArg, TypeRegistry, lift_ty,
};
use acvus_utils::Interner;

// -- Helpers ----------------------------------------------------------

/// Create an Interner and TypeRegistry with `Items` registered.
fn setup() -> (Interner, TypeRegistry) {
    let interner = Interner::new();
    let externs = acvus_extern::Externs::combine(
        acvus_ext::std_registries::<acvus_extern::TypesOnly>(),
        &interner,
    )
    .expect("standard registries combine");
    (interner, externs.types)
}

/// Build `Items<T>` as concrete Ty.
fn iter_ty(interner: &Interner, elem: Ty) -> Ty {
    let iter_qref = QualifiedRef::root(interner.intern("Items"));
    Ty::UserDefined {
        id: iter_qref,
        type_args: vec![TypeArg::uniform(elem)],
        effect_args: vec![],
        identity_args: vec![acvus_mir::ty::IdentityTerm::Known(
            <acvus_mir::ty::IdentityId as acvus_utils::LocalIdOps>::from_raw(0),
        )],
    }
}

/// Build `Items<T>` as InferTy (for unification tests).
fn iter_ity(interner: &Interner, elem: InferTy) -> InferTy {
    let iter_qref = QualifiedRef::root(interner.intern("Items"));
    InferTy::UserDefined {
        id: iter_qref,
        type_args: vec![TypeArg::uniform(elem)],
        effect_args: vec![],
        identity_args: vec![acvus_mir::ty::IdentityTerm::Known(
            <acvus_mir::ty::IdentityId as acvus_utils::LocalIdOps>::from_raw(0),
        )],
    }
}

/// `Items<T>` whose source is still open, so a coercion may name it.
fn iter_ity_open(interner: &Interner, s: &mut Solver, elem: InferTy) -> InferTy {
    let iter_qref = QualifiedRef::root(interner.intern("Items"));
    InferTy::UserDefined {
        id: iter_qref,
        type_args: vec![TypeArg::uniform(elem)],
        effect_args: vec![],
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
    let mut sources = Sources::new();
    let mut s = Solver::new(&mut sources, &reg);
    let a = iter_ity(&i, it(&Ty::I64));
    let b = iter_ity(&i, it(&Ty::I64));
    assert!(s.unify(&a, &b).is_ok());
}

#[test]
fn iterator_type_arg_mismatch_fails() {
    let (i, reg) = setup();
    let mut sources = Sources::new();
    let mut s = Solver::new(&mut sources, &reg);
    let a = iter_ity(&i, it(&Ty::I64));
    let b = iter_ity(&i, it(&Ty::String));
    assert!(s.unify(&a, &b).is_err());
}

#[test]
fn iterator_type_param_resolves() {
    let (i, reg) = setup();
    let mut sources = Sources::new();
    let mut s = Solver::new(&mut sources, &reg);
    let t = s.fresh_ty_var();
    let a = iter_ity(&i, t.clone());
    let b = iter_ity(&i, it(&Ty::I64));
    assert!(s.unify(&a, &b).is_ok());
    assert_eq!(s.resolve_ty(&t), it(&Ty::I64));
}

// ================================================================
// is_data - an extension type is data; one over a function is not
// ================================================================

#[test]
fn iterator_is_data() {
    let (i, _reg) = setup();
    assert!(iter_ty(&i, Ty::I64).is_data());
}

#[test]
fn list_of_iterator_is_data() {
    let (i, _reg) = setup();
    let list = Ty::Array(
        Box::new(iter_ty(&i, Ty::I64)),
        acvus_mir::ty::LenTerm::Known(3),
    );
    assert!(list.is_data());
}

#[test]
fn iterator_over_fn_is_not_data() {
    let (i, _reg) = setup();
    let fn_ty = Ty::Fn {
        params: vec![],
        ret: Box::new(Ty::I64),
        captures: vec![],
        effect: acvus_mir::ty::Effect::PURE.into(),
    };
    assert!(!iter_ty(&i, fn_ty).is_data());
}

// ================================================================
// Move-only semantics - UserDefined is always move-only
// ================================================================

#[test]
fn iterator_is_move_only() {
    let (i, _reg) = setup();
    assert_eq!(
        acvus_mir::validate::move_check::is_move_only(&iter_ty(&i, Ty::I64)),
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
        specializable: vec![false],
    });
    let mut builder = PolyBuilder::new();
    let t = builder.fresh_ty_var();
    let from = PolyTy::UserDefined {
        id,
        type_args: vec![TypeArg::uniform(t.clone())],
        effect_args: vec![],
        identity_args: vec![],
    };
    let to = PolyTy::Array(Box::new(t), acvus_mir::ty::LenTerm::Known(3));

    let mut sources = Sources::new();
    let mut s = Solver::new(&mut sources, &reg);
    let (inst_from, inst_to) = s.instantiate_poly_pair(&from, &to);

    // Unify inst_from with concrete -> T resolves
    let concrete_from = InferTy::UserDefined {
        id,
        type_args: vec![TypeArg::uniform(it(&Ty::I64))],
        effect_args: vec![],
        identity_args: vec![],
    };
    assert!(s.unify(&concrete_from, &inst_from).is_ok());

    // inst_to should now resolve to List<Int> (shared T)
    let resolved_to = s.resolve_ty(&inst_to);
    assert_eq!(
        resolved_to,
        InferTy::Array(Box::new(it(&Ty::I64)), acvus_mir::ty::LenTerm::Known(3))
    );
}

// ================================================================
// ExternCast coercion: soundness + completeness
// ================================================================

#[test]
fn coerce_list_to_iterator_completeness() {
    // List<Int> <= Iterator<Int> via CastRule
    let (i, reg) = setup();
    let mut sources = Sources::new();
    let mut s = Solver::new(&mut sources, &reg);
    let list = it(&Ty::Array(
        Box::new(Ty::I64),
        acvus_mir::ty::LenTerm::Known(3),
    ));
    let iter = iter_ity_open(&i, &mut s, it(&Ty::I64));
    let conversion = s.decide(Decision::conversion(&list, &iter));
    let unsettled = s.settle();
    assert!(unsettled.is_empty(), "{unsettled:?}");
    assert!(
        matches!(
            s.answer(conversion),
            Some(Answer::Conversion(Conversion::Cast(_)))
        ),
        "List -> Iterator converts through the declared cast"
    );
}

#[test]
fn coerce_iterator_to_list_soundness_rejected() {
    // Iterator -> List is NOT valid (can't materialize lazy into eager implicitly)
    let (i, reg) = setup();
    let mut sources = Sources::new();
    let mut s = Solver::new(&mut sources, &reg);
    let iter = iter_ity(&i, it(&Ty::I64));
    let list = it(&Ty::Array(
        Box::new(Ty::I64),
        acvus_mir::ty::LenTerm::Known(3),
    ));
    let conversion = s.decide(Decision::conversion(&iter, &list));
    let unsettled = s.settle();
    assert!(
        matches!(unsettled.as_slice(), [Unsettled::NoConversion { .. }]),
        "Iterator -> List has no conversion: {unsettled:?}"
    );
    assert_eq!(s.answer(conversion), None);
}

#[test]
fn coerce_invariant_rejects_list_to_iterator() {
    // Invariant polarity: no coercion allowed
    let (i, reg) = setup();
    let mut sources = Sources::new();
    let mut s = Solver::new(&mut sources, &reg);
    let list = it(&Ty::Array(
        Box::new(Ty::I64),
        acvus_mir::ty::LenTerm::Known(3),
    ));
    let iter = iter_ity(&i, it(&Ty::I64));
    assert!(
        s.unify(&list, &iter).is_err(),
        "Invariant should reject List -> Iterator"
    );
}

// ================================================================
// LUB for same-id UserDefined
// ================================================================
