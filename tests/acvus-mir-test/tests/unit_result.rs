//! A unit's result may hold what its parameters lend, as a lambda's may;
//! only the result of the unit a host runs holds no loan (ownership §5).

use acvus_mir::ty::{IntTy, Mutability, TyKind};
use acvus_mir_test::Case;

/// A unit giving back the reference it was passed: its caller reads
/// through it, and the loan it carries is its caller's own.
#[test]
fn a_unit_gives_back_what_its_parameter_lends() {
    let mut case = Case::new();
    let int = case.ty(TyKind::Int(IntTy::I64));
    case.script(&["lib"], "pick", "$r");
    let main = case.script(&["app"], "main", "let a = $n; let r = lib::pick(r: &a); *r");
    case.entry(main, &[("n", int)], Some(int));
    case.admitted();
}

/// The loan a unit's result carries is the caller's: writing what it lends
/// while the result lives is refused, as for a lambda.
#[test]
fn a_units_result_keeps_its_arguments_loan() {
    let mut case = Case::new();
    let int = case.ty(TyKind::Int(IntTy::I64));
    case.script(&["lib"], "pick", "$r");
    let main = case.script(&["app"], "main", "let a = $n; let r = lib::pick(r: &a); a = 0; *r");
    case.entry(main, &[("n", int)], Some(int));
    let refused = case.refused();
    assert!(refused.contains("app::main: LoanConflict"), "{refused}");
}

/// A unit's own storage ends with it: a result lending it is refused.
#[test]
fn a_unit_giving_its_own_storage_is_refused() {
    let mut case = Case::new();
    let int = case.ty(TyKind::Int(IntTy::I64));
    case.script(&["lib"], "own", "let a = $x; &a");
    let main = case.script(&["app"], "main", "*lib::own(x: $n)");
    case.entry(main, &[("n", int)], Some(int));
    let refused = case.refused();
    assert!(refused.contains("lib::own: LoanOutlivesBody"), "{refused}");
}

/// The unit a host runs gives a result with no positions.
#[test]
fn an_entry_whose_result_holds_a_loan_is_refused() {
    let mut case = Case::new();
    let int = case.ty(TyKind::Int(IntTy::I64));
    let shared = case.ty(TyKind::Ref(Mutability::Shared, int));
    let main = case.script(&["app"], "main", "$r");
    case.entry(main, &[("r", shared)], None);
    assert_eq!(case.refused(), "app::main: EntryHoldsLoans app::main");
}

/// A context is storage outside the body, lent only while the body holds
/// it (context §3): a result lending it outlives that, and is refused
/// where the body returns.
#[test]
fn a_unit_giving_a_context_by_reference_is_refused() {
    let mut case = Case::new();
    let int = case.ty(TyKind::Int(IntTy::I64));
    case.context("n", int);
    let main = case.script(&["app"], "main", "&@n");
    case.entry(main, &[], None);
    let refused = case.refused();
    assert!(refused.contains("app::main: LoanOutlivesBody"), "{refused}");
}
