//! An extern fn through the whole compile (extern-fn §1): called, never
//! taken as a value.

use acvus_mir::ty::{Effect, IntTy, Mutability, TyKind};
use acvus_mir_test::Case;

/// `len` named where no call of it stands is refused at the name, printed
/// in the program's names.
#[test]
fn an_extern_fn_taken_as_a_value_is_refused_at_its_name() {
    let mut case = Case::new();
    let int = case.ty(TyKind::Int(IntTy::I64));
    let shared = case.ty(TyKind::Ref(Mutability::Shared, int));
    let len = case.function(&[shared], int, Effect::pure());
    case.declare("len", len);
    let main = case.script(&["app"], "main", "let xs = 1; let f = len; f(&xs)");
    case.entry(main, &[], None);
    assert_eq!(case.refused(), "app::main: ExternAsValue len @20..23");
}
