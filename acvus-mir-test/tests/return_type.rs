//! A body's `Return` against the return type the module declares for it.

use acvus_mir::ir::{InstKind, MirBody, MirModule};
use acvus_mir::ty::{Mutability, Ty, TypeArg};
use acvus_mir::validate::validate;
use acvus_mir_test::{compile_script_mode_raw, declared_script_module, lowered_script_module};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn module(source: &str) -> MirModule {
    let i = Interner::new();
    lowered_script_module(&i, source, &[]).expect("compiles")
}

/// The value a body's `Return` leaves with.
fn returned(body: &MirBody) -> acvus_mir::ir::ValueId {
    body.insts
        .iter()
        .find_map(|inst| match inst.kind {
            InstKind::Return { value, .. } => Some(value),
            _ => None,
        })
        .expect("the body returns")
}

/// Leave the body with a reference to what it left with: the one variable
/// between this body and the one the lowering wrote.
fn return_a_reference(body: &mut MirBody) {
    let leaves = body.val_types[&returned(body)].clone();
    let reference = body.val_factory.next();
    body.val_types.insert(
        reference,
        Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(leaves))),
    );
    for inst in body.insts.iter_mut() {
        if let InstKind::Return { value, .. } = &mut inst.kind {
            *value = reference;
        }
    }
}

fn refusals(module: &MirModule) -> Vec<String> {
    let i = Interner::new();
    validate(module)
        .iter()
        .map(|e| e.display(&i).to_string())
        .collect()
}

#[test]
fn a_lambda_returning_an_array_is_accepted() {
    let module = module("let f = |x| -> [x, x, x]; f(1)");
    assert_eq!(module.closures.len(), 1);
    assert!(validate(&module).is_empty(), "{:?}", refusals(&module));
}

#[test]
fn a_lambda_declaring_an_array_returning_a_reference_to_one_is_refused() {
    let mut module = module("let f = |x| -> [x, x, x]; f(1)");
    let label = *module.closures.keys().next().expect("one closure");
    return_a_reference(module.closures.get_mut(&label).expect("the closure body"));

    let refusals = refusals(&module);
    let [refusal] = refusals.as_slice() else {
        panic!("expected one refusal, got {refusals:?}");
    };
    assert!(refusal.starts_with("Return takes value as"), "{refusal}");
    assert!(refusal.contains("Array"), "{refusal}");
    assert!(refusal.contains("&Array"), "{refusal}");
}

#[test]
fn every_lambda_of_a_script_is_checked() {
    let source = "let f = |x| -> x + 1; let g = |y| -> f(y) + 1; g(1)";
    let module = module(source);
    assert_eq!(module.closures.len(), 2);
    for label in module.closures.keys().copied() {
        let mut sabotaged = module.clone();
        return_a_reference(sabotaged.closures.get_mut(&label).expect("closure body"));
        assert_eq!(refusals(&sabotaged).len(), 1, "closure {label:?}");
    }
}

/// RFC-0054: the host declared what `main` returns, so a `main` leaving with
/// something else is refused at `validate` the way a lambda's body is. This
/// is a sabotage probe, made permanent (RFC-0054).
#[test]
fn a_main_is_checked_against_what_the_host_declared() {
    let i = Interner::new();
    let mut module =
        declared_script_module(&i, "let f = |x| -> [x, x, x]; f(1)", &[], array_of_3())
            .expect("compiles");
    return_a_reference(&mut module.main);

    let refusals = refusals(&module);
    let [refusal] = refusals.as_slice() else {
        panic!("expected one refusal, got {refusals:?}");
    };
    assert!(refusal.starts_with("Return takes value as"), "{refusal}");
    assert!(refusal.contains("Array"), "{refusal}");
    assert!(refusal.contains("&Array"), "{refusal}");
}

/// RFC-0054, one variable from `a_main_is_checked_against_what_the_host_declared`:
/// the same sabotage, declared `!` instead of `[i64; 3]`.
#[test]
fn a_main_declared_never_holds_its_return_to_nothing() {
    let i = Interner::new();
    let mut module = declared_script_module(&i, "let f = |x| -> [x, x, x]; f(1)", &[], Ty::Never)
        .expect("compiles");
    assert_eq!(module.ret, Ty::Never);
    assert!(validate(&module).is_empty(), "{:?}", refusals(&module));

    return_a_reference(&mut module.main);
    assert!(validate(&module).is_empty(), "{:?}", refusals(&module));
}

/// The declaration a host states is the type the module carries, not what
/// the body happened to produce.
#[test]
fn the_module_carries_the_declaration_the_host_stated() {
    let i = Interner::new();
    let module = declared_script_module(&i, "1 + 2", &[], Ty::I64).expect("compiles");
    assert_eq!(module.ret, Ty::I64);
}

/// A body returning other than the declaration never reaches `validate`:
/// the checker refuses it first, naming the declared type and the one the
/// body leaves with.
#[test]
fn a_body_returning_other_than_the_declaration_is_refused_by_the_checker() {
    let i = Interner::new();
    let refusal =
        declared_script_module(&i, r#""no".to_string()"#, &[], Ty::I64).expect_err("refused");
    assert!(refusal.contains("expected i64"), "{refusal}");
    assert!(refusal.contains("got String"), "{refusal}");
}

/// RFC-0038: no value of `!` exists, so a diverging body satisfies any
/// declaration.
#[test]
fn a_diverging_body_satisfies_any_declaration() {
    let i = Interner::new();
    let module =
        declared_script_module(&i, r#"panic("no".to_string())"#, &[], Ty::I64).expect("compiles");
    assert!(validate(&module).is_empty(), "{:?}", refusals(&module));
}

fn array_of_3() -> Ty {
    Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3))
}

#[test]
fn a_lambda_whose_body_diverges_is_accepted() {
    let module = module("let f = |x| -> panic(\"no\".to_string()); f(1)");
    let body = module.closures.values().next().expect("one closure");
    assert_eq!(body.val_types[&returned(body)], Ty::Never);
    assert!(validate(&module).is_empty(), "{:?}", refusals(&module));
}

#[test]
fn a_script_whose_tail_is_unit_passes() {
    let module = module("let x = 1; ()");
    assert!(validate(&module).is_empty(), "{:?}", refusals(&module));
}

#[test]
fn a_script_that_leaves_through_a_question_mark_passes() {
    let i = Interner::new();
    let mut ctx = FxHashMap::default();
    ctx.insert(
        i.intern("r"),
        Ty::Result(Box::new(Ty::I64), Box::new(Ty::String)),
    );
    compile_script_mode_raw(&i, "let v = @r?; Ok(v)", &ctx).expect("compiles and validates");
}
