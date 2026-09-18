//! A body's `Return` against the return type the module declares for it.

use acvus_mir::ir::{InstKind, MirBody, MirModule};
use acvus_mir::ty::{Mutability, Ty, TypeArg};
use acvus_mir::validate::validate;
use acvus_mir_test::{compile_script_mode_raw, lowered_script_module};
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
    assert!(refusal.contains("Ref("), "{refusal}");
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

#[test]
fn a_main_is_not_checked_against_a_return_type_the_module_does_not_hold() {
    let mut module = module("let f = |x| -> [x, x, x]; f(1)");
    return_a_reference(&mut module.main);
    assert!(validate(&module).is_empty(), "{:?}", refusals(&module));
}

#[test]
fn a_lambda_whose_body_diverges_is_accepted() {
    let module = module("let f = |x| -> panic(\"no\"); f(1)");
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
