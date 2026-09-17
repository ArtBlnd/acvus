//! A captured name is a borrow of a value the closure owns (RFC-0018), at
//! the checker's contract and at the IR the lowering writes. A test that
//! fails is a finding, kept as it fails.

use acvus_extern::{Externs, TypesOnly};
use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ty::{PolyBuilder, Ty, TyTerm};
use acvus_mir_test::compile_script_mode_optimized;
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

#[derive(Debug)]
struct Checked {
    ret: Ty,
    captures_by_lambda: Vec<Vec<Ty>>,
}

fn script_fn(i: &Interner, source: &str) -> Function {
    let mut pb = PolyBuilder::new();
    Function {
        qref: QualifiedRef::root(i.intern("script")),
        kind: FnKind::Local(ParsedAst::Script(
            acvus_ast::parse_script_mode(i, source).expect("parse"),
        )),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: pb.fresh_effect_var(),
        },
    }
}

fn check(i: &Interner, source: &str) -> Result<Checked, Vec<String>> {
    let Externs {
        mut functions,
        types,
        ..
    } = Externs::combine(acvus_ext::std_registries::<TypesOnly>(), i).expect("registries combine");
    let script = script_fn(i, source);
    let qref = script.qref;
    functions.push(script);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(i, &graph, &ext, &FxHashMap::default(), Freeze::new(types));
    if inf.has_errors() {
        return Err(inf
            .errors()
            .into_iter()
            .flat_map(|(_, errs)| errs.iter().map(|e| e.display(i).to_string()))
            .collect());
    }
    let outcome = &inf.outcomes[&qref];
    let Ty::Fn { ret, .. } = &outcome.meta().ty else {
        panic!("a script is a function: {:?}", outcome.meta().ty)
    };
    let resolution = outcome.resolution().expect("complete");
    let mut captures_by_lambda: Vec<Vec<Ty>> = resolution
        .type_map
        .iter()
        .filter_map(|(_, ty)| match ty {
            Ty::Fn { captures, .. } if !captures.is_empty() => Some(captures.clone()),
            _ => None,
        })
        .collect();
    captures_by_lambda.sort_by_key(|c| format!("{c:?}"));
    Ok(Checked {
        ret: (**ret).clone(),
        captures_by_lambda,
    })
}

fn checked(i: &Interner, source: &str) -> Checked {
    check(i, source).unwrap_or_else(|e| panic!("{}", e.join("\n")))
}

#[test]
fn a_let_bound_lambda_is_called_from_inside_another_lambda() {
    let i = Interner::new();
    let c = checked(&i, "let f = |x| -> x + 1.0; let g = |t| -> f(1.0); g(0)");
    assert_eq!(c.ret, Ty::Float);
}

#[test]
fn a_let_bound_lambda_is_called_from_a_lambda_an_extern_receives() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let f = |x| -> x + 1.0; range(0, 2) | map(|t| -> f(1.0)) | sum",
    );
    assert_eq!(c.ret, Ty::Float);
}

const TWO_LEVEL_WORD: &str = "let h = 0.5; \
     range(0, 2) | map(|j| -> range(0, 2) | map(|t| -> *h * to_float(t)) | sum) | sum";

const TWO_LEVEL_LARGE: &str = "let w = [0.5, 0.5]; \
     range(0, 2) | map(|j| -> range(0, 2) | map(|t| -> *get(w, t)) | sum) | sum";

#[test]
fn a_two_level_capture_records_the_owned_type_at_both_levels() {
    let i = Interner::new();
    let c = checked(&i, TWO_LEVEL_WORD);
    assert_eq!(c.ret, Ty::Float);
    assert_eq!(
        c.captures_by_lambda,
        vec![vec![Ty::Float], vec![Ty::Float]],
        "each lambda captures the value, never the `&T` the name reads as"
    );
}

#[test]
fn a_two_level_capture_of_a_large_value_is_a_move_out_of_the_capture() {
    let i = Interner::new();
    let errors = check(&i, TWO_LEVEL_LARGE).expect_err("the inner lambda moves `w`");
    assert!(
        errors.iter().any(|e| e
            == "cannot move `w` out of a closure's capture (type Array<Float, 2>); \
                act through the reference, or clone it"),
        "{errors:?}"
    );
}

#[test]
fn a_two_level_capture_of_a_word_lowers_to_a_closure_that_owns_its_capture() {
    let i = Interner::new();
    let ir = compile_script_mode_optimized(&i, TWO_LEVEL_WORD, &FxHashMap::default());
    assert!(ir.is_ok(), "{}", ir.unwrap_err());
}

#[test]
fn a_lambda_capturing_a_lent_parameter_is_rejected() {
    let i = Interner::new();
    let errors = check(
        &i,
        "let a = [[0.5, 0.5]]; \
         as_iter(&a) | map(|k| -> range(0, 2) | map(|t| -> *get(k, t)) | sum) | sum",
    )
    .expect_err("a lambda cannot capture a reference");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("a lambda cannot capture a reference")),
        "{errors:?}"
    );
}

#[test]
fn a_captured_lambda_passed_by_value_is_still_a_reference_in_a_value_position() {
    let i = Interner::new();
    let errors = check(
        &i,
        "let f = |x| -> x + 1.0; \
         range(0, 2) | map(|t| -> range(0, 2) | map(f) | sum) | sum",
    )
    .expect_err("a `&Fn` does not meet an `Fn` parameter");
    assert!(
        errors.iter().any(|e| e.contains("&Fn(Float) -> Float")),
        "{errors:?}"
    );
}
